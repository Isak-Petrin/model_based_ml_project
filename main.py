"""
AR-HMM for Handwritten Digit Stroke Decomposition
===================================================
A probabilistic graphical model that decomposes handwritten digit
trajectories into latent dynamical regimes using Pyro.

Generative story:
    1. Draw initial regime:      z_1 ~ Categorical(pi_0)
    2. For t = 2, ..., T:        z_t | z_{t-1} ~ Categorical(A_{z_{t-1}})
    3. For t = 1, ..., T:        delta_t | delta_{t-1}, z_t ~ N(mu_{z_t} + Phi_{z_t} * delta_{t-1}, Sigma_{z_t})

Uses pyro.markov to enable efficient forward-algorithm enumeration
over the discrete latent regimes (avoids exponential blowup).
"""

import re
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.optim import Adam
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import
import matplotlib.pyplot as plt


import matplotlib.cm as cm


# =============================================================================
# 1. DATA PARSING
# =============================================================================

def parse_uji_penchar(filepath, chars_to_keep=None):
    """
    Parse the UJI Pen Characters dataset.

    Args:
        filepath: path to ujipenchars2.txt
        chars_to_keep: list of characters to extract, e.g. ['0','1','2',...]
                       If None, extracts all digits '0'-'9'.

    Returns:
        samples: list of dicts with keys:
            'label': character label (str)
            'writer': writer ID (str)
            'strokes': list of numpy arrays, each (N_points, 2)
            'trajectory': full concatenated (x, y) array
            'offsets': computed (delta_x, delta_y) array
    """
    if chars_to_keep is None:
        chars_to_keep = [str(d) for d in range(10)]

    samples = []
    current_char = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track which character section we're in
        if line.startswith('// ASCII char:'):
            current_char = line.split(':')[1].strip()
            i += 1
            continue

        # Parse a WORD entry
        if line.startswith('WORD'):
            parts = line.split()
            label = parts[1]
            writer = parts[2] if len(parts) > 2 else 'unknown'

            # Extract writer ID (e.g., 'W01' from 'trn_UJI_W01-02')
            writer_match = re.search(r'W(\d+)', writer)
            writer_id = f'W{writer_match.group(1)}' if writer_match else writer

            i += 1
            # Read NUMSTROKES
            num_strokes_line = lines[i].strip()
            num_strokes = int(num_strokes_line.split()[1])

            strokes = []
            for s in range(num_strokes):
                i += 1
                points_line = lines[i].strip()
                # Format: POINTS N # x1 y1 x2 y2 ...
                after_hash = points_line.split('#')[1].strip()
                coords = list(map(float, after_hash.split()))
                # Reshape into (N, 2)
                points = np.array(coords).reshape(-1, 2)
                strokes.append(points)

            if label in chars_to_keep:
                # Concatenate all strokes into one trajectory
                trajectory = np.concatenate(strokes, axis=0)

                # Compute offsets: delta_t = pos_t - pos_{t-1}
                offsets = np.diff(trajectory, axis=0)

                samples.append({
                    'label': label,
                    'writer': writer_id,
                    'strokes': strokes,
                    'trajectory': trajectory,
                    'offsets': offsets,
                })

        i += 1

    return samples


def normalize_offsets(samples):
    """
    Normalize offsets to zero mean and unit variance across the dataset.
    Returns the mean and std for later denormalization.
    """
    all_offsets = np.concatenate([s['offsets'] for s in samples], axis=0)
    mean = all_offsets.mean(axis=0)
    std = all_offsets.std(axis=0) + 1e-8

    for s in samples:
        s['offsets_norm'] = (s['offsets'] - mean) / std

    return mean, std


# =============================================================================
# 2. PYRO AR-HMM MODEL (using pyro.markov for efficient enumeration)
# =============================================================================

class ARHMM:
    """
    Autoregressive Hidden Markov Model in Pyro.

    Key design: pyro.markov(range(T)) tells Pyro that z_t forms a
    first-order Markov chain. This enables the forward algorithm for
    O(T * K^2) enumeration instead of O(K^T) naive enumeration.
    """

    def __init__(self, K, obs_dim=2):
        """
        Args:
            K: number of regimes
            obs_dim: dimension of observations (2 for x, y offsets)
        """
        self.K = K
        self.obs_dim = obs_dim
        pyro.clear_param_store()

    @config_enumerate
    def model(self, sequences):
        """
        Generative model for a batch of sequences.

        Args:
            sequences: list of tensors, each (T_i, obs_dim)
        """
        K = self.K
        obs_dim = self.obs_dim
        num_sequences = len(sequences)

        # ------- Global parameters (MAP via pyro.param) -------

        # Initial regime distribution: simplex-constrained
        pi_0 = pyro.param(
            'pi_0',
            torch.ones(K) / K,
            constraint=dist.constraints.simplex
        )

        # Transition matrix: each row sums to 1
        transition_probs = pyro.param(
            'transition_probs',
            torch.ones(K, K) / K,
            constraint=dist.constraints.simplex
        )

        # Per-regime emission means: (K, obs_dim)
        mu = pyro.param('mu', torch.randn(K, obs_dim) * 0.5)

        # Per-regime AR(1) coefficient matrices: (K, obs_dim, obs_dim)
        Phi = pyro.param(
            'Phi',
            0.1 * torch.eye(obs_dim).unsqueeze(0).repeat(K, 1, 1)
        )

        # Per-regime emission scales: (K, obs_dim), must be positive
        sigma = pyro.param(
            'sigma',
            torch.ones(K, obs_dim) * 0.5,
            constraint=dist.constraints.positive
        )

        # ------- Per-sequence -------
        for seq_idx in pyro.plate('sequences', num_sequences):
            obs = sequences[seq_idx]
            T = obs.shape[0]
            z_prev = None
            delta_prev = torch.zeros(obs_dim)

            # pyro.markov tells Pyro this is a Markov chain,
            # enabling efficient forward-algorithm enumeration
            for t in pyro.markov(range(T)):

                # --- Draw regime z_t ---
                if z_prev is None:
                    probs = pi_0
                else:
                    probs = transition_probs[z_prev]

                z_t = pyro.sample(
                    f'z_{seq_idx}_{t}',
                    dist.Categorical(probs),
                )

                # --- Draw observation delta_t ---
                # AR(1): delta_t ~ N(mu_k + Phi_k @ delta_{t-1}, sigma_k)
                # matmul handles batch dimensions from enumerated z_t
                emission_mean = mu[z_t] + torch.matmul(
                    Phi[z_t], delta_prev.unsqueeze(-1)
                ).squeeze(-1)
                emission_scale = sigma[z_t]

                pyro.sample(
                    f'obs_{seq_idx}_{t}',
                    dist.Normal(emission_mean, emission_scale).to_event(1),
                    obs=obs[t]
                )

                z_prev = z_t
                delta_prev = obs[t]

    def guide(self, sequences):
        """
        Guide is empty because:
        - Continuous parameters use pyro.param (MAP estimation)
        - Discrete z_t variables are enumerated automatically
        """
        pass


# =============================================================================
# 3. TRAINING
# =============================================================================

def train(model_instance, sequences, num_steps=300, lr=0.005):
    """
    Train the AR-HMM using SVI with enumeration over discrete latents.
    """
    optimizer = Adam({'lr': lr})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    svi = SVI(model_instance.model, model_instance.guide, optimizer, loss=elbo)

    losses = []
    for step in range(num_steps):
        loss = svi.step(sequences)
        losses.append(loss)
        if step % 50 == 0:
            print(f'  Step {step:4d} | ELBO loss: {loss:.1f}')

    return losses


# =============================================================================
# 4. INFERENCE: Viterbi decoding via infer_discrete
# =============================================================================

def decode_regimes(model_instance, sequences):
    """
    Use Pyro's infer_discrete to get MAP regime assignments
    for each sequence via the Viterbi algorithm.

    Args:
        model_instance: trained ARHMM
        sequences: list of tensors

    Returns:
        all_regimes: list of numpy arrays, one per sequence
    """
    # infer_discrete with temperature=0 gives Viterbi (MAP) decoding
    decoded_model = infer_discrete(
        model_instance.model,
        temperature=0,
        first_available_dim=-2
    )

    # Run the decoded model to get a trace with MAP z values
    trace = poutine.trace(decoded_model).get_trace(sequences)

    # Extract regime assignments per sequence
    all_regimes = []
    for seq_idx in range(len(sequences)):
        T = sequences[seq_idx].shape[0]
        regimes = []
        for t in range(T):
            site_name = f'z_{seq_idx}_{t}'
            z_t = trace.nodes[site_name]['value'].item()
            regimes.append(z_t)
        all_regimes.append(np.array(regimes))

    return all_regimes


# =============================================================================
# 5. SAMPLING: Generate new digits from the learned model
# =============================================================================

def sample_trajectory(model_instance, T=50):
    """
    Ancestrally sample a new trajectory from the learned AR-HMM.
    """
    K = model_instance.K
    obs_dim = model_instance.obs_dim

    pi_0 = pyro.param('pi_0').detach().numpy()
    A = pyro.param('transition_probs').detach().numpy()
    mu = pyro.param('mu').detach().numpy()
    Phi = pyro.param('Phi').detach().numpy()
    sigma = pyro.param('sigma').detach().numpy()

    offsets = np.zeros((T, obs_dim))
    regimes = np.zeros(T, dtype=int)

    # Sample z_1
    regimes[0] = np.random.choice(K, p=pi_0)

    # Sample delta_1
    delta_prev = np.zeros(obs_dim)
    k = regimes[0]
    mean = mu[k] + Phi[k] @ delta_prev
    offsets[0] = np.random.multivariate_normal(mean, np.diag(sigma[k] ** 2))

    # Sample forward through the generative story
    for t in range(1, T):
        regimes[t] = np.random.choice(K, p=A[regimes[t - 1]])
        k = regimes[t]
        delta_prev = offsets[t - 1]
        mean = mu[k] + Phi[k] @ delta_prev
        offsets[t] = np.random.multivariate_normal(mean, np.diag(sigma[k] ** 2))

    # Accumulate offsets into absolute positions
    trajectory = np.cumsum(offsets, axis=0)

    return trajectory, regimes, offsets


# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def plot_segmented_trajectory(trajectory, regimes, K, title='', ax=None):
    """Plot a pen trajectory color-coded by inferred regime."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    colors = cm.tab10(np.linspace(0, 1, K))

    for t in range(len(trajectory) - 1):
        ax.plot(
            trajectory[t:t + 2, 0],
            trajectory[t:t + 2, 1],
            color=colors[regimes[t]],
            linewidth=2,
            solid_capstyle='round'
        )

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title)

    for k in range(K):
        ax.plot([], [], color=colors[k], linewidth=3, label=f'Regime {k}')
    ax.legend(loc='upper right', fontsize=8)

    return ax


def plot_training_loss(losses, title='ELBO loss during training'):
    """Plot the training loss curve."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(losses)
    ax.set_xlabel('SVI step')
    ax.set_ylabel('ELBO loss')
    ax.set_title(title)
    plt.tight_layout()
    return fig


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    # --- Configuration ---
    DATA_PATH = 'ujipenchars2.txt'   # <-- adjust path as needed
    DIGIT = 'o'                       # which digit to model
    K = 7                            # number of regimes
    NUM_STEPS = 2000                   # SVI training steps
    LR = 0.005                        # learning rate

    print(f'=== AR-HMM for digit "{DIGIT}" with K={K} regimes ===\n')

    # --- Parse data ---
    print('Parsing dataset...')
    samples = parse_uji_penchar(DATA_PATH, chars_to_keep=[DIGIT])
    print(f'  Found {len(samples)} samples for digit "{DIGIT}"')

    # Normalize offsets
    offset_mean, offset_std = normalize_offsets(samples)

    # Convert to torch tensors
    sequences = [
        torch.tensor(s['offsets_norm'], dtype=torch.float32)
        for s in samples
    ]

    # Use a subset for faster training
    sequences_train = sequences[:100]
    print(f'  Using {len(sequences_train)} sequences for training')
    print(f'  Sequence lengths: {[s.shape[0] for s in sequences_train[:5]]}...\n')

    # --- Build model ---
    print('Building AR-HMM model...')
    arhmm = ARHMM(K=K, obs_dim=2)

    # --- Train ---
    print('Training via SVI with enumerated discrete variables...')
    losses = train(arhmm, sequences_train, num_steps=NUM_STEPS, lr=LR)
    print(f'  Final loss: {losses[-1]:.1f}\n')

    # --- Plot training loss ---
    fig_loss = plot_training_loss(losses)
    fig_loss.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    print('  Saved training_loss.png')

    # --- Infer regimes ---
    print('\nDecoding regimes via Viterbi...')
    all_regimes = decode_regimes(arhmm, sequences_train)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(min(6, len(sequences_train))):
        traj = samples[i]['trajectory']
        # Regimes are for offsets (T-1 points), prepend first regime
        regime_seq = all_regimes[i]
        regime_extended = np.concatenate([[regime_seq[0]], regime_seq])

        plot_segmented_trajectory(
            traj, regime_extended, K,
            title=f'Digit "{DIGIT}" — sample {i} ({samples[i]["writer"]})',
            ax=axes[i]
        )

    plt.tight_layout()
    fig.savefig('inferred_regimes.png', dpi=150, bbox_inches='tight')
    print('  Saved inferred_regimes.png')

    # --- Generate new samples ---
    print('\nGenerating new trajectories by ancestral sampling...')
    fig_gen, axes_gen = plt.subplots(1, 4, figsize=(16, 4))

    for i in range(4):
        median_len = int(np.median([s['offsets'].shape[0] for s in samples]))
        traj, regimes, offsets = sample_trajectory(arhmm, T=median_len)
        plot_segmented_trajectory(
            traj, regimes, K,
            title=f'Generated sample {i}',
            ax=axes_gen[i]
        )

    plt.tight_layout()
    fig_gen.savefig('generated_samples.png', dpi=150, bbox_inches='tight')
    print('  Saved generated_samples.png')

    # --- Print learned parameters ---
    print('\n=== Learned parameters ===')
    print(f'\nInitial distribution pi_0:')
    print(f'  {pyro.param("pi_0").detach().numpy().round(3)}')
    print(f'\nTransition matrix A:')
    print(f'  {pyro.param("transition_probs").detach().numpy().round(3)}')
    print(f'\nPer-regime means mu (in normalized space):')
    mu = pyro.param('mu').detach().numpy()
    for k in range(K):
        denorm_mu = mu[k] * offset_std + offset_mean
        print(f'  Regime {k}: mu = {mu[k].round(3)} '
              f'(denormalized: dx={denorm_mu[0]:.1f}, dy={denorm_mu[1]:.1f})')

    print('\nDone!')


if __name__ == '__main__':
    main()