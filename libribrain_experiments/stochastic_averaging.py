import numpy as np
import torch


def sample_n(batch_size, n_min=50, n_max=100):
    """Sample averaging counts from a distribution with mode at n_min, tail toward n_max.

    Samples uniformly in noise-std space (u = 1/sqrt(N)), giving p(N) ~ N^{-3/2}.
    """
    u = np.random.uniform(1 / np.sqrt(n_max), 1 / np.sqrt(n_min), size=batch_size)
    n = np.round(1 / u ** 2).astype(int)
    return np.clip(n, n_min, n_max)


def average_trials(raw_trials, n_samples, channels_per_sample):
    """Randomly select n_samples trials from raw_trials and average them.

    Args:
        raw_trials: (B, n_max * C, T) concatenated raw trials
        n_samples: int or (B,) array of per-example sample counts
        channels_per_sample: C, number of channels per trial

    Returns:
        averaged: (B, C, T)
    """
    B, total_channels, T = raw_trials.shape
    n_max = total_channels // channels_per_sample
    raw = raw_trials.view(B, n_max, channels_per_sample, T)

    if isinstance(n_samples, (int, np.integer)):
        idx = torch.randperm(n_max, device=raw_trials.device)[:n_samples]
        return raw[:, idx].mean(dim=1)

    # Per-example N: average each example independently
    out = torch.zeros(B, channels_per_sample, T, device=raw_trials.device, dtype=raw_trials.dtype)
    for i, n in enumerate(n_samples):
        idx = torch.randperm(n_max, device=raw_trials.device)[:int(n)]
        out[i] = raw[i, idx].mean(dim=0)
    return out
