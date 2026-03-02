"""
Reproducibility and networking utilities shared across all phases.
"""
import random
import socket

import numpy as np
import torch

from src.config import SEED

# ── IPv4 monkey-patch (IPv6 broken on some university machines) ──────────────
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0:
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


def patch_ipv4():
    """Force all socket connections to use IPv4."""
    socket.getaddrinfo = _ipv4_getaddrinfo


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed=SEED, benchmark=False):
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed.
        benchmark: If True, enable cuDNN benchmark mode (faster but slightly
                   non-deterministic). Recommended for ViTs with fixed input size.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark


def make_generator(seed=SEED):
    """Create a seeded torch Generator for DataLoader reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    """DataLoader worker init function for reproducible data loading."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
