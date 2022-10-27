"""
PyTorch PCA implementation
"""

import torch


def pca(x: torch.Tensor, k: int, center=True):
    if center:
        m = x.mean(0, keepdim=True)
        s = x.std(0, unbiased=False, keepdim=True)
        x -= m
        x /= s
    # why pca related to svd? https://www.cs.cmu.edu/~elaw/papers/pca.pdf chap VI
    U, S, V = torch.linalg.svd(x)
    reduced = torch.mm(x, V[:k].T)

    return reduced
