"""Diagnosis posterior over (cause, s) and expected information gain (design §4, D89)."""

from __future__ import annotations

import numpy as np

from hpcdiag.layer3.probes import CAUSES, HYPOTHESES


def prior(hyp: np.ndarray, pi0_h0: float) -> np.ndarray:
    """State prior: pi0_h0 on h0; causes uniform; s uniform within a cause (D89)."""
    n_c = len(CAUSES)
    out = np.empty(len(hyp))
    for i in range(len(HYPOTHESES)):
        sel = hyp == i
        out[sel] = (pi0_h0 if i == 0 else (1 - pi0_h0) / n_c) / sel.sum()
    return out


def marginal(pi: np.ndarray, hyp: np.ndarray) -> np.ndarray:
    """Posterior of the hypotheses, state weights summed. Works on the last axis."""
    out = np.zeros((*pi.shape[:-1], len(HYPOTHESES)))
    for i in range(len(HYPOTHESES)):
        out[..., i] = pi[..., hyp == i].sum(axis=-1)
    return out


def entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis."""
    q = np.clip(p, 1e-300, None)
    return -(p * np.log(q)).sum(axis=-1)


def update_gauss(pi: np.ndarray, means: np.ndarray, sigma: float, y: float) -> np.ndarray:
    """Bayes update on one log x = y. means: mean per state for that probe."""
    ll = -0.5 * ((y - means) / sigma) ** 2
    w = pi * np.exp(ll - ll.max())
    return w / w.sum()


def update_tel(pi: np.ndarray, hyp: np.ndarray, conf: np.ndarray, o: int) -> np.ndarray:
    """Bayes update on telemetry class o; conf[h, o] = P(o | h)."""
    w = pi * conf[hyp, o]
    return w / w.sum()


def expected_entropy_gauss(
    pi: np.ndarray,
    means: np.ndarray,
    sigma: float,
    hyp: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> float:
    """E_x H(pi | x) for a Gaussian probe, with Gauss-Hermite nodes z, weights w (sum 1)."""
    keep = pi > 1e-10
    pi_k, mu, hy = pi[keep], means[keep], hyp[keep]
    x = mu[:, None] + sigma * z[None, :]  # (S, J)
    ll = -0.5 * ((x[:, :, None] - mu[None, None, :]) / sigma) ** 2  # (S, J, S)
    post = pi_k * np.exp(ll - ll.max(axis=2, keepdims=True))
    post /= post.sum(axis=2, keepdims=True)
    return float(pi_k @ (entropy(marginal(post, hy)) @ w))


def expected_entropy_tel(pi: np.ndarray, hyp: np.ndarray, conf: np.ndarray) -> float:
    """E_o H(pi | o) for the telemetry probe."""
    h = marginal(pi, hyp)
    joint = h[:, None] * conf  # (hyp, o)
    po = joint.sum(axis=0)
    post = joint / np.maximum(po, 1e-300)
    return float(po @ entropy(post.T))
