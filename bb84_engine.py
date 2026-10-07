"""Exact BB84 simulator for single-qubit Pauli channels (NumPy, no external deps).

Each qubit is always in a Z- or X-basis eigenstate, so tracking (basis, value) is an
exact simulation of the quantum protocol, not an approximation:
  - Pauli X (bit-flip)   flips Z-basis states, leaves X-basis states unchanged
  - Pauli Z (dephasing)  flips X-basis states, leaves Z-basis states unchanged
  - Pauli Y              flips both
Measuring in the wrong basis gives a uniformly random bit (Born rule).
Basis code: 0 = Z (computational), 1 = X (Hadamard).
"""
import numpy as np


def pauli_errors(n, rng, p_x=0.0, p_z=0.0, p_depol=0.0):
    """Return boolean arrays (x_component, z_component) of the Pauli error on each qubit."""
    dep = rng.random(n) < p_depol
    which = rng.integers(0, 3, n)  # 0=X, 1=Y, 2=Z
    xerr = (rng.random(n) < p_x) ^ (dep & (which < 2))
    zerr = (rng.random(n) < p_z) ^ (dep & (which > 0))
    return xerr, zerr


def run_session(n, rng, eve_mask=None, p_x=0.0, p_z=0.0, p_depol=0.0):
    """One BB84 session of n qubits. eve_mask[i]=True -> Eve does intercept-resend on qubit i.
    Returns Alice's sifted bits, Bob's sifted bits, and the sifted bases (in time order)."""
    a = rng.integers(0, 2, n)
    ba = rng.integers(0, 2, n)
    bb = rng.integers(0, 2, n)
    state_b, state_v = ba.copy(), a.copy()
    if eve_mask is not None and eve_mask.any():
        be = rng.integers(0, 2, n)
        eve_bit = np.where(be == ba, a, rng.integers(0, 2, n))
        state_b = np.where(eve_mask, be, ba)
        state_v = np.where(eve_mask, eve_bit, a)
    xerr, zerr = pauli_errors(n, rng, p_x, p_z, p_depol)
    flip = np.where(state_b == 0, xerr, zerr)
    state_v = state_v ^ flip.astype(int)
    bob = np.where(bb == state_b, state_v, rng.integers(0, 2, n))
    sift = ba == bb
    return a[sift], bob[sift], ba[sift]


def session_features(alice, bob, basis, rng, test_frac=0.5, blocks=8):
    """Features computed ONLY from the publicly disclosed test sample (as in real BB84)."""
    m = len(alice)
    idx = np.sort(rng.choice(m, int(m * test_frac), replace=False))
    err = (alice[idx] != bob[idx]).astype(float)
    bs = basis[idx]
    q = err.mean()
    qz, qx = err[bs == 0].mean(), err[bs == 1].mean()
    bq = np.array([b.mean() for b in np.array_split(err, blocks)])
    return np.array([q, qz, qx, abs(qz - qx), bq.std(), bq.max()])


FEATURE_NAMES = ["QBER", "QBER_Z", "QBER_X", "|QBER_Z-QBER_X|", "block_std", "block_max"]
