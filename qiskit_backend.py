"""Qiskit backend for the same BB84 experiment (runs on Aer, or on IBM hardware by swapping the backend).

NOTE: written against the standard Qiskit API (QuantumCircuit, AerSimulator, backend.run).
The NumPy engine in bb84_engine.py produced all reported results; run `python qiskit_backend.py`
to cross-check that both engines agree on QBER.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def _prepare(qc, value, basis):
    if value:
        qc.x(0)
    if basis:
        qc.h(0)


def _measure(qc, basis):
    if basis:
        qc.h(0)
    qc.measure(0, 0)


def _bits(result, k):
    return int(next(iter(result.get_counts(k))), 2)


def run_session_qiskit(n, rng, eve_fraction=0.0, p_x=0.0, p_z=0.0, backend=None):
    backend = backend or AerSimulator()
    a, ba, bb = (rng.integers(0, 2, n) for _ in range(3))
    be = rng.integers(0, 2, n)
    intercept = rng.random(n) < eve_fraction

    # Stage 1: Alice prepares, Eve measures (only where she intercepts)
    eve_bit = a.copy()
    idx = np.flatnonzero(intercept)
    if len(idx):
        circs = []
        for i in idx:
            qc = QuantumCircuit(1, 1); _prepare(qc, a[i], ba[i]); _measure(qc, be[i]); circs.append(qc)
        res = backend.run(transpile(circs, backend), shots=1).result()
        for k, i in enumerate(idx):
            eve_bit[i] = _bits(res, k)

    # Stage 2: (re)prepared qubit travels through a Pauli-noisy channel, Bob measures
    circs = []
    for i in range(n):
        qc = QuantumCircuit(1, 1)
        if intercept[i]:
            _prepare(qc, eve_bit[i], be[i])
        else:
            _prepare(qc, a[i], ba[i])
        if rng.random() < p_x: qc.x(0)
        if rng.random() < p_z: qc.z(0)
        _measure(qc, bb[i]); circs.append(qc)
    res = backend.run(transpile(circs, backend), shots=1).result()
    bob = np.array([_bits(res, k) for k in range(n)])
    sift = ba == bb
    return a[sift], bob[sift], ba[sift]


if __name__ == "__main__":
    from bb84_engine import run_session
    rng = np.random.default_rng(1)
    for f in (0.0, 0.5, 1.0):
        a, b, _ = run_session_qiskit(400, rng, eve_fraction=f)
        a2, b2, _ = run_session(20000, rng, eve_mask=rng.random(20000) < f)
        print(f"Eve fraction {f}: Qiskit QBER {np.mean(a != b):.3f} | NumPy QBER {np.mean(a2 != b2):.3f} | theory {f/4:.3f}")
