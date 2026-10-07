QuantumGuard — Q-Hack India 2026
Team Quantum Syndicate · Team Lead: Pradeep PK · Track: Quantum Security & Cryptography
Problem statement: Telling an eavesdropper apart from ordinary hardware noise in BB84 quantum key distribution.
The idea
BB84 detects an eavesdropper because measuring a qubit disturbs it (no-cloning). But real channels are noisy, so the classical abort rule ("stop if QBER > 11%") either misses short or weak attacks or throws away good keys. QuantumGuard is a hybrid pipeline:
Quantum layer – BB84 sessions with intercept-resend attackers and Pauli noise (bit-flip, dephasing, depolarizing). Exact NumPy engine (bb84_engine.py) plus a Qiskit/Aer circuit backend (qiskit_backend.py).
Classical layer – a gradient-boosting classifier reads only the publicly disclosed test sample (overall QBER, per-basis QBER, basis asymmetry, per-block QBER spread/max) and flags Eve vs honest noise.
Run it
pip install -r requirements.txt
python train_detector.py        # simulates 9,000 sessions, trains, writes results/ (metrics.json + 3 plots)
python qiskit_backend.py        # cross-check: Qiskit circuits vs NumPy engine vs theory (QBER = f/4)
Results (held-out 3,000 sessions, 2,000 qubits each, seed 7)

QBER > 11% rule
QuantumGuard
Accuracy
87.6%
96.4%
Eve detection rate
74.9%
95.5%
False-alarm rate
0.2%
2.8%
ROC AUC
0.977
0.994
By attacker type (fresh sessions): steady intercept-resend 83.3% → 91.5%; burst attacker (taps one window of the session) 66.4% → 99.4%.
Honest limitations
Results are on simulated data from the threat model above; we have not tested on real IBM hardware yet.
Most of the gain comes from catching burst attacks (the per-block features). Steady, weak attackers hiding inside depolarizing noise remain hard (see results/detection_by_eve_fraction.png).
The ML detector trades a higher false-alarm rate (2.8% vs 0.2%) for much higher detection.
BB84 at this scale is classically simulable; the value here is the security monitoring layer, not a quantum speed-up.
References
Bennett & Brassard, "Quantum cryptography: Public key distribution and coin tossing" (BB84), 1984
Shor, "Algorithms for quantum computation: discrete logarithms and factoring", 1994
Qiskit documentation: https://docs.quantum.ibm.com
scikit-learn GradientBoostingClassifier documentation