import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(
    page_title="QuantumGuard | BB84 Security Monitor",
    page_icon="⚛️",
    layout="wide"
)

# ---------- BB84 simulator ----------
def simulate_session(n_qubits=2000, noise=0.03, eve_fraction=0.0, burst=False, seed=42):
    rng = np.random.default_rng(seed)

    alice_bits = rng.integers(0, 2, n_qubits)
    alice_basis = rng.integers(0, 2, n_qubits)
    bob_basis = rng.integers(0, 2, n_qubits)

    sift = alice_basis == bob_basis
    sift_count = max(int(sift.sum()), 1)

    errors = np.zeros(n_qubits, dtype=bool)

    # Hardware/channel noise
    errors |= rng.random(n_qubits) < noise

    # Intercept-resend Eve: approximately 25% error on sifted bits
    if eve_fraction > 0:
        attacked = rng.random(n_qubits) < eve_fraction
        basis_mismatch = rng.random(n_qubits) < 0.5
        eve_error = attacked & basis_mismatch
        if burst:
            start = int(n_qubits * 0.35)
            end = min(start + int(n_qubits * 0.20), n_qubits)
            mask = np.zeros(n_qubits, dtype=bool)
            mask[start:end] = True
            eve_error &= mask
        errors |= eve_error

    sift_errors = errors[sift]
    sift_basis = alice_basis[sift]

    qber = float(sift_errors.mean())
    z_mask = sift_basis == 0
    x_mask = sift_basis == 1

    z_qber = float(sift_errors[z_mask].mean()) if z_mask.any() else 0.0
    x_qber = float(sift_errors[x_mask].mean()) if x_mask.any() else 0.0

    # Block spread
    block_size = max(sift_count // 10, 1)
    block_rates = []
    for i in range(0, sift_count, block_size):
        b = sift_errors[i:i + block_size]
        if len(b):
            block_rates.append(float(b.mean()))

    block_spread = float(np.std(block_rates)) if block_rates else 0.0
    basis_asymmetry = abs(z_qber - x_qber)

    return {
        "QBER": qber,
        "Z-basis QBER": z_qber,
        "X-basis QBER": x_qber,
        "Basis asymmetry": basis_asymmetry,
        "Block spread": block_spread,
        "sifted": sift_count
    }


# ---------- Synthetic training data ----------
@st.cache_resource
def train_detector():
    rng = np.random.default_rng(7)
    rows, labels = [], []

    for _ in range(1800):
        # Honest/noisy sessions
        noise = rng.uniform(0.0, 0.10)
        r = simulate_session(
            n_qubits=1200,
            noise=noise,
            eve_fraction=0.0,
            seed=int(rng.integers(1_000_000_000))
        )
        rows.append([
            r["QBER"], r["Z-basis QBER"], r["X-basis QBER"],
            r["Basis asymmetry"], r["Block spread"]
        ])
        labels.append(0)

        # Eve sessions
        noise = rng.uniform(0.0, 0.06)
        eve = rng.uniform(0.08, 0.75)
        burst = bool(rng.integers(0, 2))
        r = simulate_session(
            n_qubits=1200,
            noise=noise,
            eve_fraction=eve,
            burst=burst,
            seed=int(rng.integers(1_000_000_000))
        )
        rows.append([
            r["QBER"], r["Z-basis QBER"], r["X-basis QBER"],
            r["Basis asymmetry"], r["Block spread"]
        ])
        labels.append(1)

    X = np.array(rows)
    y = np.array(labels)

    model = GradientBoostingClassifier(random_state=7)
    model.fit(X, y)
    return model


model = train_detector()

# ---------- UI ----------
st.title("⚛️ QuantumGuard")
st.subheader("Telling an eavesdropper from hardware noise in BB84 quantum key distribution")
st.caption("Q-Hack India 2026 • Team Quantum Syndicate")

with st.sidebar:
    st.header("Simulation Controls")
    n_qubits = st.slider("Number of qubits", 500, 5000, 2000, 100)
    noise_pct = st.slider("Channel noise (%)", 0.0, 15.0, 3.0, 0.5)
    eve_pct = st.slider("Eavesdropper interception (%)", 0.0, 80.0, 0.0, 1.0)
    burst = st.checkbox("Burst attack", value=False)
    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42)

    run = st.button("Run QuantumGuard", type="primary", use_container_width=True)

if "result" not in st.session_state or run:
    result = simulate_session(
        n_qubits=n_qubits,
        noise=noise_pct / 100,
        eve_fraction=eve_pct / 100,
        burst=burst,
        seed=int(seed)
    )
    features = np.array([[
        result["QBER"],
        result["Z-basis QBER"],
        result["X-basis QBER"],
        result["Basis asymmetry"],
        result["Block spread"]
    ]])
    probability = float(model.predict_proba(features)[0, 1])
    verdict = "EAVESDROPPER DETECTED" if probability >= 0.5 else "HONEST / NOISE-DOMINATED"
    st.session_state.result = result
    st.session_state.probability = probability
    st.session_state.verdict = verdict

r = st.session_state.result
probability = st.session_state.probability
verdict = st.session_state.verdict

if probability >= 0.5:
    st.error(f"🚨 {verdict}")
else:
    st.success(f"✅ {verdict}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall QBER", f"{r['QBER']*100:.2f}%")
c2.metric("Z-basis QBER", f"{r['Z-basis QBER']*100:.2f}%")
c3.metric("X-basis QBER", f"{r['X-basis QBER']*100:.2f}%")
c4.metric("Eve probability", f"{probability*100:.1f}%")

st.divider()

left, right = st.columns(2)

with left:
    st.markdown("### Detection Features")
    df = pd.DataFrame({
        "Feature": [
            "Overall QBER",
            "Z-basis QBER",
            "X-basis QBER",
            "Basis asymmetry",
            "Block spread"
        ],
        "Value": [
            r["QBER"],
            r["Z-basis QBER"],
            r["X-basis QBER"],
            r["Basis asymmetry"],
            r["Block spread"]
        ]
    })
    df["Value"] = (df["Value"] * 100).round(3)
    df = df.rename(columns={"Value": "Percentage"})
    st.dataframe(df, use_container_width=True, hide_index=True)

with right:
    st.markdown("### Security Decision")
    st.write(
        "QuantumGuard uses publicly disclosed test-bit statistics to distinguish "
        "ordinary channel noise from an intercept-resend eavesdropping pattern."
    )

    if r["QBER"] > 0.11:
        st.warning("Traditional 11% QBER rule: ABORT")
    else:
        st.info("Traditional 11% QBER rule: KEEP")

    if probability >= 0.5:
        st.warning("QuantumGuard: ABORT — suspicious eavesdropping pattern detected.")
    else:
        st.success("QuantumGuard: KEEP — channel appears noise-dominated.")

st.divider()

st.markdown("### How the prototype works")
st.markdown("""
**Alice → BB84 quantum channel → Bob → Sifting → QBER features → ML detector → Verdict**

- **Quantum layer:** BB84-style preparation, basis selection and measurement.
- **Noise layer:** simulated channel noise.
- **Attack layer:** optional intercept-resend Eve, including burst attacks.
- **Classical layer:** QBER, per-basis QBER, basis asymmetry and block spread.
- **ML layer:** GradientBoostingClassifier.
""")

st.caption(
    "Prototype/demo only. The current implementation is a simulation and is not a "
    "production QKD security system or a replacement for formal security proofs."
)
