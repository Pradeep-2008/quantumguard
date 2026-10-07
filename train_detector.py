"""Generate BB84 sessions (honest noisy channel vs Eve), train the classical detector,
compare against the standard fixed-QBER-threshold rule, and save plots + metrics."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from bb84_engine import run_session, session_features, FEATURE_NAMES

SEED, N_QUBITS, N_TRAIN, N_TEST, QBER_ABORT = 7, 2000, 6000, 3000, 0.11


def sample_noise(rng):
    kind = rng.choice(["bitflip", "dephase", "depol", "mixed"])
    if kind == "bitflip":
        return dict(p_x=rng.uniform(0, 0.16))
    if kind == "dephase":
        return dict(p_z=rng.uniform(0, 0.16))
    if kind == "depol":
        return dict(p_depol=rng.uniform(0, 0.12))
    return dict(p_x=rng.uniform(0, 0.08), p_z=rng.uniform(0, 0.08), p_depol=rng.uniform(0, 0.05))


def sample_eve(rng, n):
    """Intercept-resend attacker. Mode 'steady': each qubit intercepted with prob f.
    Mode 'burst': Eve taps one contiguous window of the session."""
    if rng.random() < 0.5:
        f = rng.uniform(0.15, 1.0)
        return rng.random(n) < f, f
    frac = rng.uniform(0.15, 0.6)
    start = rng.integers(0, int(n * (1 - frac)) + 1)
    mask = np.zeros(n, bool)
    mask[start:start + int(n * frac)] = True
    return mask, frac


def make_dataset(count, rng):
    X, y, f_used = [], [], []
    for _ in range(count):
        noise = sample_noise(rng)
        eve = rng.random() < 0.5
        mask, f = sample_eve(rng, N_QUBITS) if eve else (None, 0.0)
        a, b, bs = run_session(N_QUBITS, rng, eve_mask=mask, **noise)
        X.append(session_features(a, b, bs, rng)); y.append(int(eve)); f_used.append(f)
    return np.array(X), np.array(y), np.array(f_used)


def attack_mode_breakdown(clf, count=1500):
    """Detection rate separately for steady vs burst attackers (fresh sessions, separate RNG)."""
    r = np.random.default_rng(99); out = {}
    for mode in ("steady", "burst"):
        X = []
        for _ in range(count):
            noise = sample_noise(r)
            if mode == "steady":
                mask = r.random(N_QUBITS) < r.uniform(0.15, 1.0)
            else:
                fr = r.uniform(0.15, 0.6); st = r.integers(0, int(N_QUBITS * (1 - fr)) + 1)
                mask = np.zeros(N_QUBITS, bool); mask[st:st + int(N_QUBITS * fr)] = True
            a, b, bs = run_session(N_QUBITS, r, eve_mask=mask, **noise)
            X.append(session_features(a, b, bs, r))
        X = np.array(X)
        out[mode] = dict(ml_detection=float(clf.predict(X).mean()), baseline_detection=float((X[:, 0] > QBER_ABORT).mean()))
    return out


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr, _ = make_dataset(N_TRAIN, rng)
    Xte, yte, fte = make_dataset(N_TEST, rng)

    clf = GradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred_ml = p > 0.5
    pred_base = Xte[:, 0] > QBER_ABORT

    def rates(pred):
        return dict(accuracy=float((pred == yte).mean()),
                    detection_rate=float(pred[yte == 1].mean()),
                    false_alarm_rate=float(pred[yte == 0].mean()))

    metrics = dict(
        setup=dict(qubits_per_session=N_QUBITS, train_sessions=N_TRAIN, test_sessions=N_TEST, seed=SEED),
        baseline_qber_threshold_11pct=rates(pred_base),
        ml_detector=rates(pred_ml),
        auc_qber_only=float(roc_auc_score(yte, Xte[:, 0])),
        auc_ml=float(roc_auc_score(yte, p)),
        feature_importance=dict(zip(FEATURE_NAMES, map(float, clf.feature_importances_))),
    )
    # detection by Eve strength (only Eve sessions)
    bins = [0.15, 0.3, 0.45, 0.6, 0.8, 1.0001]
    labels, d_ml, d_base = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (yte == 1) & (fte >= lo) & (fte < hi)
        labels.append(f"{lo:.2f}-{min(hi,1):.2f}")
        d_ml.append(float(pred_ml[m].mean())); d_base.append(float(pred_base[m].mean()))
    metrics["detection_by_eve_fraction"] = dict(bins=labels, ml=d_ml, baseline=d_base)
    metrics["by_attack_mode"] = attack_mode_breakdown(clf)
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)

    # plots
    fig, ax = plt.subplots(figsize=(5, 4))
    for score, name in [(Xte[:, 0], "QBER threshold only"), (p, "Hybrid ML detector")]:
        fpr, tpr, _ = roc_curve(yte, score)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(yte, score):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("False alarm rate"); ax.set_ylabel("Eavesdropper detection rate")
    ax.set_title("ROC on held-out BB84 sessions"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig("results/roc.png", dpi=160)

    fig, ax = plt.subplots(figsize=(5.5, 4)); w = 0.38; xs = np.arange(len(labels))
    ax.bar(xs - w/2, d_base, w, label="QBER > 11% rule", color="#9aa0a6"); ax.bar(xs + w/2, d_ml, w, label="Hybrid ML detector", color="#7c4dff")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.set_xlabel("Fraction of session Eve intercepts"); ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Detection vs attack strength"); fig.tight_layout(); fig.savefig("results/detection_by_eve_fraction.png", dpi=160)

    fig, ax = plt.subplots(figsize=(5, 4)); hon, eve = yte == 0, yte == 1
    ax.scatter(Xte[hon, 1], Xte[hon, 2], s=6, alpha=.4, label="Honest (noisy) channel", color="#1a73e8")
    ax.scatter(Xte[eve, 1], Xte[eve, 2], s=6, alpha=.4, label="Eavesdropper present", color="#e91e63")
    ax.set_xlabel("QBER in Z basis"); ax.set_ylabel("QBER in X basis"); ax.legend(markerscale=2); ax.set_title("Honest noise vs eavesdropper")
    fig.tight_layout(); fig.savefig("results/basis_scatter.png", dpi=160)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
