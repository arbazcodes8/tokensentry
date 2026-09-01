"""
The actual AGENT: takes Model B's risk score for a token and turns it into
a bounded, explainable, auditable decision.

Three actions only -- all defensive, all reversible by a human, never an
autonomous ban:
  APPROVE            -- let it through
  STEP_UP_AUTH       -- ask for an extra verification factor before allowing
  HOLD_FOR_REVIEW     -- pause and route to a human fraud analyst

Thresholds for these tiers are chosen on a VALIDATION split carved out of
TRAIN only -- never on the test set -- then applied once, frozen, to test.
Picking thresholds on the test set would silently leak information and
inflate the reported numbers; this is a common honest-metrics mistake and
deliberately avoided here.

Every decision is explained with SHAP-based reason codes (which features
actually pushed this specific token's score up) and written to an
append-only audit log.
"""
import json
import joblib
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

from features import compute_features, MODEL_A_FEATS, MODEL_B_FEATS
from paths import DATA_DIR as DATA
from thresholds import pick_thresholds
FEATS = MODEL_A_FEATS + MODEL_B_FEATS

REASON_PHRASES = {
    "device_card_degree": "this device has provisioned tokens for {v:.0f} other cardholder(s)",
    "device_is_shared": "this device is shared with other cardholders",
    "provisioning_home_dist_km": "provisioned {v:.0f} km from the cardholder's home",
    "provisioning_to_first_txn_hours": "first transaction followed provisioning within {v:.1f} hours",
    "auth_is_otp": "provisioned using OTP only (no biometric factor)",
    "avg_amount": "average transaction amount is Rs{v:,.0f}",
    "max_amount": "largest transaction is Rs{v:,.0f}",
    "avg_txn_home_dist_km": "transactions average {v:.0f} km from home",
    "max_txn_home_dist_km": "a transaction occurred {v:.0f} km from home",
    "num_txns": "{v:.0f} transactions on this token so far",
    "txn_velocity": "{v:.2f} transactions per hour",
    "category_diversity": "spread across {v:.0f} merchant categories",
}


def reason_code(feat, value):
    template = REASON_PHRASES.get(feat, f"{feat}={value}")
    try:
        return template.format(v=value)
    except Exception:
        return f"{feat}={value}"


def pick_thresholds_local(y_val, proba_val):
    """Deprecated -- use thresholds.pick_thresholds (kept here as a no-op
    alias only in case something still imports this name)."""
    return pick_thresholds(y_val, proba_val)


def main():
    cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    transactions = pd.read_csv(f"{DATA}/transactions.csv")
    df = compute_features(cardholders, tokens, transactions)

    train_all = df[df.split == "train"]
    test = df[df.split == "test"]

    # carve validation out of TRAIN clusters only (never touches test)
    train_clusters = train_all["cluster_id"].unique().tolist()
    rng = np.random.default_rng(11)
    rng.shuffle(train_clusters)
    n_val = int(len(train_clusters) * 0.2)
    val_clusters = set(train_clusters[:n_val])
    fit = train_all[~train_all["cluster_id"].isin(val_clusters)]
    val = train_all[train_all["cluster_id"].isin(val_clusters)]

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf.fit(fit[FEATS], fit["is_fraudulent_provisioning"])

    proba_val = clf.predict_proba(val[FEATS])[:, 1]
    hold_t, stepup_t = pick_thresholds(val["is_fraudulent_provisioning"].values, proba_val)
    print(f"Thresholds chosen on validation (n={len(val)}, never touches test): "
          f"HOLD>={hold_t}  STEP_UP>={stepup_t}")

    # retrain on full train (fit+val) for the deployed model, apply frozen thresholds to test
    clf_final = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf_final.fit(train_all[FEATS], train_all["is_fraudulent_provisioning"])
    proba_test = clf_final.predict_proba(test[FEATS])[:, 1]

    def action_for(p):
        if p >= hold_t:
            return "HOLD_FOR_REVIEW"
        if p >= stepup_t:
            return "STEP_UP_AUTH"
        return "APPROVE"

    actions = np.array([action_for(p) for p in proba_test])
    print("\nAction distribution on held-out test set:")
    print(pd.Series(actions).value_counts().to_string())

    acted = actions != "APPROVE"
    print(f"\nPrecision at HOLD+STEP_UP tier: {precision_score(test['is_fraudulent_provisioning'], acted):.3f}")
    print(f"Recall at HOLD+STEP_UP tier:    {recall_score(test['is_fraudulent_provisioning'], acted):.3f}")

    explainer = shap.TreeExplainer(clf_final)
    shap_values = explainer.shap_values(test[FEATS])

    flagged_idx = np.where(actions != "APPROVE")[0]
    audit_entries = []
    for i in flagged_idx:
        row = test.iloc[i]
        sv = shap_values[i]
        top_idx = np.argsort(-np.abs(sv))[:3]
        reasons = [reason_code(FEATS[j], row[FEATS[j]]) for j in top_idx]
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_id": row["token_id"],
            "cardholder_id_masked": row["cardholder_id"][:8] + "...",
            "risk_score": round(float(proba_test[i]), 4),
            "action": actions[i],
            "amount_inr": round(float(row["avg_amount"]), 2),
            "reason_codes": reasons,
            "ground_truth_fraud": bool(row["is_fraudulent_provisioning"]),  # kept for eval only, not shown to analyst
            "model_version": "tokensentry-modelB-v1",
        }
        audit_entries.append(entry)

    with open(f"{DATA}/audit_log.jsonl", "w") as fh:
        for e in audit_entries:
            fh.write(json.dumps({k: v for k, v in e.items() if k != "ground_truth_fraud"}) + "\n")
    print(f"\nWrote {len(audit_entries)} audit log entries to data/audit_log.jsonl")

    print("\nExample decision (from the audit trail):")
    example = [e for e in audit_entries if e["ground_truth_fraud"]][0]
    print(json.dumps({k: v for k, v in example.items() if k != "ground_truth_fraud"}, indent=2))

    joblib.dump({"model": clf_final, "features": FEATS, "hold_threshold": hold_t,
                 "stepup_threshold": stepup_t}, f"{DATA}/model_b.joblib")
    print(f"\nSaved data/model_b.joblib (for live_server.py)")

    return clf_final, hold_t, stepup_t


if __name__ == "__main__":
    main()
