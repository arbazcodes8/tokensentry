"""
Trains Model A (transaction-only) and Model B (transaction + provisioning
intelligence) on the SAME train/test split (held-out clusters -- see
generate_data.py) and reports honest metrics, matching the capstone
abstract's stated experiment: does provisioning intelligence achieve a
statistically significant improvement over the transaction-only baseline?

Statistical test: McNemar's test on the paired predictions (same test
tokens, two different models) -- the correct test for "did adding a
feature set change what this specific model gets right/wrong", not a
generic accuracy comparison.

Also reports a breakdown by fraud subtype (mule-device vs one-off) to
show WHERE provisioning intelligence helps, not just a single headline
number -- one-off fraud has no device-graph signal at all, so if Model B
still beats Model A there, the lift is coming from geo-velocity/latency,
not just "catching repeat devices".
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

from features import compute_features, MODEL_A_FEATS, MODEL_B_FEATS
from paths import DATA_DIR as DATA
COST_PER_FALSE_POSITIVE_REVIEW = 40   # INR, analyst review of a held transaction
AVG_FRAUD_LOSS_IF_MISSED = 9500       # INR, avg loss of a missed fraudulent token


def fraud_subtype(cluster_id):
    if cluster_id.startswith("mule_"):
        return "mule_device"
    if cluster_id.startswith("oneoff_"):
        return "one_off"
    return "legit"


def evaluate(y_true, y_pred, label):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = fp * COST_PER_FALSE_POSITIVE_REVIEW + fn * AVG_FRAUD_LOSS_IF_MISSED
    print(f"  [{label}] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f} "
          f"TP={tp} FP={fp} FN={fn} TN={tn} est.cost=Rs{cost:,}")
    return dict(precision=p, recall=r, f1=f1, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), cost=int(cost))


def main():
    cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    transactions = pd.read_csv(f"{DATA}/transactions.csv")
    df = compute_features(cardholders, tokens, transactions)
    df["fraud_subtype"] = df["cluster_id"].apply(
        lambda c: fraud_subtype(c) if not c.startswith(("solo_", "family_")) else "legit")

    train, test = df[df.split == "train"], df[df.split == "test"]

    # ---- Model A: transaction-only ----
    clf_a = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf_a.fit(train[MODEL_A_FEATS], train["is_fraudulent_provisioning"])
    proba_a = clf_a.predict_proba(test[MODEL_A_FEATS])[:, 1]
    pred_a = (proba_a >= 0.5).astype(int)

    # ---- Model B: transaction + provisioning intelligence ----
    feats_b = MODEL_A_FEATS + MODEL_B_FEATS
    clf_b = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf_b.fit(train[feats_b], train["is_fraudulent_provisioning"])
    proba_b = clf_b.predict_proba(test[feats_b])[:, 1]
    pred_b = (proba_b >= 0.5).astype(int)

    y = test["is_fraudulent_provisioning"].values
    print("=" * 64)
    print("MODEL A vs MODEL B -- held-out test set (never-seen clusters)")
    print("=" * 64)
    res_a = evaluate(y, pred_a, "Model A: transaction-only")
    res_b = evaluate(y, pred_b, "Model B: + provisioning intelligence")

    # ---- McNemar's test: is the DIFFERENCE statistically significant? ----
    both_wrong_a_right = int(((pred_a == y) & (pred_b != y)).sum())
    both_wrong_b_right = int(((pred_a != y) & (pred_b == y)).sum())
    table = [[0, both_wrong_a_right], [both_wrong_b_right, 0]]
    result = mcnemar(table, exact=True)
    print(f"\n[McNemar's test] A-right/B-wrong: {both_wrong_a_right}  "
          f"B-right/A-wrong: {both_wrong_b_right}  p-value={result.pvalue:.5f}")
    sig = "YES -- statistically significant (p<0.05)" if result.pvalue < 0.05 else "NOT significant at p<0.05"
    print(f"  Is Model B's improvement significant? {sig}")

    # ---- Breakdown by fraud subtype ----
    print("\n[Breakdown by fraud subtype -- where does provisioning intelligence help?]")
    for subtype in ["mule_device", "one_off"]:
        mask = (test["fraud_subtype"].values == subtype)
        n = mask.sum()
        if n == 0:
            continue
        rec_a = recall_score(y[mask], pred_a[mask], zero_division=0)
        rec_b = recall_score(y[mask], pred_b[mask], zero_division=0)
        print(f"  {subtype:12s} (n={n:3d})  Model A recall={rec_a:.3f}  Model B recall={rec_b:.3f}  "
              f"lift={rec_b-rec_a:+.3f}")

    # ---- Feature importances for Model B (for the agent's reason codes later) ----
    print("\n[Model B feature importances]")
    for f, imp in sorted(zip(feats_b, clf_b.feature_importances_), key=lambda x: -x[1])[:8]:
        print(f"  {f:35s} {imp:.3f}")

    return clf_a, clf_b, df


if __name__ == "__main__":
    main()
