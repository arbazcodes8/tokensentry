"""
Model P: scored at the PROVISIONING MOMENT ONLY, before any transaction has
happened. This is the honest question a real-time system actually has to
answer first: "should we act on this token right now, at the moment it was
created, using only device/geo/auth signal?" -- not "how would this token
look once we already know its transaction history," which is what Model B
answers and which literally cannot be known yet at provisioning time.

Completes the three-stage story:
  Model P -- provisioning-only   (available in ~0ms, before any money moves)
  Model A -- transaction-only    (needs transaction history to exist)
  Model B -- provisioning + txn  (best possible, once a first txn exists)

Persists the trained model + a fitted SHAP explainer so live_server.py
doesn't retrain on every request.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from features import compute_features, MODEL_A_FEATS, MODEL_B_FEATS, MODEL_P_FEATS
from paths import DATA_DIR as DATA
from thresholds import pick_thresholds


def main():
    cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
    tokens = pd.read_csv(f"{DATA}/tokens.csv")
    transactions = pd.read_csv(f"{DATA}/transactions.csv")
    df = compute_features(cardholders, tokens, transactions)

    train_all, test = df[df.split == "train"], df[df.split == "test"]

    # carve validation from train only, same discipline as agent.py
    train_clusters = train_all["cluster_id"].unique().tolist()
    rng = np.random.default_rng(13)
    rng.shuffle(train_clusters)
    n_val = int(len(train_clusters) * 0.2)
    val_clusters = set(train_clusters[:n_val])
    fit = train_all[~train_all["cluster_id"].isin(val_clusters)]
    val = train_all[train_all["cluster_id"].isin(val_clusters)]

    clf_fit = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf_fit.fit(fit[MODEL_P_FEATS], fit["is_fraudulent_provisioning"])
    proba_val = clf_fit.predict_proba(val[MODEL_P_FEATS])[:, 1]
    hold_t, stepup_t = pick_thresholds(val["is_fraudulent_provisioning"].values, proba_val)
    print(f"Model P thresholds chosen on validation (n={len(val)}): "
          f"immediate STEP_UP>={hold_t}  WATCH>={stepup_t}")

    clf_p = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    clf_p.fit(train_all[MODEL_P_FEATS], train_all["is_fraudulent_provisioning"])
    proba_p = clf_p.predict_proba(test[MODEL_P_FEATS])[:, 1]
    pred_p = (proba_p >= 0.5).astype(int)

    y = test["is_fraudulent_provisioning"].values
    p = precision_score(y, pred_p, zero_division=0)
    r = recall_score(y, pred_p, zero_division=0)
    f1 = f1_score(y, pred_p, zero_division=0)

    print("=" * 64)
    print("MODEL P -- provisioning-only, available before any transaction")
    print("=" * 64)
    print(f"Features ({len(MODEL_P_FEATS)}): {MODEL_P_FEATS}")
    print(f"Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}")
    print("\nThis is what the agent can act on the instant a token is born --")
    print("before the more accurate Model B is even computable, because Model B")
    print("needs a transaction to exist first.")

    joblib.dump({"model": clf_p, "features": MODEL_P_FEATS,
                 "stepup_threshold": hold_t, "watch_threshold": stepup_t}, f"{DATA}/model_p.joblib")
    print(f"\nSaved data/model_p.joblib")

    return clf_p


if __name__ == "__main__":
    main()
