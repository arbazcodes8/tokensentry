"""
Builds two feature sets from the same underlying data:

MODEL A (transaction-only baseline) -- what most fraud systems already do:
  behaviour of the transactions themselves (amount, velocity, how far the
  transaction is from the cardholder's home).

MODEL B (provisioning-enhanced) -- adds signal that only exists at the
  moment the TOKEN was created: how far provisioning was from home, how
  fast the first transaction followed provisioning, how many different
  cardholders this exact device has provisioned tokens for (mule-device
  signal), and which authentication factor was used.

This split lets train_eval.py honestly test the abstract's hypothesis:
does provisioning intelligence add real, measurable lift over transaction
data alone -- and if so, on which kind of fraud specifically.
"""
import pandas as pd
import numpy as np
import os
from paths import DATA_DIR

EARTH_R = 6371


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


MODEL_A_FEATS = ["num_txns", "avg_amount", "max_amount", "txn_velocity",
                  "avg_txn_home_dist_km", "max_txn_home_dist_km", "category_diversity"]
MODEL_B_FEATS = ["device_card_degree", "device_is_shared", "provisioning_home_dist_km",
                  "provisioning_to_first_txn_hours", "auth_is_otp"]
# Available at the PROVISIONING MOMENT ONLY -- before any transaction exists.
# provisioning_to_first_txn_hours is deliberately excluded: it cannot be known
# until a first transaction actually happens.
MODEL_P_FEATS = ["device_card_degree", "device_is_shared", "provisioning_home_dist_km", "auth_is_otp"]


def compute_features(cardholders_df, tokens_df, transactions_df):
    ch = cardholders_df.set_index("cardholder_id")
    tok = tokens_df.copy()
    tok["provisioning_ts"] = pd.to_datetime(tok["provisioning_ts"])
    tx = transactions_df.copy()
    tx["ts"] = pd.to_datetime(tx["ts"])

    tok = tok.join(ch[["home_lat", "home_lon"]], on="cardholder_id")

    # --- transaction-level aggregates per token (Model A) ---
    tx = tx.merge(tok[["token_id", "home_lat", "home_lon"]], on="token_id", how="left")
    tx["txn_home_dist_km"] = haversine_km(tx["txn_lat"], tx["txn_lon"], tx["home_lat"], tx["home_lon"])

    tx_agg = tx.groupby("token_id").agg(
        num_txns=("txn_id", "count"),
        avg_amount=("amount", "mean"),
        max_amount=("amount", "max"),
        first_txn_ts=("ts", "min"),
        last_txn_ts=("ts", "max"),
        avg_txn_home_dist_km=("txn_home_dist_km", "mean"),
        max_txn_home_dist_km=("txn_home_dist_km", "max"),
        category_diversity=("merchant_category", "nunique"),
    ).reset_index()

    df = tok.merge(tx_agg, on="token_id", how="left")
    fill_cols = ["num_txns", "avg_amount", "max_amount", "avg_txn_home_dist_km",
                 "max_txn_home_dist_km", "category_diversity"]
    df[fill_cols] = df[fill_cols].fillna(0)
    span_h = (df["last_txn_ts"] - df["first_txn_ts"]).dt.total_seconds() / 3600
    df["txn_velocity"] = (df["num_txns"] / span_h.replace(0, np.nan)).fillna(df["num_txns"])

    # --- provisioning-level features (Model B) ---
    df["provisioning_home_dist_km"] = haversine_km(
        df["provisioning_lat"], df["provisioning_lon"], df["home_lat"], df["home_lon"])
    df["provisioning_to_first_txn_hours"] = (
        (df["first_txn_ts"] - df["provisioning_ts"]).dt.total_seconds() / 3600
    ).fillna(999)
    df["auth_is_otp"] = (df["auth_factor"] == "otp").astype(int)

    device_degree = tok.groupby("device_id")["cardholder_id"].nunique() - 1
    df["device_card_degree"] = df["device_id"].map(device_degree).fillna(0)
    df["device_is_shared"] = (df["device_card_degree"] > 0).astype(int)

    return df


if __name__ == "__main__":
    cardholders = pd.read_csv(os.path.join(DATA_DIR, "cardholders.csv"))
    tokens = pd.read_csv(os.path.join(DATA_DIR, "tokens.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    df = compute_features(cardholders, tokens, transactions)
    cols = MODEL_A_FEATS + MODEL_B_FEATS
    print(df[cols + ["is_fraudulent_provisioning"]].groupby("is_fraudulent_provisioning").mean(numeric_only=True).T)
    df.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)
    print(f"\nSaved data/features.csv | Model A feats: {len(MODEL_A_FEATS)} | Model B adds: {len(MODEL_B_FEATS)}")
