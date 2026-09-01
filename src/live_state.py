"""
The real-time feature store. This is the piece that makes real-time
actually honest rather than theater: instead of a groupby over the whole
dataset (which needs everything to already exist), it keeps small running
state and updates it as each event arrives -- the same shape a production
system would need (a feature store / cache), just in-memory for a demo.

Seeded from TRAIN data only at startup (representing "state as of go-live")
-- never seeded from the held-out test set, so a live demo never
accidentally leaks the evaluation data it's supposed to be independent of.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from features import haversine_km
from paths import DATA_DIR as DATA


class FeatureStore:
    def __init__(self):
        self.device_to_cardholders = {}   # device_id -> set(cardholder_id)
        self.cardholder_home = {}         # cardholder_id -> (lat, lon)
        self.tokens = {}                  # token_id -> dict(state)
        self._seed_from_train()

    def _seed_from_train(self):
        cardholders = pd.read_csv(f"{DATA}/cardholders.csv")
        tokens = pd.read_csv(f"{DATA}/tokens.csv")
        transactions = pd.read_csv(f"{DATA}/transactions.csv")

        for _, row in cardholders.iterrows():
            self.cardholder_home[row["cardholder_id"]] = (row["home_lat"], row["home_lon"])

        train_tokens = tokens[tokens.split == "train"]
        for _, row in train_tokens.iterrows():
            self.device_to_cardholders.setdefault(row["device_id"], set()).add(row["cardholder_id"])
            self.tokens[row["token_id"]] = {
                "cardholder_id": row["cardholder_id"], "device_id": row["device_id"],
                "provisioning_ts": pd.to_datetime(row["provisioning_ts"]),
                "provisioning_lat": row["provisioning_lat"], "provisioning_lon": row["provisioning_lon"],
                "auth_factor": row["auth_factor"], "transactions": [],
            }
        tx_by_token = transactions.groupby("token_id")
        for token_id, group in tx_by_token:
            if token_id in self.tokens:
                for _, t in group.iterrows():
                    self.tokens[token_id]["transactions"].append({
                        "ts": pd.to_datetime(t["ts"]), "amount": t["amount"],
                        "merchant_category": t["merchant_category"],
                        "txn_lat": t["txn_lat"], "txn_lon": t["txn_lon"],
                    })
        print(f"[FeatureStore] seeded from {len(train_tokens)} train tokens, "
              f"{len(self.device_to_cardholders)} distinct devices")

    # ---------- STAGE 1: provisioning-time features ----------
    def score_provisioning(self, event):
        """event: token_id, cardholder_id, device_id, provisioning_lat,
        provisioning_lon, auth_factor, provisioning_ts (ISO string)."""
        device_id = event["device_id"]
        cardholder_id = event["cardholder_id"]

        # degree computed from state BEFORE this event is added -- "how many
        # OTHER cardholders has this device provisioned for, prior to now"
        prior_holders = self.device_to_cardholders.get(device_id, set()) - {cardholder_id}
        device_card_degree = len(prior_holders)
        device_is_shared = int(device_card_degree > 0)

        home = self.cardholder_home.get(cardholder_id, (event["provisioning_lat"], event["provisioning_lon"]))
        provisioning_home_dist_km = float(haversine_km(
            event["provisioning_lat"], event["provisioning_lon"], home[0], home[1]))
        auth_is_otp = int(event["auth_factor"] == "otp")

        features = {
            "device_card_degree": device_card_degree,
            "device_is_shared": device_is_shared,
            "provisioning_home_dist_km": provisioning_home_dist_km,
            "auth_is_otp": auth_is_otp,
        }

        # NOW commit the event to state
        self.device_to_cardholders.setdefault(device_id, set()).add(cardholder_id)
        self.tokens[event["token_id"]] = {
            "cardholder_id": cardholder_id, "device_id": device_id,
            "provisioning_ts": pd.to_datetime(event["provisioning_ts"]),
            "provisioning_lat": event["provisioning_lat"], "provisioning_lon": event["provisioning_lon"],
            "auth_factor": event["auth_factor"], "transactions": [],
        }
        return features

    # ---------- STAGE 2: provisioning + transaction features ----------
    def score_transaction(self, token_id, txn):
        """txn: ts (ISO string), amount, merchant_category, txn_lat, txn_lon."""
        if token_id not in self.tokens:
            raise KeyError(f"Unknown token_id {token_id} -- was it provisioned first?")
        state = self.tokens[token_id]
        state["transactions"].append({
            "ts": pd.to_datetime(txn["ts"]), "amount": txn["amount"],
            "merchant_category": txn["merchant_category"],
            "txn_lat": txn["txn_lat"], "txn_lon": txn["txn_lon"],
        })
        txns = state["transactions"]
        home = self.cardholder_home.get(state["cardholder_id"],
                                          (state["provisioning_lat"], state["provisioning_lon"]))
        dists = [haversine_km(t["txn_lat"], t["txn_lon"], home[0], home[1]) for t in txns]
        amounts = [t["amount"] for t in txns]
        first_ts = min(t["ts"] for t in txns)
        last_ts = max(t["ts"] for t in txns)
        span_h = max((last_ts - first_ts).total_seconds() / 3600, 1e-6)
        velocity = len(txns) / span_h if span_h > 0.01 else float(len(txns))

        device_id = state["device_id"]
        cardholder_id = state["cardholder_id"]
        prior_holders = self.device_to_cardholders.get(device_id, set()) - {cardholder_id}
        device_card_degree = len(prior_holders)

        features = {
            "num_txns": len(txns),
            "avg_amount": float(np.mean(amounts)),
            "max_amount": float(np.max(amounts)),
            "txn_velocity": float(velocity),
            "avg_txn_home_dist_km": float(np.mean(dists)),
            "max_txn_home_dist_km": float(np.max(dists)),
            "category_diversity": len(set(t["merchant_category"] for t in txns)),
            "device_card_degree": device_card_degree,
            "device_is_shared": int(device_card_degree > 0),
            "provisioning_home_dist_km": float(haversine_km(
                state["provisioning_lat"], state["provisioning_lon"], home[0], home[1])),
            "provisioning_to_first_txn_hours": float(
                (first_ts - state["provisioning_ts"]).total_seconds() / 3600),
            "auth_is_otp": int(state["auth_factor"] == "otp"),
        }
        return features
