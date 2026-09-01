"""
Synthetic data generator: Card-on-File Token Provisioning + Transactions.

Models Razorpay's real flow (per RBI CoFT rules): a token is unique to a
(card, token requestor/merchant, device) combination, created at a
PROVISIONING event, then used for TRANSACTIONS afterwards.

Design goals (why this isn't a toy dataset -- see fraud-ring-sentinel's
earlier mistake for what NOT to do):
1. Fraud comes in TWO forms: (a) mule devices that provision tokens for
   several different stolen identities [a graph/device-reuse signal], and
   (b) one-off fraud devices used exactly once [NOT visible to any
   device-reuse feature at all -- must be caught by geo-velocity, latency,
   or amount instead]. If provisioning intelligence only caught (a), the
   ablation would be trivial. Splitting fraud across both forces genuine
   reliance on multiple signal types.
2. Benign "family devices" shared by 2-4 real cardholders exist specifically
   to create false-positive risk for the device-reuse signal -- a device
   provisioning several DIFFERENT cards is not automatically fraud.
3. Some fraud is same-city (OTP phishing/social engineering -- device is
   unfamiliar but location matches) so geo-velocity alone cannot separate
   all fraud either.
4. Legitimate customers travel too (10% of legit provisioning happens away
   from home city), creating false-positive risk for geo-velocity alone.
5. Train/test split happens at the CLUSTER level (cardholders linked via a
   shared family/mule device end up entirely in train or entirely in test)
   so the model can never memorise a specific device_id -- it has to learn
   the underlying pattern.
"""
import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
from paths import DATA_DIR
import os

RNG = np.random.default_rng(7)

CITIES = {
    "Mumbai": (19.076, 72.877), "Delhi": (28.704, 77.102), "Bangalore": (12.972, 77.594),
    "Chennai": (13.083, 80.270), "Kolkata": (22.573, 88.364), "Hyderabad": (17.385, 78.487),
    "Pune": (18.520, 73.856), "Ahmedabad": (23.023, 72.571), "Jaipur": (26.912, 75.787),
    "Lucknow": (26.847, 80.947),
}
ABROAD = {"Dubai": (25.204, 55.271), "Singapore": (1.352, 103.820), "London": (51.507, -0.128)}
MERCHANT_CATS = ["ecommerce", "food_delivery", "travel", "electronics", "fuel",
                  "grocery", "entertainment", "jewellery", "subscription"]
ISSUERS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "IDFC"]
BASE_DATE = datetime(2026, 1, 1)
WINDOW_DAYS = 200

N_CARDHOLDERS = 4000
N_FAMILY_DEVICES = 45
N_MULE_DEVICES = 20
N_ONEOFF_FRAUD = 85


def rid(prefix, n=10):
    return f"{prefix}_{uuid.uuid4().hex[:n]}"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def rand_city():
    name = RNG.choice(list(CITIES.keys()))
    lat, lon = CITIES[name]
    return name, lat + RNG.normal(0, 0.05), lon + RNG.normal(0, 0.05)


def rand_ts(start, span_hours):
    return start + timedelta(hours=float(RNG.uniform(0, max(span_hours, 0.1))))


# ---------- 1. Cardholders ----------
cardholders = []
for i in range(N_CARDHOLDERS):
    cid = rid("card")
    city, lat, lon = rand_city()
    cardholders.append({
        "cardholder_id": cid, "home_city": city, "home_lat": lat, "home_lon": lon,
        "card_bin": f"{RNG.integers(400000,499999)}", "card_issuer": RNG.choice(ISSUERS),
        "personal_device": rid("dev"),
    })
cardholders_df = pd.DataFrame(cardholders)

# ---------- 2. Cluster devices (family = benign sharing, mule = fraud reuse) ----------
family_clusters = []  # list of cardholder_id lists sharing one device
pool = cardholders_df["cardholder_id"].tolist()
RNG.shuffle(pool)
idx = 0
for _ in range(N_FAMILY_DEVICES):
    size = int(RNG.integers(2, 5))
    if idx + size > len(pool):
        break
    family_clusters.append((rid("dev"), pool[idx:idx + size]))
    idx += size

mule_clusters = []
for _ in range(N_MULE_DEVICES):
    size = int(RNG.integers(3, 9))
    if idx + size > len(pool):
        break
    mule_clusters.append((rid("dev"), pool[idx:idx + size]))
    idx += size

oneoff_fraud_cardholders = pool[idx:idx + N_ONEOFF_FRAUD]
idx += N_ONEOFF_FRAUD

# ---------- 3. cluster_id per cardholder (for train/test split) ----------
cluster_of = {c: f"solo_{c}" for c in cardholders_df["cardholder_id"]}
for dev, members in family_clusters:
    cl = f"family_{dev}"
    for m in members:
        cluster_of[m] = cl
for dev, members in mule_clusters:
    cl = f"mule_{dev}"
    for m in members:
        cluster_of[m] = cl
for c in oneoff_fraud_cardholders:
    cluster_of[c] = f"oneoff_{c}"

ch_lookup = cardholders_df.set_index("cardholder_id").to_dict("index")

tokens, transactions = [], []


def add_transactions(token_id, prov_ts, is_fraud_token, base_lat, base_lon, aggressive):
    n_txn = int(RNG.integers(1, 4)) if is_fraud_token else int(RNG.integers(1, 12))
    for k in range(n_txn):
        latency_hours = float(RNG.exponential(1.5)) if (is_fraud_token and aggressive and k == 0) \
            else float(RNG.exponential(48))
        ts = prov_ts + timedelta(hours=latency_hours)
        amt_base = 8000 if (is_fraud_token and aggressive) else 1800
        amount = float(np.clip(RNG.lognormal(np.log(amt_base), 0.8), 100, 150000))
        jitter_lat, jitter_lon = RNG.normal(0, 0.3), RNG.normal(0, 0.3)
        transactions.append({
            "txn_id": rid("txn", 8), "token_id": token_id, "ts": ts, "amount": amount,
            "merchant_category": RNG.choice(MERCHANT_CATS),
            "txn_lat": base_lat + jitter_lat, "txn_lon": base_lon + jitter_lon,
        })


# ---------- 4. Legit tokens for every cardholder ----------
for cid, info in ch_lookup.items():
    n_tokens = int(RNG.integers(1, 5))
    device = info["personal_device"]
    for _ in range(n_tokens):
        traveling = RNG.random() < 0.10
        if traveling:
            _, plat, plon = rand_city() if RNG.random() < 0.8 else (None, *list(ABROAD.values())[int(RNG.integers(0,3))])
        else:
            plat, plon = info["home_lat"] + RNG.normal(0, 0.05), info["home_lon"] + RNG.normal(0, 0.05)
        prov_ts = BASE_DATE + timedelta(days=float(RNG.uniform(0, WINDOW_DAYS)))
        token_id = rid("tok")
        tokens.append({
            "token_id": token_id, "cardholder_id": cid, "device_id": device,
            "token_requestor": RNG.choice(MERCHANT_CATS), "provisioning_ts": prov_ts,
            "provisioning_lat": plat, "provisioning_lon": plon,
            "auth_factor": RNG.choice(["otp", "biometric"], p=[0.6, 0.4]),
            "is_fraudulent_provisioning": 0, "cluster_id": cluster_of[cid],
        })
        add_transactions(token_id, prov_ts, is_fraud_token=False,
                          base_lat=plat, base_lon=plon, aggressive=False)

# ---------- 5. Family devices ALSO provision extra legit tokens jointly (benign multi-card device) ----------
for dev, members in family_clusters:
    for m in members:
        info = ch_lookup[m]
        prov_ts = BASE_DATE + timedelta(days=float(RNG.uniform(0, WINDOW_DAYS)))
        token_id = rid("tok")
        tokens.append({
            "token_id": token_id, "cardholder_id": m, "device_id": dev,
            "token_requestor": RNG.choice(MERCHANT_CATS), "provisioning_ts": prov_ts,
            "provisioning_lat": info["home_lat"] + RNG.normal(0, 0.05),
            "provisioning_lon": info["home_lon"] + RNG.normal(0, 0.05),
            "auth_factor": RNG.choice(["otp", "biometric"], p=[0.6, 0.4]),
            "is_fraudulent_provisioning": 0, "cluster_id": cluster_of[m],
        })
        add_transactions(token_id, prov_ts, is_fraud_token=False,
                          base_lat=info["home_lat"], base_lon=info["home_lon"], aggressive=False)

# ---------- 6. Mule-device fraud (device reused across several stolen identities) ----------
for dev, victims in mule_clusters:
    burst_start = BASE_DATE + timedelta(days=float(RNG.uniform(0, WINDOW_DAYS)))
    same_city_fraud = RNG.random() < 0.4  # some mule fraud is local (OTP phishing), not cross-city
    for v in victims:
        info = ch_lookup[v]
        prov_ts = rand_ts(burst_start, 72)
        if same_city_fraud:
            plat, plon = info["home_lat"] + RNG.normal(0, 0.1), info["home_lon"] + RNG.normal(0, 0.1)
        else:
            _, plat, plon = rand_city()
        token_id = rid("tok")
        tokens.append({
            "token_id": token_id, "cardholder_id": v, "device_id": dev,
            "token_requestor": RNG.choice(MERCHANT_CATS), "provisioning_ts": prov_ts,
            "provisioning_lat": plat, "provisioning_lon": plon,
            "auth_factor": RNG.choice(["otp", "biometric"], p=[0.85, 0.15]),
            "is_fraudulent_provisioning": 1, "cluster_id": cluster_of[v],
        })
        add_transactions(token_id, prov_ts, is_fraud_token=True,
                          base_lat=plat, base_lon=plon, aggressive=RNG.random() < 0.75)

# ---------- 7. One-off fraud (unique device, never reused -- no device-graph signal at all) ----------
for v in oneoff_fraud_cardholders:
    info = ch_lookup[v]
    dev = rid("dev")
    same_city_fraud = RNG.random() < 0.35
    prov_ts = BASE_DATE + timedelta(days=float(RNG.uniform(0, WINDOW_DAYS)))
    if same_city_fraud:
        plat, plon = info["home_lat"] + RNG.normal(0, 0.1), info["home_lon"] + RNG.normal(0, 0.1)
    else:
        _, plat, plon = rand_city()
    token_id = rid("tok")
    tokens.append({
        "token_id": token_id, "cardholder_id": v, "device_id": dev,
        "token_requestor": RNG.choice(MERCHANT_CATS), "provisioning_ts": prov_ts,
        "provisioning_lat": plat, "provisioning_lon": plon,
        "auth_factor": RNG.choice(["otp", "biometric"], p=[0.85, 0.15]),
        "is_fraudulent_provisioning": 1, "cluster_id": cluster_of[v],
    })
    add_transactions(token_id, prov_ts, is_fraud_token=True,
                      base_lat=plat, base_lon=plon, aggressive=RNG.random() < 0.75)

tokens_df = pd.DataFrame(tokens)
transactions_df = pd.DataFrame(transactions)

# ---------- 8. Train/test split at CLUSTER level ----------
clusters = tokens_df["cluster_id"].unique().tolist()
RNG.shuffle(clusters)
n_test = int(len(clusters) * 0.3)
test_clusters = set(clusters[:n_test])
tokens_df["split"] = tokens_df["cluster_id"].apply(lambda c: "test" if c in test_clusters else "train")

for split in ["train", "test"]:
    sub = tokens_df[tokens_df.split == split]
    print(f"{split}: {len(sub)} tokens | fraud: {sub.is_fraudulent_provisioning.sum()} "
          f"({sub.is_fraudulent_provisioning.mean()*100:.1f}%)")

cardholders_df.to_csv(os.path.join(DATA_DIR, "cardholders.csv"), index=False)
tokens_df.to_csv(os.path.join(DATA_DIR, "tokens.csv"), index=False)
transactions_df.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
print(f"\nTotal tokens: {len(tokens_df)} | fraudulent: {tokens_df.is_fraudulent_provisioning.sum()} "
      f"({tokens_df.is_fraudulent_provisioning.mean()*100:.2f}%)")
print(f"Total transactions: {len(transactions_df)}")
print(f"Mule-device fraud clusters: {len(mule_clusters)} | one-off fraud: {len(oneoff_fraud_cardholders)} "
      f"| family (benign) clusters: {len(family_clusters)}")
