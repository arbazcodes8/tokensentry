"""
Optional continuous variant of simulate_live_stream.py -- runs indefinitely,
mixing realistic legit activity with a slowly escalating mule device, so
you can leave it running during a live panel demo instead of a fixed
15-second script. Stop anytime with Ctrl+C.

Same server, same endpoints -- this is just a different event generator.
"""
import time
import uuid
import random
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

BASE = "http://127.0.0.1:5050"
CITIES = [(19.076, 72.877, "Mumbai"), (28.704, 77.102, "Delhi"), (12.972, 77.594, "Bangalore"),
          (13.083, 80.270, "Chennai"), (17.385, 78.487, "Hyderabad")]
CATS = ["ecommerce", "food_delivery", "travel", "electronics", "grocery", "subscription"]


def rid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso(offset_hours=0):
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).isoformat()


def post(path, payload, quiet=False):
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=5)
        d = r.json()
        if not quiet:
            tag = f"{d.get('action','?'):16s}"
            print(f"  -> {tag} score={d.get('risk_score','?')}")
        return d
    except Exception as e:
        print(f"  -> FAILED: {e}")
        return {}


print("Loading real cardholders for realistic home-distance signal...")
cardholders = pd.read_csv("../data/cardholders.csv").sample(200, random_state=None)
print(f"Loaded {len(cardholders)} cardholders. Starting continuous stream -- Ctrl+C to stop.\n")

# a small pool of "recurring shady devices" that persist across the whole run,
# so their degree genuinely climbs the longer this stays running
shady_devices = [rid("dev_shady") for _ in range(3)]
shady_device_active_tokens = []

try:
    tick = 0
    while True:
        tick += 1
        is_suspicious = random.random() < 0.15  # ~1 in 7 events is suspicious

        if is_suspicious:
            device = random.choice(shady_devices)
            row = cardholders.sample(1).iloc[0]
            lat, lon, city = random.choice(CITIES)
            token_id = rid("tok")
            print(f"[{tick}] SUSPICIOUS provisioning on a recurring device, in {city}")
            post("/events/provisioning", {
                "token_id": token_id, "cardholder_id": row["cardholder_id"], "device_id": device,
                "provisioning_lat": lat, "provisioning_lon": lon,
                "auth_factor": "otp", "provisioning_ts": now_iso(),
            })
            if random.random() < 0.6:
                time.sleep(0.8)
                print(f"      ...followed by a transaction")
                post("/events/transaction", {
                    "token_id": token_id, "ts": now_iso(0.02),
                    "amount": random.choice([15000, 28000, 45000]),
                    "merchant_category": random.choice(CATS),
                    "txn_lat": lat, "txn_lon": lon,
                })
        else:
            row = cardholders.sample(1).iloc[0]
            device = rid("dev_personal")
            token_id = rid("tok")
            print(f"[{tick}] Ordinary customer provisions at home")
            post("/events/provisioning", {
                "token_id": token_id, "cardholder_id": row["cardholder_id"], "device_id": device,
                "provisioning_lat": row["home_lat"], "provisioning_lon": row["home_lon"],
                "auth_factor": "biometric", "provisioning_ts": now_iso(),
            })
            if random.random() < 0.7:
                time.sleep(0.5)
                post("/events/transaction", {
                    "token_id": token_id, "ts": now_iso(1),
                    "amount": random.randint(200, 3000),
                    "merchant_category": random.choice(CATS),
                    "txn_lat": row["home_lat"], "txn_lon": row["home_lon"],
                })

        time.sleep(random.uniform(1.5, 3.0))

except KeyboardInterrupt:
    print("\nStopped.")
