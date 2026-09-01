"""
Sends a small, realistic stream of events into live_server.py (must
already be running: python3 live_server.py in another terminal).

This is deliberately the thing to screen-record for the video: run
live_server.py in one terminal, this in another, and watch each decision
print in real time in the server's terminal, then check GET /queue.
"""
import time
import uuid
import requests
from datetime import datetime, timezone

BASE = "http://127.0.0.1:5050"


def rid(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def now_iso(offset_hours=0):
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).isoformat()


def post(path, payload):
    r = requests.post(f"{BASE}{path}", json=payload, timeout=5)
    print(f"  -> {r.status_code} {json_summary(r.json())}")
    return r.json()


def json_summary(d):
    keep = {k: d[k] for k in ("action", "risk_score", "reason_codes") if k in d}
    return keep


print("Checking server is up...")
try:
    h = requests.get(f"{BASE}/health", timeout=3).json()
    print(f"  OK -- {h}\n")
except Exception as e:
    print(f"Server not reachable at {BASE} -- start it first with: python3 live_server.py")
    raise SystemExit(1)

# ---- Event 1: an ordinary, legitimate provisioning ----
print("[1] Legit customer provisions a card on their own phone, at home")
legit_token = rid("tok")
post("/events/provisioning", {
    "token_id": legit_token, "cardholder_id": "card_27cf82cca2", "device_id": rid("dev_personal"),
    "provisioning_lat": 19.0667, "provisioning_lon": 72.7512,  # this cardholder's actual home, Mumbai
    "auth_factor": "biometric", "provisioning_ts": now_iso(),
})
time.sleep(1)

print("[2] ...and makes a normal purchase shortly after")
post("/events/transaction", {
    "token_id": legit_token, "ts": now_iso(0.5), "amount": 1200,
    "merchant_category": "grocery", "txn_lat": 19.08, "txn_lon": 72.88,
})
time.sleep(1)

# ---- Event 3: a device already flagged in seed state provisions ANOTHER card ----
print("\n[3] A device that's already provisioned tokens for several other cardholders "
      "now provisions one more, in a different city, via OTP only")
mule_device = rid("dev_shared")
victim_token = rid("tok")
# provision it 6 times rapid-fire under different cardholders to build up degree live,
# then the 7th is the one we actually watch escalate
for i in range(6):
    post("/events/provisioning", {
        "token_id": rid("tok"), "cardholder_id": rid("card_victim"), "device_id": mule_device,
        "provisioning_lat": 28.704, "provisioning_lon": 77.102,  # Delhi
        "auth_factor": "otp", "provisioning_ts": now_iso(),
    })
print("  ...device now has provisioning history with 6 different cardholders")
time.sleep(0.5)

print("[4] Now the SAME device provisions a 7th card -- this cardholder's actual home is "
      "Delhi, but the provisioning happens in Bangalore, via OTP only -- watch the score jump")
post("/events/provisioning", {
    "token_id": victim_token, "cardholder_id": "card_bbfcfefa90", "device_id": mule_device,
    "provisioning_lat": 12.972, "provisioning_lon": 77.594,  # Bangalore -- ~1700km from actual home
    "auth_factor": "otp", "provisioning_ts": now_iso(),
})
time.sleep(1)

print("[5] ...and immediately, a large transaction follows -- watch it escalate to HOLD")
post("/events/transaction", {
    "token_id": victim_token, "ts": now_iso(0.05), "amount": 42000,
    "merchant_category": "electronics", "txn_lat": 12.97, "txn_lon": 77.60,
})

print("\nDone. Check the full queue: curl http://127.0.0.1:5050/queue")
