"""
Live enforcement layer: takes the agent's already-decided actions
(data/audit_log.jsonl) and enacts them against REAL Razorpay test-mode
infrastructure -- not a simulation of Razorpay's behaviour, the actual
sandbox.

Requires your own test-mode credentials (free, no KYC needed):
  1. https://dashboard.razorpay.com -> stay in Test Mode
  2. Account & Settings -> API Keys -> Generate Key
  3. export RAZORPAY_KEY_ID=rzp_test_xxxxx
     export RAZORPAY_KEY_SECRET=xxxxx

What this deliberately does NOT try to do: fully automate an end-to-end
checkout. Razorpay's Checkout requires a browser to actually authorize a
payment with a test card -- that's true for every merchant integration,
not a limitation of this project. What server-side code CAN honestly do,
and what this module does:

  - Create a real test-mode Order for each flagged/approved token, with
    the `payment_capture` setting driven directly by the agent's decision:
      APPROVE          -> payment_capture=1 (auto-capture if paid)
      STEP_UP_AUTH      -> payment_capture=0 (held, requires manual capture)
      HOLD_FOR_REVIEW   -> payment_capture=0 (held, requires manual capture)
  - This uses Razorpay's REAL authorize-then-capture mechanism: an order
    marked manual-capture, if paid, sits in `authorized` state and
    auto-refunds if nobody captures it within Razorpay's timeout window.
    That is a genuine, non-simulated way to prove the agent's "hold"
    decision actually stops money from moving automatically.

Every order created is logged to data/razorpay_orders_log.jsonl with the
real order_id Razorpay returns -- visible in your own Test Mode dashboard
under Transactions -> Orders, for anyone reviewing this to verify directly.
"""
import os
import json
import sys
from paths import DATA_DIR as DATA


def get_client():
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        print("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set.")
        print("Get free test-mode keys at https://dashboard.razorpay.com "
              "(Account & Settings -> API Keys), then:")
        print("  export RAZORPAY_KEY_ID=rzp_test_xxxxx")
        print("  export RAZORPAY_KEY_SECRET=xxxxx")
        return None
    try:
        import razorpay
    except ImportError as e:
        print(f"Run: pip install razorpay  (import error: {e})")
        return None
    client = razorpay.Client(auth=(key_id, key_secret))
    return client


def enforce(entry, client):
    """Create a real test-mode order reflecting this decision."""
    action = entry["action"]
    amount_paise = int(round(entry.get("amount_inr", 500) * 100))
    auto_capture = 1 if action == "APPROVE" else 0

    order_payload = {
        "amount": amount_paise,
        "currency": "INR",
        "payment_capture": auto_capture,
        "receipt": entry["token_id"][:40],
        "notes": {
            "tokensentry_action": action,
            "risk_score": str(entry["risk_score"]),
        },
    }
    try:
        order = client.order.create(data=order_payload)
        result = {
            "token_id": entry["token_id"],
            "action": action,
            "razorpay_order_id": order["id"],
            "razorpay_status": order["status"],
            "payment_capture_mode": "auto" if auto_capture else "manual",
            "amount_inr": entry.get("amount_inr"),
        }
        print(f"  [{action:16s}] order {order['id']}  status={order['status']}  "
              f"capture={'auto' if auto_capture else 'manual (held)'}")
        return result
    except Exception as e:
        print(f"  [{action:16s}] FAILED for {entry['token_id']}: {e}")
        return {"token_id": entry["token_id"], "action": action, "error": str(e)}


def main():
    client = get_client()
    if client is None:
        print("\nNo credentials -- skipping. This module is optional; the "
              "core pipeline does not depend on it.")
        return

    with open(f"{DATA}/audit_log.jsonl") as fh:
        entries = [json.loads(l) for l in fh]

    print(f"Enforcing {len(entries)} agent decisions against Razorpay TEST MODE...\n")
    results = [enforce(e, client) for e in entries]

    with open(f"{DATA}/razorpay_orders_log.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    ok = sum(1 for r in results if "razorpay_order_id" in r)
    print(f"\n{ok}/{len(results)} orders created successfully in Razorpay Test Mode.")
    print(f"Verify directly at: https://dashboard.razorpay.com -> Test Mode -> Transactions -> Orders")
    print(f"Log written to data/razorpay_orders_log.jsonl")


if __name__ == "__main__":
    main()
