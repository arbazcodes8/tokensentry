"""
The live agent. Run this, then hit it with events -- either curl, the
included simulate_live_stream.py, or Postman.

Two endpoints, matching the real timeline:

  POST /events/provisioning
    A token was just created. Score it with Model P (provisioning-only
    signal) and decide immediately: APPROVE, WATCH, or STEP_UP_AUTH.
    This is the score that can act before any money has moved.

  POST /events/transaction
    A transaction just happened against an already-provisioned token.
    Score it with Model B (full signal, now that behaviour exists) and
    decide: APPROVE, STEP_UP_AUTH, or HOLD_FOR_REVIEW. May escalate past
    whatever Stage 1 decided.

  GET /queue
    Every decision made since the server started, most recent first.

Start with: python3 live_server.py
Then, in another terminal: python3 simulate_live_stream.py
"""
import json
import joblib
import shap
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from flask import Flask, request, jsonify

from live_state import FeatureStore
from investigate import investigate
from paths import DATA_DIR as DATA

app = Flask(__name__)
store = FeatureStore()

model_p_bundle = joblib.load(f"{DATA}/model_p.joblib")
model_b_bundle = joblib.load(f"{DATA}/model_b.joblib")
clf_p, FEATS_P = model_p_bundle["model"], model_p_bundle["features"]
clf_b, FEATS_B = model_b_bundle["model"], model_b_bundle["features"]
explainer_p = shap.TreeExplainer(clf_p)
explainer_b = shap.TreeExplainer(clf_b)

STEPUP_T_P = model_p_bundle["stepup_threshold"]
WATCH_T_P = model_p_bundle["watch_threshold"]
HOLD_T_B = model_b_bundle["hold_threshold"]
STEPUP_T_B = model_b_bundle["stepup_threshold"]

DECISION_LOG = []  # in-memory, most-recent-first

LIVE_DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>TokenSentry — Live</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root { --ink:#0B1220; --panel:#121A2B; --hairline:#263047; --text:#E7ECF5; --muted:#8593AD;
          --hold:#E5484D; --stepup:#F0A93E; --watch:#5AA9E6; --approve:#3FB68B; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--ink); color:var(--text); font-family:'Inter',sans-serif; padding:40px 24px 80px; }
  .wrap { max-width:920px; margin:0 auto; }
  .eyebrow { font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.12em; color:var(--muted);
             text-transform:uppercase; margin-bottom:8px; }
  h1 { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:28px; margin:0 0 4px; display:flex; align-items:center; gap:10px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--approve); animation:pulse 1.6s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .sub { color:var(--muted); font-size:13px; margin-bottom:28px; }
  .row { display:flex; background:var(--panel); border-radius:8px; margin-bottom:8px; overflow:hidden; border:1px solid var(--hairline); animation:in .3s ease; }
  @keyframes in { from{opacity:0; transform:translateY(-6px)} to{opacity:1; transform:translateY(0)} }
  .row-bar { width:4px; flex-shrink:0; }
  .row-bar.HOLD_FOR_REVIEW{background:var(--hold)} .row-bar.STEP_UP_AUTH{background:var(--stepup)}
  .row-bar.WATCH{background:var(--watch)} .row-bar.APPROVE{background:var(--approve)}
  .row-body { flex:1; padding:14px 16px; }
  .row-top { display:flex; justify-content:space-between; gap:10px; align-items:baseline; }
  .token-id { font-family:'IBM Plex Mono',monospace; font-size:12.5px; }
  .stage { font-family:'IBM Plex Mono',monospace; font-size:10px; color:var(--muted); text-transform:uppercase; }
  .action-tag { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:.06em; padding:3px 8px; border-radius:4px; text-transform:uppercase; flex-shrink:0; }
  .action-tag.HOLD_FOR_REVIEW{background:#E5484D22;color:var(--hold)} .action-tag.STEP_UP_AUTH{background:#F0A93E22;color:var(--stepup)}
  .action-tag.WATCH{background:#5AA9E622;color:var(--watch)} .action-tag.APPROVE{background:#3FB68B22;color:var(--approve)}
  .score { font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--muted); }
  .reason { font-family:'IBM Plex Mono',monospace; font-size:11px; background:#0000002a; padding:3px 7px; border-radius:4px; display:inline-block; margin:6px 4px 0 0; }
  .empty { color:var(--muted); font-size:13px; font-family:'IBM Plex Mono',monospace; padding:20px 0; }
</style></head>
<body><div class="wrap">
  <div class="eyebrow">TokenSentry / Live</div>
  <h1><span class="dot"></span>Live decision feed</h1>
  <div class="sub">Polling every 1s. Send events via simulate_live_stream.py or curl.</div>
  <div id="queue"><div class="empty">Waiting for events...</div></div>
</div>
<script>
let lastSnapshot = null;
async function poll() {
  try {
    const res = await fetch('/queue');
    const entries = await res.json();
    const snapshot = JSON.stringify(entries.map(e => e.timestamp));
    if (snapshot === lastSnapshot) return;  // nothing changed -- skip the redraw entirely
    lastSnapshot = snapshot;

    const el = document.getElementById('queue');
    if (entries.length === 0) { el.innerHTML = '<div class="empty">Waiting for events...</div>'; return; }
    el.innerHTML = entries.map(e => `
      <div class="row">
        <div class="row-bar ${e.action}"></div>
        <div class="row-body">
          <div class="row-top">
            <span class="token-id">${e.token_id}</span>
            <span class="stage">${e.stage}</span>
            <span class="score">risk ${e.risk_score.toFixed(3)}</span>
            <span class="action-tag ${e.action}">${e.action.replace(/_/g,' ')}</span>
          </div>
          <div>${(e.reason_codes||[]).map(r => `<span class="reason">${r}</span>`).join('')}</div>
        </div>
      </div>`).join('');
  } catch (err) { console.error(err); }
}
poll();
setInterval(poll, 1000);
</script></body></html>"""

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


def top_reasons(explainer, feat_names, feat_values, k=3):
    x = np.array([feat_values], dtype=float)
    sv = explainer.shap_values(x)[0]
    idx = sorted(range(len(sv)), key=lambda i: -abs(sv[i]))[:k]
    return [reason_code(feat_names[i], feat_values[i]) for i in idx]


@app.route("/events/provisioning", methods=["POST"])
def provisioning_event():
    event = request.json
    required = ["token_id", "cardholder_id", "device_id", "provisioning_lat",
                "provisioning_lon", "auth_factor", "provisioning_ts"]
    missing = [f for f in required if f not in event]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    feats = store.score_provisioning(event)
    feat_values = [feats[f] for f in FEATS_P]
    proba = float(clf_p.predict_proba(pd.DataFrame([feat_values], columns=FEATS_P))[0, 1])

    if proba >= STEPUP_T_P:
        action = "STEP_UP_AUTH"
    elif proba >= WATCH_T_P:
        action = "WATCH"
    else:
        action = "APPROVE"

    reasons = top_reasons(explainer_p, FEATS_P, feat_values) if action != "APPROVE" else []
    entry = {
        "stage": "provisioning", "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_id": event["token_id"], "risk_score": round(proba, 4), "action": action,
        "reason_codes": reasons, "model_version": "tokensentry-modelP-v1",
    }
    if reasons:
        entry = investigate(entry)
    DECISION_LOG.insert(0, entry)
    print(f"[PROVISIONING] {event['token_id']}  score={proba:.3f}  -> {action}")
    return jsonify(entry)


@app.route("/events/transaction", methods=["POST"])
def transaction_event():
    payload = request.json
    token_id = payload.get("token_id")
    if not token_id:
        return jsonify({"error": "token_id required"}), 400
    try:
        feats = store.score_transaction(token_id, payload)
    except KeyError as e:
        return jsonify({"error": str(e)}), 404

    feat_values = [feats[f] for f in FEATS_B]
    proba = float(clf_b.predict_proba(pd.DataFrame([feat_values], columns=FEATS_B))[0, 1])

    if proba >= HOLD_T_B:
        action = "HOLD_FOR_REVIEW"
    elif proba >= STEPUP_T_B:
        action = "STEP_UP_AUTH"
    else:
        action = "APPROVE"

    reasons = top_reasons(explainer_b, FEATS_B, feat_values) if action != "APPROVE" else []
    entry = {
        "stage": "transaction", "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_id": token_id, "risk_score": round(proba, 4), "action": action,
        "amount_inr": payload.get("amount"), "reason_codes": reasons,
        "model_version": "tokensentry-modelB-v1",
    }
    if reasons:
        entry = investigate(entry)
    DECISION_LOG.insert(0, entry)
    print(f"[TRANSACTION]  {token_id}  score={proba:.3f}  -> {action}")
    return jsonify(entry)


@app.route("/queue", methods=["GET"])
def queue():
    return jsonify(DECISION_LOG)


@app.route("/dashboard", methods=["GET"])
def live_dashboard():
    return LIVE_DASHBOARD_HTML


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tokens_in_state": len(store.tokens),
                     "devices_in_state": len(store.device_to_cardholders)})


if __name__ == "__main__":
    print(f"Model P thresholds: STEP_UP>={STEPUP_T_P}  WATCH>={WATCH_T_P}")
    print(f"Model B thresholds: HOLD>={HOLD_T_B}  STEP_UP>={STEPUP_T_B}")
    app.run(host="127.0.0.1", port=5050, debug=False)
