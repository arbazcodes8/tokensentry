"""Builds docs/dashboard.html — a self-contained, static risk-ops dashboard
for the flagged-token queue. No server required; open the file directly."""
import json
import os
from paths import DATA_DIR as DATA, DOCS_DIR

with open(f"{DATA}/audit_log_with_notes.jsonl") as fh:
    entries = [json.loads(l) for l in fh]

entries.sort(key=lambda e: -e["risk_score"])

hold_n = sum(1 for e in entries if e["action"] == "HOLD_FOR_REVIEW")
stepup_n = sum(1 for e in entries if e["action"] == "STEP_UP_AUTH")

DATA_JSON = json.dumps(entries)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TokenSentry — Flagged Provisioning Queue</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #0B1220;
    --panel: #121A2B;
    --panel-hover: #16203590;
    --hairline: #263047;
    --text: #E7ECF5;
    --muted: #8593AD;
    --hold: #E5484D;
    --stepup: #F0A93E;
    --approve: #3FB68B;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ink);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    padding: 40px 24px 80px;
  }}
  .wrap {{ max-width: 920px; margin: 0 auto; }}
  .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 32px;
    margin: 0 0 28px;
    letter-spacing: -0.01em;
  }}
  .stats {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: var(--hairline);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 36px;
  }}
  .stat {{ background: var(--panel); padding: 18px 20px; }}
  .stat .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
  .stat .value {{ font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .stat .value.hold {{ color: var(--hold); }}
  .stat .value.stepup {{ color: var(--stepup); }}
  .stat .value.good {{ color: var(--approve); }}
  .section-label {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px;
    font-weight: 600;
    color: var(--muted);
    margin: 36px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--hairline);
  }}
  .row {{
    display: flex;
    background: var(--panel);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
    border: 1px solid var(--hairline);
  }}
  .row-rank {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    width: 36px;
    flex-shrink: 0;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 18px;
  }}
  .row-bar {{ width: 4px; flex-shrink: 0; }}
  .row-bar.HOLD_FOR_REVIEW {{ background: var(--hold); }}
  .row-bar.STEP_UP_AUTH {{ background: var(--stepup); }}
  .row-body {{ flex: 1; padding: 16px 18px; cursor: pointer; }}
  .row-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
  .token-id {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: var(--text); }}
  .action-tag {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.06em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    flex-shrink: 0;
  }}
  .action-tag.HOLD_FOR_REVIEW {{ background: #E5484D22; color: var(--hold); }}
  .action-tag.STEP_UP_AUTH {{ background: #F0A93E22; color: var(--stepup); }}
  .score {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .note {{ font-size: 13.5px; color: var(--muted); margin-top: 8px; line-height: 1.5; display: none; }}
  .row.open .note {{ display: block; }}
  .reasons {{ margin-top: 8px; display: none; }}
  .row.open .reasons {{ display: block; }}
  .reason {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--text);
    background: #0000002a;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
    margin: 3px 4px 0 0;
  }}
  footer {{
    margin-top: 48px;
    font-size: 11.5px;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    border-top: 1px solid var(--hairline);
    padding-top: 16px;
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">TokenSentry / Risk Ops</div>
  <h1>Flagged provisioning queue</h1>

  <div class="stats">
    <div class="stat"><div class="label">Model B recall</div><div class="value good">0.683</div></div>
    <div class="stat"><div class="label">Model B precision</div><div class="value good">0.935</div></div>
    <div class="stat"><div class="label">Held for review</div><div class="value hold">{hold_n}</div></div>
    <div class="stat"><div class="label">Step-up required</div><div class="value stepup">{stepup_n}</div></div>
  </div>

  <div class="section-label">Ranked by risk score — click a row for reason codes and the investigation note</div>
  <div id="queue"></div>

  <footer>
    Metrics from the last full run of train_eval.py + agent.py on the held-out test set (never-seen clusters).
    Regenerate via: python3 src/train_eval.py &amp;&amp; python3 src/agent.py &amp;&amp; python3 src/investigate.py &amp;&amp; python3 src/build_dashboard.py
  </footer>
</div>

<script>
const entries = {DATA_JSON};
const queue = document.getElementById('queue');
entries.forEach((e, i) => {{
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <div class="row-rank">${{String(i+1).padStart(2,'0')}}</div>
    <div class="row-bar ${{e.action}}"></div>
    <div class="row-body">
      <div class="row-top">
        <span class="token-id">${{e.token_id}}</span>
        <span class="score">risk ${{e.risk_score.toFixed(3)}}</span>
        <span class="action-tag ${{e.action}}">${{e.action.replace(/_/g,' ')}}</span>
      </div>
      <div class="reasons">${{e.reason_codes.map(r => `<span class="reason">${{r}}</span>`).join('')}}</div>
      <div class="note">${{e.investigation_note}}</div>
    </div>
  `;
  row.querySelector('.row-body').addEventListener('click', () => row.classList.toggle('open'));
  queue.appendChild(row);
}});
</script>
</body>
</html>
"""

with open(os.path.join(DOCS_DIR, "dashboard.html"), "w") as fh:
    fh.write(HTML)

print(f"Built docs/dashboard.html with {len(entries)} flagged entries "
      f"({hold_n} hold, {stepup_n} step-up)")
