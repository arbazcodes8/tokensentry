# Pitch Script — TokenSentry (5 minutes)

Read this as a guide, not a script to memorize word-for-word — say it in your own voice. Timings are approximate; practice once with a timer and trim, don't rush.

---

### 0:00–0:40 — The hook (screen: the token lifecycle diagram or a blank slide with just the title)

"Every time someone saves a card on an app, Razorpay's TokenHQ swaps that card for a token — that's how RBI's tokenisation rules work. Most fraud systems only start watching *after* that token gets used in a transaction. We built an agent that watches the moment the token is *created* — because that's where the fraud signal actually starts, and almost nobody is looking there."

### 0:40–1:30 — The problem, made concrete (screen: your token lifecycle diagram)

Walk through it plainly:
- A card gets provisioned — a token is born.
- A fraudster who's stolen card details wants to move fast: provision it, use it, before anyone notices.
- The gap between "token created" and "token first used" — device, location, timing — is exactly where that urgency shows up. That gap is provisioning intelligence, and it's what our baseline models throw away.

### 1:30–2:15 — Why this is Razorpay's problem, specifically (screen: README's "why now" section, or just talk)

"This isn't hypothetical for Razorpay. TokenHQ is Razorpay acting as a Token Requestor, provisioning tokens at scale, right now. RBI's stronger-authentication tokenisation rules are due April 1, 2026 — this is live, dated regulation."

Then, proactively — don't wait for a panelist to bring this up: "Visa and Mastercard already score provisioning risk for issuers — Visa's own numbers put token provisioning fraud at $809M a year globally. But that scoring happens at the network-to-issuer layer, before the issuer approves a request. We built the layer downstream of that — the PSP's own safety net, for tokens that already cleared the network's check. Mastercard's own fraud literature warns this is exactly where under-monitoring happens, because everyone assumes provisioning was already vetted."

### 2:15–3:30 — The build, honestly (screen: the Model A vs B results table)

This is your strongest material — don't rush it:
- "We tested one specific hypothesis: does provisioning intelligence actually improve fraud detection over transaction data alone, or is that just a nice story? So we built two models on identical data and an identical held-out test set — Model A sees only transaction behaviour, Model B adds provisioning signal: device history, geo-velocity, provisioning-to-transaction latency, auth factor."
- "Model A: 69% precision, 40% recall. Model B: 93% precision, 68% recall. We ran McNemar's test on the paired predictions — p less than 0.00001. That's not noise."
- "And we checked *where* the lift comes from — it's not just catching repeat devices. Even fraud with zero device-reuse signal at all — a device used exactly once — still improved by 13 points of recall. Provisioning intelligence is pulling real weight beyond graph structure."

### 3:30–4:15 — The agent, not just the model (screen: docs/dashboard.html, click open a flagged row)

- "The model only outputs a number. The agent is what turns that into something a human can act on."
- Click a flagged row live: "Here's a token that scored 0.99. The agent explains itself with SHAP-based reason codes — this device has provisioned tokens for seven other cardholders, here's the average transaction amount — and it takes one of three bounded actions: approve, ask for step-up authentication, or hold for a human analyst. It never bans anyone on its own. Every decision is logged to an audit trail."
- If you wired up the LLM investigation-note layer, show one: "It can also write the analyst a plain-English investigation note — and if there's no API key available, it falls back to a clear template automatically, so this never breaks."

### 4:15–4:35 — It's actually real-time, not just a static report (screen: terminal running live_server.py)

- "Everything so far was a batch evaluation, to prove the numbers honestly. But fraud has to be caught the moment it happens, not overnight — so we also built a live version."
- Run `simulate_live_stream.py` on screen, live: "Watch this device provision a card for a seventh different person — same device, six different identities already. Watch it escalate in real time: approve, watch, step-up, hold — as evidence accumulates, before a single rupee has moved."
- Be upfront if asked: "This runs as a synchronous scoring API, not a Kafka pipeline — that was a deliberate scope call for two weeks, not a gap in understanding. The scoring logic is what's hard and what's tested; in production it sits behind a real message broker instead of a Flask route, unchanged."

### 4:35–4:50 — What's honest about this, on purpose (screen: back to README or just talk)

"We want to be upfront about the parts of this that are synthetic: all data here is generated, and fraud prevalence is elevated above real-world rates so we'd have enough test cases to measure honestly. The methodology — cluster-level train/test split so nothing leaks, thresholds chosen on validation and never touched again on test, a statistical significance test instead of just a headline number — that part is real, and it's the part we think matters most for a system that's allowed to touch money."

### 4:50–5:00 — Close

"TokenSentry: watch the moment the token is born, not just the moment it's spent. Thanks."

---

## Recording notes
- Screen-record the dashboard interaction live — don't just describe it, click through it.
- Have the results table on screen while you say the numbers out loud; don't make people read and listen at the same time for the important part.
- If you get panel questions afterward, three things are most likely to get probed: the McNemar's test and cluster-level split (know exactly why each was necessary — leakage prevention, statistical rigor), and the VPI/Mastercard comparison (know the one-line answer cold: network-to-issuer, pre-provisioning vs. PSP-side, post-provisioning — don't fumble this one, it's the easiest to sound prepared on).
