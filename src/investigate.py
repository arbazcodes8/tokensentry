"""
Turns a flagged token's SHAP reason codes into a short, human-readable
investigation note a fraud analyst can act on immediately.

Design choice, stated plainly: a judge/panelist reviewing this repo should
NOT need their own Anthropic API key to run it. So this module works two
ways:
  - ANTHROPIC_API_KEY set  -> calls Claude for a natural-language note
  - no key set             -> deterministic template, same information,
                               slightly more mechanical phrasing

Both paths produce the same structured fields (summary, recommended_action,
confidence_note) so downstream code never needs to know which path ran.
"""
import os
import json

SYSTEM_PROMPT = (
    "You are a fraud-risk assistant writing a one-paragraph investigation "
    "note for a human analyst reviewing a flagged card-tokenisation event "
    "at a payments company. You are given a risk score, a bounded action "
    "already decided by the system (you do not choose or change it), and "
    "up to three reason codes. Write 2-3 plain sentences: what looks "
    "suspicious, referencing the reason codes, and what the analyst should "
    "specifically verify. Do not invent facts not present in the reason "
    "codes. Do not recommend blocking the cardholder permanently — only "
    "the given bounded action. Keep it under 60 words."
)


def _template_note(entry):
    reasons = "; ".join(entry["reason_codes"])
    action_text = {
        "HOLD_FOR_REVIEW": "held for manual review before any transaction is allowed",
        "STEP_UP_AUTH": "asked for an additional verification factor",
    }.get(entry["action"], "flagged for monitoring")
    return (
        f"Token {entry['token_id']} scored {entry['risk_score']:.2f} risk. "
        f"Flagged because: {reasons}. Recommended: {action_text}. "
        f"Analyst should confirm the cardholder recognises this device and "
        f"this token-requestor before releasing the hold."
    )


def investigate(entry, use_llm=None):
    """entry: one audit_log.jsonl record (dict). Returns entry + 'investigation_note'."""
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not use_llm:
        note = _template_note(entry)
        entry_out = dict(entry, investigation_note=note, note_source="template_fallback")
        return entry_out

    try:
        import anthropic
        client = anthropic.Anthropic()
        user_content = json.dumps({
            "risk_score": entry["risk_score"],
            "action": entry["action"],
            "reason_codes": entry["reason_codes"],
        })
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        note = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        entry_out = dict(entry, investigation_note=note, note_source="claude")
        return entry_out
    except Exception as e:
        # never let a missing/invalid key or network hiccup break the pipeline
        note = _template_note(entry)
        entry_out = dict(entry, investigation_note=note, note_source=f"template_fallback (llm_error: {e.__class__.__name__})")
        return entry_out


if __name__ == "__main__":
    from paths import DATA_DIR as DATA
    with open(f"{DATA}/audit_log.jsonl") as fh:
        entries = [json.loads(l) for l in fh]

    enriched = [investigate(e) for e in entries]

    with open(f"{DATA}/audit_log_with_notes.jsonl", "w") as fh:
        for e in enriched:
            fh.write(json.dumps(e) + "\n")

    print(f"Enriched {len(enriched)} audit entries with investigation notes "
          f"(source: {enriched[0]['note_source'] if enriched else 'n/a'})")
    print("\nExample:")
    print(json.dumps(enriched[0], indent=2))
