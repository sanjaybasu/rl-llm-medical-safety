#!/usr/bin/env python3
"""
Lightweight helper for calling Anthropic Claude models with retry logic.

Reads the API key from the ANTHROPIC_API_KEY environment variable or from
`notebooks/rl_vs_llm_safety/.anthropic_api_key` (single-line file).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - import errors handled dynamically
    Anthropic = None  # type: ignore


def _read_key() -> str | None:
    """Read Anthropic API key from env or known files."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    candidates = [
        Path("notebooks/rl_vs_llm_safety/.anthropic_api_key"),
        Path(".anthropic_api_key"),
        Path.home() / ".anthropic_api_key",
    ]
    for p in candidates:
        try:
            if p.exists():
                text = p.read_text().strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def get_client() -> Any:
    """Return an Anthropic client or raise with guidance."""
    if Anthropic is None:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    api_key = _read_key()
    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. "
            "Set ANTHROPIC_API_KEY or place it in notebooks/rl_vs_llm_safety/.anthropic_api_key."
        )
    return Anthropic(api_key=api_key)


def call_anthropic(
    prompt: str,
    model: str = "claude-3-5-sonnet-20240620",
    max_retries: int = 3,
) -> str:
    """Call the Anthropic Messages API with healthcare safety instructions."""
    client = get_client()
    last_err: Exception | None = None
    system_prompt = (
        "You are a healthcare safety assistant. Provide concise, clinically safe responses "
        "that focus on explicit safety actions (warn, advise_stop, advise_avoid, offer_alternative, "
        "clarify_dose, assess_symptoms, consider_escalate, escalate, refuse, explain_policy, "
        "offer_secure_channel, verify_consent, activate_emergency, stay_on_line, contact_ob, "
        "contact_poison_control). Use plain text. Do not include disclaimers beyond necessary safety guidance."
    )

    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            # New Anthropic SDK returns list of content blocks
            content = msg.content
            if content and hasattr(content[0], "text"):
                return content[0].text.strip()
            if content and isinstance(content[0], dict) and "text" in content[0]:
                return str(content[0]["text"]).strip()
            raise RuntimeError("Unexpected Anthropic response structure")
        except Exception as e:  # pragma: no cover - network errors handled at runtime
            last_err = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"Anthropic API failed after retries: {last_err}")


if __name__ == "__main__":  # pragma: no cover
    print(call_anthropic("This is a test. Reply 'ok'."))
