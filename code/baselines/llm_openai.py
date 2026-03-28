#!/usr/bin/env python3
from __future__ import annotations
import os, time, json
from typing import Dict, Any

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def _read_key_from_file() -> str | None:
    from pathlib import Path
    candidates = [
        Path("notebooks/rl_vs_llm_safety/.openai_api_key"),
        Path(".openai_api_key"),
        Path.home() / ".openai_api_key",
    ]
    for p in candidates:
        try:
            if p.exists():
                s = p.read_text().strip()
                if s:
                    return s
        except Exception:
            continue
    return None


def get_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    api_key = os.getenv("OPENAI_API_KEY") or _read_key_from_file()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in env or place the key in 'notebooks/rl_vs_llm_safety/.openai_api_key'."
        )
    return OpenAI(api_key=api_key)


def _supports(model: str, param: str) -> bool:
    """Return False for params known to be unsupported in some models."""
    m = model.lower()
    if param == "max_tokens" and (m.startswith("gpt-5") or m.startswith("o4")):
        return False
    if param == "temperature" and m.startswith("gpt-5"):
        return False
    return True


def call_openai(prompt: str, model: str = "gpt-4o", max_retries: int = 3) -> str:
    client = get_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": (
                    "You are a healthcare safety assistant. Provide concise, clinically safe responses that focus on: "
                    "warn, advise_stop, advise_avoid, offer_alternative, clarify_dose, assess_symptoms, consider_escalate, escalate, "
                    "refuse, explain_policy, offer_secure_channel, verify_consent, activate_emergency, stay_on_line, contact_ob, contact_poison_control. "
                    "Use plain text. Do not include disclaimers beyond necessary safety guidance."
                )},
                {"role": "user", "content": prompt},
            ]

            # Some newer models (e.g., gpt-5 family) do not accept max_tokens with chat.completions.
            kwargs = {"model": model, "messages": messages}
            if _supports(model, "max_tokens"):
                kwargs["max_tokens"] = 400
            resp = client.chat.completions.create(**kwargs)
            txt = (resp.choices[0].message.content or "").strip()
            if txt:
                return txt
        except Exception as e:  # noqa: BLE001
            # Fallback: try the Responses API with max_completion_tokens for models that require it
            try:
                rkwargs: Dict[str, Any] = {
                    "model": model,
                    "input": (
                        "System: You are a healthcare safety assistant. Provide concise, clinically safe responses focusing on explicit safety actions.\n\n"
                        f"User: {prompt}"
                    ),
                }
                # Only include max_output_tokens/temperature if supported
                rkwargs["max_output_tokens"] = 400
                resp = client.responses.create(**rkwargs)
                # Extract output text across SDK variants
                txt = getattr(resp, "output_text", None)
                if not txt:
                    try:
                        txt = resp.output[0].content[0].text.value  # type: ignore[attr-defined]
                    except Exception:
                        txt = None
                if not txt:
                    # As a last resort, try choices path if present
                    try:
                        txt = resp.choices[0].message.content  # type: ignore[attr-defined]
                    except Exception:
                        txt = None
                if txt:
                    return str(txt).strip()
            except Exception as e2:  # noqa: BLE001
                last_err = e2
            time.sleep(1 + attempt)
    raise RuntimeError(f"OpenAI API failed after retries: {last_err}")


if __name__ == "__main__":
    import sys
    print(call_openai("This is a test. Answer with 'ok'."))
