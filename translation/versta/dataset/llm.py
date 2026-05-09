import json
import time

import requests

from .config import (
    api_key,
    default_model,
    max_retries,
    request_timeout,
    retry_backoff,
    vllm_url,
)
from .types import TonalTranslation


def call_llm(
    payload: dict | None = None, retries: int | None = None, timeout: int | None = None
) -> dict | None:
    """Make an LLM inference call via HTTP POST with retries on timeout.

    Args:
        payload: Complete request payload for the chat completion endpoint.
        retries: Number of retry attempts on timeout. Defaults to config value.
        timeout: Request timeout in seconds. Defaults to config value.

    Returns:
        The response JSON as a dict, or None if all attempts failed.
    """
    retries = retries if retries is not None else max_retries()
    timeout = timeout if timeout is not None else request_timeout()

    url = vllm_url() + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }

    payload = payload or {}
    payload.setdefault("temperature", 0.3)
    payload.setdefault("reasoning_effort", "none")
    payload.setdefault("model", default_model())

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as e:
            backoff = retry_backoff() ** attempt
            print(
                f"Timeout on attempt {attempt + 1}/{retries} ({backoff}s backoff): {e}"
            )
            time.sleep(backoff)
        except Exception as e:
            print(f"LLM call failed: {e}")
            return None

    return None


def get_content(resp: dict | None) -> str | None:
    if not resp:
        return None
    return resp["choices"][0]["message"]["content"]


def generate_tonal_translations(
    source_text: str,
    target_text: str,
    source_lang: str,
    target_lang: str,
) -> TonalTranslation | None:
    """Generate formal, informal, and casual translations for a given source text.

    Args:
        source_text: The text in the source language.
        target_text: The existing translation in the target language.
        source_lang: Source language code (e.g. 'en').
        target_lang: Target language code (e.g. 'es').

    Returns:
        A dict with keys 'formal', 'informal', 'casual', or None on failure.
    """
    prompt = (
        f"Transform the translation from {source_lang} to {target_lang} into three tonalities."
        f"Provide three versions: formal, informal, and casual. "
        f"If you spot a big mistake in the translation, rewrite it or fix it."
        f"Some translations might be missing some numbers or special characters that the input text do have; Add these back into the text in the proper positions."
        f'Format as JSON: `{{"formal": "...", "informal": "...", "casual": "..."}}` \n\n'
        f"Text: {source_text}\nTranslation: {target_text}."
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a translation assistant specialized in tonal translations.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": 2048,
    }

    resp = call_llm(payload)
    content = get_content(resp)

    if not content:
        return None

    try:
        result = json.loads(content)
        return TonalTranslation(
            formal=result.get("formal", ""),
            informal=result.get("informal", ""),
            casual=result.get("casual", ""),
        )
    except json.JSONDecodeError:
        print(f"Bad JSON: {content[:300]}")
        return None
