import json
import time

import requests

from .config import (
    api_key,
    max_retries,
    request_timeout,
    retry_backoff,
    vllm_url,
)
from .types import TonalTranslation


def call_llm(
    prompts: list[str],
    retries: int | None = None,
    timeout: int | None = None,
) -> list[str | None]:
    """Make a batched LLM inference call via HTTP POST with retries on timeout.

    Uses the /chat/completions/batch endpoint with a list of messages for batching.

    Args:
        prompts: List of prompt strings to send in a single batch.
        retries: Number of retry attempts on timeout. Defaults to config value.
        timeout: Request timeout in seconds. Defaults to config value.

    Returns:
        List of response content strings (one per prompt), or None for failed items.
    """
    retries = retries if retries is not None else max_retries()
    timeout = timeout if timeout is not None else request_timeout()

    url = vllm_url() + "/chat/completions/batch"
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }

    messages = [
        [
            {
                "role": "system",
                "content": "You are a translation assistant specialized in tonal translations.",
            },
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    payload = {
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 2048,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            resp_data = resp.json()
            return [
                choice.get("message", {}).get("content")
                for choice in resp_data.get("choices", [])
            ]
        except requests.exceptions.Timeout as e:
            backoff = retry_backoff() ** attempt
            print(
                f"Timeout on attempt {attempt + 1}/{retries} ({backoff}s backoff): {e}"
            )
            time.sleep(backoff)
        except Exception as e:
            print(f"LLM batch call failed: {e}")
            return [None] * len(prompts)

    return [None] * len(prompts)


def generate_tonal_translations(
    pairs: list[dict],
    source_lang: str,
    target_lang: str,
) -> list[TonalTranslation | None]:
    """Generate tonal translations for multiple sentence pairs in a single batched request.

    Args:
        pairs: List of dicts with 'prompt' and 'completion' keys.
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        List of TonalTranslation dicts or None for each pair (None on failure).
    """
    if not pairs:
        return []

    prompts = []
    for pair in pairs:
        prompt = (
            f"Transform the translation from {source_lang} to {target_lang} into three tonalities. "
            "Provide three versions: formal, neutral, and casual. "
            "If you spot a big mistake in the translation, rewrite it or fix it. "
            "Some translations might be missing some numbers or special characters that the input text do have; Add these back into the text in the proper positions. "
            "Critical Constraint: Do not use filler words unless they are explicitly in the source or translated text. Avoid dialect slang."
            'Format as JSON: `{{"formal": "...", "neutral": "...", "casual": "..."}}`\n\n'
            f"Text: {pair['prompt']}\nTranslation: {pair['completion']}."
        )
        prompts.append(prompt)

    contents = call_llm(prompts)

    results: list[TonalTranslation | None] = []
    for content in contents:
        if not content:
            results.append(None)
            continue
        try:
            result = json.loads(content)
            results.append(
                TonalTranslation(
                    formal=result.get("formal", ""),
                    neutral=result.get("neutral", ""),
                    casual=result.get("casual", ""),
                )
            )
        except json.JSONDecodeError:
            print(f"Bad JSON: {content[:300]}")
            results.append(None)

    return results
