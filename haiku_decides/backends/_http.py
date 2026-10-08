from __future__ import annotations

import asyncio

import httpx

RETRYABLE_STATUS = {408, 409, 429}


async def post_json(
    http: httpx.AsyncClient,
    url: str,
    *,
    headers: dict,
    payload: dict,
    max_attempts: int = 3,
    sleep=asyncio.sleep,
) -> tuple[dict | None, str | None]:
    """POST JSON with retries. Returns (body, None) on success or (None, error) on failure."""
    error = None
    for attempt in range(max_attempts):
        try:
            response = await http.post(url, headers=headers, json=payload)
        except httpx.TransportError as e:
            # error text is persisted in result files, so it carries no response body or message
            error = type(e).__name__
        else:
            if response.status_code < 300:
                try:
                    return response.json(), None
                except ValueError:
                    return None, "response body is not JSON"
            error = f"HTTP {response.status_code}"
            if response.status_code not in RETRYABLE_STATUS and response.status_code < 500:
                return None, error
        if attempt + 1 < max_attempts:
            await sleep(2**attempt)
    return None, error
