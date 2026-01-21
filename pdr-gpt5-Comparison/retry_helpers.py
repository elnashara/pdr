# retry_helpers.py
import time, random, json
import openai

# Older SDK exposes exceptions under openai.error.*
try:
    from openai import error as oe  # type: ignore
except Exception:
    oe = None  # running on a newer SDK

def _status_code(e):
    for attr in ("status", "status_code", "http_status", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    s = str(e)
    if "Error code 520" in s or "code 520" in s:
        return 520
    return None

def _is_retryable(e) -> bool:
    s = _status_code(e)
    if s is not None and (s == 429 or 500 <= s <= 599):
        return True

    # Newer SDK exception classes (may or may not exist)
    if isinstance(e, tuple(filter(None, (
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APIError", None),
    )))):
        return True

    # Older SDK exceptions
    if oe and isinstance(e, tuple(filter(None, (
        getattr(oe, "APIConnectionError", None),
        getattr(oe, "Timeout", None),
        getattr(oe, "RateLimitError", None),
        getattr(oe, "ServiceUnavailableError", None),
        getattr(oe, "TryAgain", None),
        getattr(oe, "APIError", None),
    )))):
        return True

    # Last resort: Cloudflare HTML page / non-JSON response
    txt = str(e).lower()
    if "<!doctype html>" in txt or "cloudflare" in txt:
        return True

    return False

def chat_with_retries(
    *,
    max_attempts: int = 6,
    base: float = 0.5,
    cap: float = 10.0,
    jitter: float = 0.3,
    request_timeout: int = 90,
    **kwargs
):
    """
    Wrapper for openai.ChatCompletion.create with exponential backoff + jitter.
    Retries on 429, 5xx, and network-ish failures. Passes through **kwargs.
    """
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            return openai.ChatCompletion.create(request_timeout=request_timeout, **kwargs)
        except Exception as e:
            last_err = e
            if not _is_retryable(e) or attempt >= max_attempts:
                raise
            sleep = min(cap, base * (2 ** (attempt - 1))) * (1.0 + jitter * random.random())
            time.sleep(sleep)
    raise last_err
