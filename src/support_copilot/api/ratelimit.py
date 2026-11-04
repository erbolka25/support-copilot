"""Per-IP rate limit for the expensive endpoints (LLM drafts, uploads) of a public demo."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

WINDOW_S = 3600.0


class RateLimiter:
    """Sliding one-hour window per key, in memory (one process is enough for the demo)."""

    def __init__(self, per_hour: int, clock=time.monotonic) -> None:
        self.per_hour = per_hour
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float:
        """Record a request; return 0 if allowed, else the seconds until the next free slot."""
        now = self._clock()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - WINDOW_S:
                hits.popleft()
            if len(hits) >= self.per_hour:
                return hits[0] + WINDOW_S - now
            hits.append(now)
            return 0.0


def client_ip(request: Request) -> str:
    """The client behind the proxy (Hugging Face Spaces sets X-Forwarded-For), else the peer.

    Best effort: the header can be forged, the LLM provider's own quota is the hard limit.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request) -> None:
    """Dependency: 429 once this IP spent its hourly budget; no-op when the limit is 0."""
    limiter: RateLimiter | None = request.app.state.rate_limiter
    if limiter is None:
        return
    wait = limiter.hit(client_ip(request))
    if wait > 0:
        minutes = math.ceil(wait / 60)
        raise HTTPException(
            429,
            detail={
                "code": "rate_limited",
                "message": f"Demo limit: {limiter.per_hour} requests per hour. "
                f"Try again in {minutes} min.",
            },
            headers={"Retry-After": str(math.ceil(wait))},
        )
