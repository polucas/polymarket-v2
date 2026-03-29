from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

_simulated_time: Optional[datetime] = None


class Clock:
    """Singleton controlling simulated time for backtesting.

    In live mode (_simulated_time is None), utcnow() returns real wall time —
    zero behavior change in production.
    """

    @classmethod
    def set_time(cls, dt: datetime) -> None:
        global _simulated_time
        _simulated_time = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

    @classmethod
    def utcnow(cls) -> datetime:
        if _simulated_time is None:
            return datetime.now(timezone.utc)
        return _simulated_time

    @classmethod
    def advance(cls, minutes: int) -> None:
        global _simulated_time
        if _simulated_time is None:
            _simulated_time = datetime.now(timezone.utc)
        _simulated_time += timedelta(minutes=minutes)

    @classmethod
    def reset(cls) -> None:
        """Reset to live mode. Used in tests to ensure isolation."""
        global _simulated_time
        _simulated_time = None

    @classmethod
    def is_simulated(cls) -> bool:
        return _simulated_time is not None
