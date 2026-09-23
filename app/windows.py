"""Earliest wall-clock window preview for a stored release plan.

Given a UTC *earliest-start instant*, a **positive duration** for every
service of the plan and, per service, a list of half-open
``[blocked_from, blocked_to)`` **blocked (no-publish) intervals**, the preview
projects the stored waves onto the time axis without changing their order:

* every member of a wave starts at the **same** instant;
* each member runs for exactly its reported duration, so a wave finishes at
  the **latest** member completion;
* a service may not run inside any of *its own* blocked intervals — its
  half-open window ``[start, start + duration)`` must not intersect any of
  them (touching an interval end is allowed, touching a start is not).

Waves are scheduled in stored order. Wave 1 is tried at the requested start
instant; if at that instant any member intersects one of its blocked
intervals, the whole wave is pushed to **after the end of the conflicting
interval(s)** and retried, repeatedly, until every member is clear. Every
later wave starts exactly at the previous wave's finish and undergoes the
same conflict push, so a delay cascades to all following waves. Repeatedly
restarting at the maximum conflicting interval end yields the *earliest*
feasible start: no feasible instant exists between the tried instant and
that end (it is forbidden by a member), and beyond it the argument repeats.

The request is validated first: every stored service must be reported exactly
once with a positive duration, and unknown services or degenerate blocked
intervals raise a structured :class:`~app.errors.PlanningError` (HTTP 422).
Durations are integer milliseconds and all instants are normalized to UTC;
sub-millisecond interval boundaries are supported. The stored plan is never
modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .errors import PlanningError
from .schemas import Issue, WindowPreviewRequest


@dataclass(frozen=True)
class ServiceInput:
    duration_ms: int
    blocked: list[tuple[datetime, datetime]]  # half-open [from, to), UTC


@dataclass(frozen=True)
class ScheduledWave:
    index: int  # 1-based
    started_at: datetime
    finished_at: datetime
    # service id -> (started_at, finished_at)
    members: dict[str, tuple[datetime, datetime]]


@dataclass(frozen=True)
class WindowPreviewOutput:
    start_at: datetime
    waves: list[ScheduledWave]
    finished_at: datetime
    total_duration_ms: int


def _as_utc(value: datetime) -> datetime:
    """Normalize an ISO 8601 instant to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _ms_between(start: datetime, end: datetime) -> int:
    """Whole milliseconds from ``start`` to ``end`` (truncated toward zero)."""
    return int((end - start).total_seconds() * 1000)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_windows(
    services: list[str], payload: WindowPreviewRequest
) -> dict[str, ServiceInput]:
    """Validate coverage, references and blocked intervals of the request."""
    known = set(services)
    issues: list[Issue] = []

    inputs: dict[str, ServiceInput] = {}
    seen: set[str] = set()
    for entry in payload.service_windows:
        if entry.service in seen:
            issues.append(
                Issue(
                    code="duplicate_service_window",
                    message=(
                        f"Service '{entry.service}' has more than one duration "
                        "and blocked-window entry."
                    ),
                    constraint="service_windows",
                    service=entry.service,
                )
            )
            continue
        seen.add(entry.service)

        blocked: list[tuple[datetime, datetime]] = []
        for window in entry.blocked_windows:
            blocked_from = _as_utc(window.blocked_from)
            blocked_to = _as_utc(window.blocked_to)
            if blocked_to <= blocked_from:
                issues.append(
                    Issue(
                        code="blocked_window_invalid",
                        message=(
                            f"Blocked window of service '{entry.service}' from "
                            f"{blocked_from.isoformat()} to "
                            f"{blocked_to.isoformat()} is empty or inverted; "
                            "blocked_to must be strictly after blocked_from."
                        ),
                        constraint="service_windows",
                        service=entry.service,
                        context={
                            "blocked_from": blocked_from.isoformat(),
                            "blocked_to": blocked_to.isoformat(),
                        },
                    )
                )
                continue
            blocked.append((blocked_from, blocked_to))

        # Unknown entries are still flagged for the client, but their durations
        # and blocked intervals never participate in the schedule.
        if entry.service in known:
            inputs[entry.service] = ServiceInput(
                duration_ms=entry.duration_ms, blocked=blocked
            )

    for s in sorted(seen - known):
        issues.append(
            Issue(
                code="unknown_service",
                message=(
                    f"Service '{s}' is not part of this plan and cannot appear "
                    "in its window preview."
                ),
                constraint="service_windows",
                service=s,
            )
        )

    for s in sorted(known - seen):
        issues.append(
            Issue(
                code="missing_service_window",
                message=(
                    f"Service '{s}' belongs to the plan but has no reported "
                    "duration and blocked-window entry."
                ),
                constraint="service_windows",
                service=s,
            )
        )

    if issues:
        # Deterministic order: offending service first, then issue code so the
        # response does not depend on the request list order.
        issues.sort(key=lambda i: (i.service or "", i.code))
        raise PlanningError(
            "The window preview does not cover exactly the services of the "
            "plan or contains invalid blocked intervals.",
            issues,
        )

    return inputs


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _earliest_wave_start(
    members: list[str],
    candidate: datetime,
    inputs: dict[str, ServiceInput],
) -> datetime:
    """Earliest instant >= ``candidate`` at which every member is publishable.

    A member conflicts when its half-open run window ``[t, t + duration)``
    intersects one of its half-open blocked intervals, i.e. when
    ``start < blocked_to and start + duration > blocked_from``. On conflict the
    wave restarts at the *maximum* offending interval end: every instant in
    [candidate, that end) stays forbidden by a member, while the bound strictly
    advances past a finite number of intervals, so the loop terminates at the
    earliest feasible start.
    """
    while True:
        next_candidate = candidate
        for service in members:
            entry = inputs[service]
            finish = candidate + timedelta(milliseconds=entry.duration_ms)
            for blocked_from, blocked_to in entry.blocked:
                if candidate < blocked_to and finish > blocked_from:
                    if blocked_to > next_candidate:
                        next_candidate = blocked_to
        if next_candidate == candidate:
            return candidate
        candidate = next_candidate


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def preview_windows(
    services: list[str],
    wave_members: list[list[str]],
    payload: WindowPreviewRequest,
) -> WindowPreviewOutput:
    """Compute the earliest feasible wall-clock window of every stored wave.

    ``services`` lists every service of the stored plan (sorted) and
    ``wave_members`` the stored waves in order; nothing is persisted by the
    caller.
    """
    inputs = _validate_windows(services, payload)
    start_at = _as_utc(payload.start_at)

    waves: list[ScheduledWave] = []
    cursor = start_at
    for index, members in enumerate(wave_members, start=1):
        wave_start = _earliest_wave_start(members, cursor, inputs)
        scheduled: dict[str, tuple[datetime, datetime]] = {}
        wave_finish = wave_start
        for service in sorted(members):
            entry = inputs[service]
            finish = wave_start + timedelta(milliseconds=entry.duration_ms)
            scheduled[service] = (wave_start, finish)
            if finish > wave_finish:
                wave_finish = finish
        waves.append(
            ScheduledWave(
                index=index,
                started_at=wave_start,
                finished_at=wave_finish,
                members=scheduled,
            )
        )
        # The next wave starts exactly when this one finished.
        cursor = wave_finish

    finished_at = waves[-1].finished_at if waves else start_at
    return WindowPreviewOutput(
        start_at=start_at,
        waves=waves,
        finished_at=finished_at,
        total_duration_ms=_ms_between(start_at, finished_at),
    )
