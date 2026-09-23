"""Read-only deployment-window preview for a stored release plan.

Given a UTC earliest start and, for every stored service, a positive release
duration plus its own no-publish (blocked) intervals, the preview lays the
*stored* wave order onto a time axis:

* the services of one wave all start at the same instant, and the wave ends
  at the latest service completion (the maximum member duration);
* a half-open blocked interval ``[start_at, end_at)`` of any member that
  intersects the member's own release window ``[wave_start, wave_start +
  duration)`` makes the candidate start infeasible; the whole wave is then
  postponed to the end of the conflicting interval and recomputed (which may
  push it past a later interval, so the check repeats);
* the first wave is tried at the requested start, every later wave starts at
  the previous wave's finish instant.

Intervals are left-closed/right-open, so starting exactly when a blocked
interval ends — or finishing exactly when one begins — is allowed. Every
stored service must be specified exactly once with a positive duration;
unknown services, duplicate specs, non-positive durations and inverted
blocked intervals raise a structured :class:`~app.errors.PlanningError`
(HTTP 422). All outputs are deterministically ordered and the stored plan is
never modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .errors import PlanningError
from .schemas import Issue, WindowPreviewRequest, WindowPreviewWave


@dataclass(frozen=True)
class _Spec:
    duration_ms: int
    # Blocked intervals as integer-millisecond epoch pairs [start, end),
    # sorted by (start, end) so iteration and conflict picks are deterministic.
    blocked: list[tuple[int, int]]


def _as_utc(value: datetime) -> datetime:
    """Normalize an ISO 8601 instant to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _epoch_ms(value: datetime) -> int:
    """Whole milliseconds since the epoch (truncated toward zero)."""
    return int(value.timestamp() * 1000)


def _from_epoch_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_services(
    services: list[str], payload: WindowPreviewRequest
) -> dict[str, _Spec]:
    """Validate coverage, durations and blocked intervals of the request."""
    known = set(services)
    issues: list[Issue] = []

    specs: dict[str, _Spec] = {}
    seen: set[str] = set()
    for entry in payload.services:
        if entry.service in seen:
            issues.append(
                Issue(
                    code="duplicate_service_window",
                    message=(
                        f"Service '{entry.service}' has more than one duration "
                        "and blocked-window specification."
                    ),
                    constraint="services",
                    service=entry.service,
                )
            )
            continue
        seen.add(entry.service)

        if entry.service not in known:
            issues.append(
                Issue(
                    code="unknown_service",
                    message=(
                        f"Service '{entry.service}' is not part of this plan "
                        "and cannot appear in its window preview."
                    ),
                    constraint="services",
                    service=entry.service,
                )
            )

        if entry.duration_ms <= 0:
            issues.append(
                Issue(
                    code="non_positive_service_duration",
                    message=(
                        f"Service '{entry.service}' has duration "
                        f"{entry.duration_ms} ms; release durations must be a "
                        "positive number of milliseconds."
                    ),
                    constraint="services",
                    service=entry.service,
                    context={"duration_ms": entry.duration_ms},
                )
            )

        blocked: list[tuple[int, int]] = []
        for window in entry.blocked_windows:
            start = _epoch_ms(_as_utc(window.start_at))
            end = _epoch_ms(_as_utc(window.end_at))
            if end < start:
                issues.append(
                    Issue(
                        code="blocked_window_inverted",
                        message=(
                            f"Service '{entry.service}' has a blocked interval "
                            "ending before it starts "
                            f"({window.end_at.isoformat()} < "
                            f"{window.start_at.isoformat()})."
                        ),
                        constraint="services",
                        service=entry.service,
                        context={
                            "start_at": _as_utc(window.start_at).isoformat(),
                            "end_at": _as_utc(window.end_at).isoformat(),
                        },
                    )
                )
                continue
            blocked.append((start, end))

        if entry.duration_ms > 0 and entry.service in known:
            blocked.sort()
            specs[entry.service] = _Spec(duration_ms=entry.duration_ms, blocked=blocked)

    for s in sorted(known - seen):
        issues.append(
            Issue(
                code="missing_service_window",
                message=(
                    f"Service '{s}' belongs to the plan but has no positive "
                    "duration and blocked-window specification."
                ),
                constraint="services",
                service=s,
            )
        )

    if issues:
        # Deterministic order even when the request lists entries out of order.
        issues.sort(key=lambda i: (i.service or "", i.code, i.constraint))
        raise PlanningError(
            "The window preview does not cover exactly the services of the "
            "plan with positive durations and well-formed blocked intervals.",
            issues,
        )

    return specs


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def _schedule_wave_start(
    members: list[str], start_ms: int, specs: dict[str, _Spec]
) -> int:
    """Earliest feasible common start of one wave, tried at/after ``start_ms``.

    The candidate is feasible only when every member's own release window
    ``[candidate, candidate + duration)`` is disjoint from each of its
    half-open blocked intervals. On any intersection the whole wave moves to
    the end of the *latest* conflicting interval and is retried: jumping past
    every interval conflicting at the current candidate skips only starts
    that some member would already reject, so the fixed point reached is the
    earliest feasible start.
    """
    candidate = start_ms
    while True:
        resume_at = candidate
        for service in members:
            spec = specs[service]
            finish = candidate + spec.duration_ms
            for block_start, block_end in spec.blocked:
                if block_end == block_start:
                    continue  # empty [x, x) interval never blocks
                # Two half-open intervals [candidate, finish) and
                # [block_start, block_end) are disjoint exactly when one ends
                # at or before the other starts.
                if block_end <= candidate or block_start >= finish:
                    continue
                if block_end > resume_at:
                    resume_at = block_end
        if resume_at == candidate:
            return candidate
        candidate = resume_at


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowPreviewOutput:
    start_at: datetime
    waves: list[WindowPreviewWave]
    total_duration_ms: int


def preview_windows(
    services: list[str],
    wave_members: list[list[str]],
    payload: WindowPreviewRequest,
) -> WindowPreviewOutput:
    """Compute the earliest feasible calendar windows of the stored waves.

    ``services`` lists every service of the stored plan and ``wave_members``
    the stored waves in order. The first wave is tried at the payload's
    ``start_at``; each following wave starts where the previous one finished.
    Nothing is persisted by the caller.
    """
    specs = _validate_services(services, payload)

    start_at = _as_utc(payload.start_at)
    cursor = _epoch_ms(start_at)

    waves: list[WindowPreviewWave] = []
    for index, members in enumerate(wave_members):
        ordered = sorted(members)
        wave_start = _schedule_wave_start(ordered, cursor, specs)
        wave_finish = wave_start + max(specs[s].duration_ms for s in ordered)
        waves.append(
            WindowPreviewWave(
                index=index + 1,
                started_at=_from_epoch_ms(wave_start),
                finished_at=_from_epoch_ms(wave_finish),
                duration_ms=wave_finish - wave_start,
                services=ordered,
            )
        )
        cursor = wave_finish

    total_duration_ms = _epoch_ms(waves[-1].finished_at) - _epoch_ms(
        waves[0].started_at
    )
    return WindowPreviewOutput(
        start_at=start_at, waves=waves, total_duration_ms=total_duration_ms
    )
