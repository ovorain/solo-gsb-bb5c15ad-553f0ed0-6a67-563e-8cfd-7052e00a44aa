"""Read-only responsibility-domain (ownership) audit for a stored release plan.

Given a *service -> team* ownership map and a per-team simultaneous on-call
cap, the audit projects the stored dependency graph, wave assignment and
fault domains onto teams:

* **cross-team dependency chains** — every stored dependency edge whose
  endpoints are owned by different teams is reported, and the quotient graph
  on teams (edge ``downstream_team -> upstream_team`` for every such edge)
  is ordered by dependency topology so the teams that must coordinate first
  (providers of prerequisites) come first;
* **unowned services** — plan services absent from the ownership mapping are
  audit findings, not request errors;
* **fault-domain team coupling** — a fault domain couples the teams that own
  its members, so a failure of the domain hits several teams at once;
* **on-call overload** — for every stored wave the number of a team's
  services simultaneously in that wave is checked against its cap.

All lists are deterministically ordered (sorted ids / stored order,
lexicographic tie-breaks) and the result does not depend on the ordering of
the request lists. The request itself is validated first: unknown service
references, a service mapped twice, a blank owning team, a missing cap for a
referenced team or a non-positive cap raise a structured
:class:`~app.errors.PlanningError` (HTTP 422). The stored plan is never
modified.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

from .errors import PlanningError
from .schemas import (
    CrossTeamDependency,
    FaultDomainTeamCoupling,
    Issue,
    OwnershipAuditRequest,
    OwnershipTeamSummary,
    TeamOncallOverlimit,
)


@dataclass(frozen=True)
class OwnershipAuditOutput:
    teams: list[OwnershipTeamSummary]
    unowned_services: list[str]
    cross_team_dependencies: list[CrossTeamDependency]
    coordination_order: list[str]
    fault_domain_coupling: list[FaultDomainTeamCoupling]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_ownership(
    services: list[str], payload: OwnershipAuditRequest
) -> dict[str, str]:
    """Validate the ownership map and caps; return the service -> team map.

    Rejects unknown services, a service mapped more than once, blank owning
    teams and missing/non-positive per-team caps. Coverage is intentionally
    *not* required: services without an entry are unowned findings.
    """
    known = set(services)
    issues: list[Issue] = []

    owner_of: dict[str, str] = {}
    referenced_teams: set[str] = set()
    for entry in payload.ownership:
        if not entry.team:
            issues.append(
                Issue(
                    code="missing_service_owner",
                    message=(
                        f"Service '{entry.service}' has no owning team: the "
                        "team id must not be blank."
                    ),
                    constraint="ownership",
                    service=entry.service,
                )
            )
            continue
        if entry.service in owner_of:
            issues.append(
                Issue(
                    code="duplicate_service_owner",
                    message=(
                        f"Service '{entry.service}' is mapped to a team more "
                        "than once; each service may have at most one owner."
                    ),
                    constraint="ownership",
                    service=entry.service,
                    context={
                        "teams": sorted(
                            {owner_of[entry.service], entry.team}
                        ),
                    },
                )
            )
            continue
        if entry.service not in known:
            issues.append(
                Issue(
                    code="unknown_service",
                    message=(
                        f"Service '{entry.service}' is not part of this plan "
                        "and cannot appear in its ownership audit."
                    ),
                    constraint="ownership",
                    service=entry.service,
                )
            )
            continue
        owner_of[entry.service] = entry.team
        referenced_teams.add(entry.team)

    # The JSON schema (ge=1) rejects non-positive and non-integer caps before
    # this point; here a team with owned services but no declared cap is a
    # semantic gap the client must fix.
    for team in sorted(referenced_teams - set(payload.team_oncall_caps)):
        issues.append(
            Issue(
                code="team_oncall_cap_missing",
                message=(
                    f"Team '{team}' owns services but no simultaneous on-call "
                    "cap is declared for it in team_oncall_caps."
                ),
                constraint="team_oncall_caps",
                context={"team": team},
            )
        )

    if issues:
        raise PlanningError(
            "The ownership audit references unknown or duplicate services, "
            "blank owners or teams without an on-call cap.",
            issues,
        )

    return owner_of


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def _cross_team_dependencies(
    edges: list[tuple[str, str]], owner_of: dict[str, str]
) -> list[CrossTeamDependency]:
    """Stored edges crossing a team boundary; both endpoints must be owned."""
    cross: list[CrossTeamDependency] = []
    for service, depends_on in edges:
        if service not in owner_of or depends_on not in owner_of:
            continue
        service_team = owner_of[service]
        depends_on_team = owner_of[depends_on]
        if service_team == depends_on_team:
            continue
        cross.append(
            CrossTeamDependency(
                service=service,
                service_team=service_team,
                depends_on=depends_on,
                depends_on_team=depends_on_team,
            )
        )
    cross.sort(key=lambda item: (item.service, item.depends_on))
    return cross


def _coordination_order(
    cross: list[CrossTeamDependency],
) -> list[str]:
    """Teams in dependency-topological order (Kahn, lexicographic ties).

    The quotient edge runs downstream_team -> upstream_team: the upstream
    team's services must exist first, so prerequisite teams are emitted
    first. It is a DAG because it is a quotient of the stored DAG — a team
    cycle would imply a service cycle.
    """
    teams: set[str] = set()
    prerequisites: dict[str, set[str]] = {}
    dependents: dict[str, set[str]] = {}
    for item in cross:
        teams.add(item.service_team)
        teams.add(item.depends_on_team)
        prerequisites.setdefault(item.service_team, set()).add(
            item.depends_on_team
        )
        dependents.setdefault(item.depends_on_team, set()).add(
            item.service_team
        )

    indegree = {
        team: len(prerequisites.get(team, set())) for team in teams
    }
    ready = [team for team, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        team = heapq.heappop(ready)
        order.append(team)
        for downstream in sorted(dependents.get(team, set())):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                heapq.heappush(ready, downstream)
    return order


def _oncall_overlimits(
    team: str,
    owned: list[str],
    oncall_cap: int,
    wave_of: dict[str, int],
) -> list[TeamOncallOverlimit]:
    """Waves in which the team's simultaneous service count exceeds its cap."""
    by_wave: dict[int, list[str]] = {}
    for service in owned:
        by_wave.setdefault(wave_of[service], []).append(service)

    overlimit: list[TeamOncallOverlimit] = []
    for wave in sorted(by_wave):
        members = sorted(by_wave[wave])
        count = len(members)
        if count > oncall_cap:
            overlimit.append(
                TeamOncallOverlimit(
                    team=team,
                    wave=wave,
                    services=members,
                    service_count=count,
                    oncall_cap=oncall_cap,
                    over_by=count - oncall_cap,
                )
            )
    return overlimit


def _fault_domain_coupling(
    domains: list[dict], owner_of: dict[str, str]
) -> list[FaultDomainTeamCoupling]:
    """Coupling report for each stored fault domain, in stored order."""
    reports: list[FaultDomainTeamCoupling] = []
    for fd in domains:
        members = sorted(fd["services"])
        teams = sorted({owner_of[s] for s in members if s in owner_of})
        reports.append(
            FaultDomainTeamCoupling(
                domain=fd["domain"],
                teams=teams,
                services=members,
                coupled=len(teams) >= 2,
            )
        )
    return reports


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def audit_ownership(
    services: list[str],
    edges: list[tuple[str, str]],
    wave_members: list[list[str]],
    domains: list[dict],
    payload: OwnershipAuditRequest,
) -> OwnershipAuditOutput:
    """Audit team ownership against the stored graph, waves and fault domains.

    ``services`` lists every service of the stored plan, ``edges`` the
    dependency edges as ``(service, depends_on)``, ``wave_members`` the
    stored waves in order and ``domains`` the stored fault domains. Nothing
    is persisted by the caller.
    """
    owner_of = _validate_ownership(services, payload)

    unowned = sorted(set(services) - set(owner_of))

    owned_services: dict[str, list[str]] = {}
    for service, team in owner_of.items():
        owned_services.setdefault(team, []).append(service)

    wave_of = {
        service: index + 1
        for index, members in enumerate(wave_members)
        for service in members
    }

    team_summaries: list[OwnershipTeamSummary] = []
    for team in sorted(owned_services):
        owned = sorted(owned_services[team])
        cap = payload.team_oncall_caps[team]
        team_summaries.append(
            OwnershipTeamSummary(
                team=team,
                service_count=len(owned),
                services=owned,
                oncall_cap=cap,
                overlimit=_oncall_overlimits(team, owned, cap, wave_of),
            )
        )

    cross = _cross_team_dependencies(edges, owner_of)
    coordination_order = _coordination_order(cross)
    coupling = _fault_domain_coupling(domains, owner_of)

    return OwnershipAuditOutput(
        teams=team_summaries,
        unowned_services=unowned,
        cross_team_dependencies=cross,
        coordination_order=coordination_order,
        fault_domain_coupling=coupling,
    )
