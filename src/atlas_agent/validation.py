from __future__ import annotations

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context
from .tools import MUTATING_TOOLS

FLDS = ("project_id", "resource_id", "start", "end")

PRE = {
    "arguments_complete": "reservation_arguments_incomplete",
    "policy_retrieved": "policy_not_retrieved",
    "authorizations_checked": "authorizations_not_checked",
    "resource_status_checked": "resource_status_not_checked",
    "project_authorized": "project_not_authorized",
    "qualification_current": "required_qualification_missing",
    "resource_available": "resource_not_available",
    "calibration_current": "calibration_not_current",
    "maintenance_clear": "resource_under_maintenance",
    "after_hours_approval_satisfied": "after_hours_approval_missing",
}


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    errs: list[str] = []
    a = decision.arguments

    if "user_id" in a and a["user_id"] != request.authenticated_user_id:
        errs.append("user_id_must_match_authenticated_user")

    for f in FLDS:
        v = getattr(request, f)
        if v is None:
            if decision.tool in MUTATING_TOOLS:
                errs.append(f"{f}_unresolved_in_request")
        elif f in a and a[f] != v:
            errs.append(f"{f}_does_not_match_request")
    return errs


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    return [e for k, e in PRE.items() if not getattr(context, k)]


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    errs: list[str] = []

    bad = sorted(set(response.citations) - retrieved_passage_ids(trace))
    if bad:
        errs.append(f"citations_not_retrieved: {', '.join(bad)}")

    muts = [e.observation.tool for e in trace if e.mutating and e.observation.ok]
    if any(x not in muts for x in response.completed_actions):
        errs.append("completed_action_not_in_successful_trace")

    if response.status == "completed":
        if not muts:
            errs.append("completed_without_successful_mutation")
        if not response.completed_actions:
            errs.append("completed_without_claimed_action")
    else:
        if response.completed_actions:
            errs.append("non_completed_response_claims_actions")
        if muts:
            errs.append("unreported_successful_mutation")
    return errs
