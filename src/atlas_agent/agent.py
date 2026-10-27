from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _fail(msg: str, tr: list[TraceEvent], why: str) -> AgentRun:
    return AgentRun(
        response=AgentResponse(status="failed", message=msg),
        trace=tr,
        termination_reason=why,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    msgs = initialize_messages(request)
    tr: list[TraceEvent] = []
    memo: dict[str, ToolObservation] = {}

    for i in range(1, max_steps + 1):
        try:
            d = llm.generate(messages=msgs, response_model=DecisionAdapter)
        except (ValidationError, ValueError, TypeError, RuntimeError) as ex:
            return _fail(
                f"The model did not produce a valid decision: {ex}",
                tr,
                "invalid_model_decision",
            )

        if isinstance(d, ClarificationDecision):
            return AgentRun(
                response=AgentResponse(
                    status="clarification_required", message=d.question
                ),
                trace=tr,
                termination_reason="clarification_required",
            )

        if isinstance(d, FinalDecision):
            errs = validate_final_response(d.response, tr)
            if errs:
                return _fail(
                    "The proposed final response is inconsistent with the trace: "
                    + "; ".join(errs),
                    tr,
                    "inconsistent_final_response",
                )
            return AgentRun(
                response=d.response,
                trace=tr,
                termination_reason="model_finished",
            )

        if not isinstance(d, ToolCallDecision):
            return _fail("Unsupported decision type.", tr, "invalid_model_decision")

        mut = d.tool in MUTATING_TOOLS
        errs = validate_proposed_tool_call(d, request=request, trace=tr)
        if mut and any(e.mutating and e.observation.ok for e in tr):
            errs.append("repeated_mutating_call")

        if errs:
            obs = ToolObservation(
                ok=False,
                tool=d.tool,
                error_code="action_gate_denied",
                message="; ".join(errs),
            )
        elif mut:
            obs = environment.dispatch(d.tool, d.arguments)
            if obs.ok:
                memo.clear()
        else:
            k = d.tool + json.dumps(d.arguments, sort_keys=True, default=str)
            obs = memo.get(k)
            if obs is None:
                obs = environment.dispatch(d.tool, d.arguments)
                if obs.ok:
                    memo[k] = obs

        tr.append(TraceEvent(step=i, decision=d.model_dump(), observation=obs, mutating=mut))

        msgs.append({"role": "assistant", "content": d.model_dump_json()})
        msgs.append({"role": "tool", "content": obs.model_dump_json()})

    return _fail(
        f"Stopped after {max_steps} steps without a final response.",
        tr,
        "maximum_steps_exceeded",
    )
