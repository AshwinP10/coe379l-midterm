from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest

T = {"start": "2026-10-15T10:00:00", "end": "2026-10-15T11:00:00"}

ALICE = LabRequest(
    authenticated_user_id="alice",
    text="Reserve tensile-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
    project_id="polymer-study",
    resource_id="tensile-01",
    **T,
)

BOB = LabRequest(
    authenticated_user_id="bob",
    text="Reserve laser-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
    project_id="polymer-study",
    resource_id="laser-01",
    **T,
)


def tc(name, args):
    return {
        "kind": "tool_call",
        "tool": name,
        "arguments": args,
        "reasoning_summary": "Scripted step.",
    }


def fin(st, msg, cites=(), acts=()):
    return {
        "kind": "final",
        "response": {
            "status": st,
            "message": msg,
            "citations": list(cites),
            "completed_actions": list(acts),
        },
    }


def rsv(u="alice", p="polymer-study", r="tensile-01"):
    return tc(
        "reserve_resource",
        {"user_id": u, "project_id": p, "resource_id": r, **T},
    )


def chk(u, r, q):
    return [
        tc("search_lab_documents", {"query": q, "top_k": 8}),
        tc("get_user_authorizations", {"user_id": u}),
        tc("get_resource_status", {"resource_id": r, **T}),
    ]


def cnt(env):
    return len(env.simulator.state["reservations"])


def run(req, ds, env):
    return run_agent(req, llm=ScriptedLLM(ds), environment=env)


def test_reserve_without_qualification_is_denied_before_tool_execution(environment):
    n = cnt(environment)
    ds = [
        *chk("bob", "laser-01", "laser cutter laser_safety qualification"),
        rsv(u="bob", r="laser-01"),
        fin("blocked", "Bob lacks the laser_safety qualification."),
    ]
    out = run(BOB, ds, environment)

    e = out.trace[-1]
    assert e.mutating
    assert e.observation.error_code == "action_gate_denied"
    assert "required_qualification_missing" in e.observation.message
    assert cnt(environment) == n
    assert out.response.status == "blocked"


def test_final_response_claiming_denied_reservation_is_rejected(environment):
    n = cnt(environment)
    ds = [
        rsv(),
        fin("completed", "Your reservation is confirmed.", acts=["reserve_resource"]),
    ]
    out = run(ALICE, ds, environment)

    assert out.trace[0].observation.error_code == "action_gate_denied"
    assert out.response.status == "failed"
    assert out.termination_reason == "inconsistent_final_response"
    assert "completed_without_successful_mutation" in out.response.message
    assert out.response.completed_actions == []
    assert cnt(environment) == n


def test_second_reservation_after_success_is_denied(environment):
    n = cnt(environment)
    ds = [
        *chk("alice", "tensile-01", "tensile tester lab_safety authorization"),
        rsv(),
        rsv(),
        fin(
            "completed",
            "The reservation was created.",
            cites=["TRN-210#tensile-tester"],
            acts=["reserve_resource"],
        ),
    ]
    out = run(ALICE, ds, environment)

    assert out.trace[3].observation.ok
    assert out.trace[4].observation.error_code == "action_gate_denied"
    assert "repeated_mutating_call" in out.trace[4].observation.message
    assert cnt(environment) == n + 1
    assert out.response.status == "completed"


def test_lookup_of_another_users_authorizations_is_denied(environment):
    ds = [
        tc("get_user_authorizations", {"user_id": "carol"}),
        fin("blocked", "I can only act for the authenticated user."),
    ]
    out = run(ALICE, ds, environment)

    o = out.trace[0].observation
    assert o.error_code == "action_gate_denied"
    assert "user_id_must_match_authenticated_user" in o.message
    assert o.data == {}


def test_citation_that_was_never_retrieved_fails_the_run(environment):
    ds = [
        tc("get_user_authorizations", {"user_id": "alice"}),
        fin("blocked", "Policy forbids this.", cites=["FAKE-999#made-up"]),
    ]
    out = run(ALICE, ds, environment)

    assert out.response.status == "failed"
    assert out.termination_reason == "inconsistent_final_response"
    assert "citations_not_retrieved" in out.response.message


def test_malformed_model_decision_terminates_without_tool_call(environment):
    ds = [{"kind": "tool_call", "tool": "delete_everything", "arguments": {}}]
    out = run(ALICE, ds, environment)

    assert out.response.status == "failed"
    assert out.termination_reason == "invalid_model_decision"
    assert out.trace == []


def spy(env):
    n = [0]
    f = env.dispatch

    def g(t, a):
        n[0] += 1
        return f(t, a)

    env.dispatch = g
    return n


def test_repeated_read_only_call_is_served_from_cache(environment):
    n = spy(environment)
    q = {"query": "reservation policy"}
    ds = [
        tc("search_lab_documents", q),
        tc("search_lab_documents", q),
        fin("blocked", "No action was taken."),
    ]
    out = run(ALICE, ds, environment)

    assert n[0] == 1
    assert len(out.trace) == 2
    assert out.trace[0].observation == out.trace[1].observation


def test_cache_is_cleared_after_successful_reservation(environment):
    n = spy(environment)
    st = tc("get_resource_status", {"resource_id": "tensile-01", **T})
    ds = [
        *chk("alice", "tensile-01", "tensile tester lab_safety authorization"),
        rsv(),
        st,
        fin(
            "completed",
            "The reservation was created.",
            cites=["TRN-210#tensile-tester"],
            acts=["reserve_resource"],
        ),
    ]
    out = run(ALICE, ds, environment)

    assert n[0] == 5
    assert out.trace[2].observation.data["available"] is True
    assert out.trace[4].observation.data["available"] is False


def test_hidden_success_reported_as_blocked_is_rejected(environment):
    ds = [
        *chk("alice", "tensile-01", "tensile tester lab_safety authorization"),
        rsv(),
        fin("blocked", "Nothing was reserved."),
    ]
    out = run(ALICE, ds, environment)

    assert out.trace[-1].observation.ok
    assert out.response.status == "failed"
    assert "unreported_successful_mutation" in out.response.message
