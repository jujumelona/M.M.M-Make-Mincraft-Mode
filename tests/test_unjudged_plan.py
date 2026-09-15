import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import complete_planner, model_router
from minecraft_mod_ai.api import CompleteModAISession
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.colab_run_modes import run_plan_dialog
from minecraft_mod_ai.mcp_tools import MMMToolService

RESPONSES = (
    "우주선을 부위별로 제작하고 무기, 선원, 성능을 거래로 업그레이드합니다.",
    '```json\n[{"capability_label":"space-fleet","acceptance_conditions":["Trade minerals"]}]\n```',
    "0",
    "# 자유 설계\n\n행성마다 다른 광물을 사고팔고 연료와 선원을 선택한다.",
)


def router_for(monkeypatch, output):
    router = model_router.ModelRouter(profile="fast_test")
    requests = []

    def generate(request):
        requests.append(request)
        return output

    monkeypatch.setattr(
        router,
        "_generation_adapter",
        lambda role: (
            SimpleNamespace(adapter="llama_cpp"),
            SimpleNamespace(generate=generate),
        ),
    )
    monkeypatch.setattr(router, "_generation_scope", lambda config: nullcontext())
    monkeypatch.setattr(
        model_router,
        "validate_structured_output",
        lambda *a, **k: pytest.fail("plan validation called"),
    )
    monkeypatch.setattr(
        complete_planner,
        "PlanningPipeline",
        lambda *a, **k: pytest.fail("production pipeline called while planning"),
    )
    return router, requests


def session_for(tmp_path, router):
    session = object.__new__(CompleteModAISession)
    session.router = router
    session.planner = complete_planner.CompleteGameDesignPlanner(router)
    session.output_root = tmp_path
    session.workspace_root = tmp_path
    session.existing_input = None
    session.brief = ""
    session.complete_proposal = None
    return session


@pytest.mark.parametrize("output", RESPONSES)
def test_plan_accepts_actual_log_response_shapes_without_parsing(
    monkeypatch, tmp_path, output
):
    router, requests = router_for(monkeypatch, output)
    session = session_for(tmp_path, router)
    reply = session.plan("우주 모드")
    assert reply.message == output
    assert reply.complete_proposal.text == output
    assert len(requests) == 1
    assert requests[0].response_format == "text"
    assert requests[0].response_schema is None
    assert not requests[0].tools
    loaded = session.load_plan()
    assert loaded.message == output
    assert loaded.approval_hash == reply.approval_hash


def test_colab_plan_saves_text_and_finishes_without_production(monkeypatch, tmp_path):
    router, requests = router_for(monkeypatch, RESPONSES[0])
    session = session_for(tmp_path, router)
    target = tmp_path / "colab-plan.json"
    result = run_plan_dialog(
        session=session,
        run_mode="Plan",
        prompt="우주 모드",
        plan_path=target,
        print_fn=lambda *a: None,
    )
    assert result.reply.message == RESPONSES[0]
    assert json.loads(target.read_text(encoding="utf-8"))["text"] == RESPONSES[0]
    assert len(requests) == 1


def test_build_compilation_cannot_destroy_saved_design(monkeypatch, tmp_path):
    router, _ = router_for(monkeypatch, RESPONSES[3])
    session = session_for(tmp_path, router)
    reply = session.plan("우주 모드")
    before = (tmp_path / "proposal.json").read_bytes()

    def compile_design(prompt, **kwargs):
        assert RESPONSES[3] in prompt
        raise RuntimeError("production compiler unavailable")

    monkeypatch.setattr(session.planner, "compile_for_production", compile_design)
    with pytest.raises(RuntimeError, match="production compiler"):
        session.build(reply)
    assert session.complete_proposal is reply.complete_proposal
    assert (tmp_path / "proposal.json").read_bytes() == before


def test_mcp_stores_and_reads_the_same_unjudged_plan(monkeypatch, tmp_path):
    router, requests = router_for(monkeypatch, RESPONSES[3])
    tools = MMMToolService(workspace_root=tmp_path, router_factory=lambda: router)
    result = tools.plan_complete_game("우주 모드")
    assert result["message"] == RESPONSES[3]
    plan, _ = tools._resolve_complete_proposal(
        complete_proposal=None, proposal_ref=result["proposal_ref"]
    )
    assert isinstance(plan, AuthoredPlan)
    assert plan.text == RESPONSES[3]
    page = tools.read_complete_plan_section(result["proposal_ref"], limit=100)
    assert page["text"] == RESPONSES[3]
    assert tools.read_quality_contract(result["proposal_ref"])["schema_version"] == "mmm/authored-plan-production-v1"
    assert len(requests) == 1


def test_cli_plan_prints_and_saves_raw_model_text(monkeypatch, tmp_path, capsys):
    from minecraft_mod_ai import cli

    router, requests = router_for(monkeypatch, RESPONSES[3])
    monkeypatch.setattr(cli, "ModelRouter", lambda **kwargs: router)
    target = tmp_path / "cli-plan.json"
    assert cli.main(["plan", "우주 모드", "--save", str(target)]) == 0
    assert RESPONSES[3] in capsys.readouterr().out
    assert json.loads(target.read_text(encoding="utf-8"))["text"] == RESPONSES[3]
    assert len(requests) == 1


def test_build_receives_the_authored_design_only_after_explicit_build(
    monkeypatch, tmp_path
):
    router, _ = router_for(monkeypatch, RESPONSES[3])
    session = session_for(tmp_path, router)
    reply = session.plan("우주 모드")
    compiled = SimpleNamespace(calculate_hash=lambda: "compiled-hash")
    calls = []

    def compile_design(prompt, **kwargs):
        calls.append(prompt)
        return compiled

    monkeypatch.setattr(session.planner, "compile_for_production", compile_design)

    def execute(proposal, **kwargs):
        assert proposal is compiled
        return "production-result"

    session.orchestrator = SimpleNamespace(execute=execute)
    assert session.build(reply) == "production-result"
    assert calls == [reply.complete_proposal.production_prompt()]
    assert session.load_plan().message == RESPONSES[3]
