"""Exercise the real CLI plan entrypoint, saved artifact and source evidence reuse."""
from __future__ import annotations

import json

from minecraft_mod_ai import cli, complete_planner
from minecraft_mod_ai import authored_structured_design, request_scope
from minecraft_mod_ai import model_router, research_reuse_candidates
from minecraft_mod_ai.authored_plan import AuthoredPlan


def test_cli_plan_persists_host_catalogue_receipt_and_reuses_it(monkeypatch, tmp_path):
    prompt = "Build a spaceship that can travel to other planets"
    discovered = [{
        "repository": "example/space-donor",
        "origin": "host_grounded_retrieval",
        "reference_only": True,
        "source_reuse_authority": "verification_required",
        "source_ids": ["curseforge:42"],
        "source_urls": ["https://github.com/example/space-donor"],
        "catalog_evidence": [{
            "provider": "curseforge",
            "source_id": "curseforge:42",
            "summary": "Spacecraft and interplanetary travel",
        }],
    }]
    calls = []

    def fake_collect(_router, value):
        calls.append(value)
        return {
            "schema_version": "mmm/pre-design-source-candidate-handoff-v1",
            "prompt_sha256": _canonical_sha256(value),
            "repository_candidates": discovered,
            "candidate_count": 1,
            "model_called": False,
            "status": "available",
        }

    from minecraft_mod_ai.pre_design_research_pipeline import _canonical_sha256
    from minecraft_mod_ai import pre_design_research_pipeline
    monkeypatch.setattr(pre_design_research_pipeline, "collect_source_candidate_handoff", fake_collect)

    class Checkpoint:
        def __init__(self, **_kwargs):
            pass

        def load_request_scope(self):
            return {"requirements": []}

        def load_sections(self):
            return {"sections": "canonical"}

        def save_request_scope(self, *_args):
            raise AssertionError("request scope was already available")

        def save_sections(self, *_args):
            raise AssertionError("structured sections were already available")

    monkeypatch.setattr(complete_planner, "PlannerCheckpoint", Checkpoint)
    monkeypatch.setattr(complete_planner, "repository_revision", lambda: {"git_commit": "test"})
    monkeypatch.setattr(authored_structured_design, "normalize_structured_sections", lambda row: row)
    monkeypatch.setattr(authored_structured_design, "render_structured_sections", lambda _: "Design")
    monkeypatch.setattr(request_scope, "request_scope_context", lambda _: "Original scope")
    monkeypatch.setattr(model_router, "ModelRouter", lambda **_kwargs: object())

    saved = tmp_path / "authored-plan.json"
    assert cli.main(["plan", prompt, "--save", str(saved), "--json"]) == 0
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["schema_version"] == "mmm/authored-plan-v2"
    handoff = document["content_design"]["_pre_design_source_handoff"]
    assert handoff["candidate_count"] == 1
    assert handoff["repository_candidates"][0]["repository"] == "example/space-donor"
    assert calls == [prompt]

    restored = AuthoredPlan.from_dict(document)
    payload, reuse_candidates = research_reuse_candidates.ensure_authored_source_handoff(
        restored.content_design, router=object(), prompt=prompt,
    )
    assert payload == restored.content_design
    assert len(reuse_candidates) == 1
    assert calls == [prompt]  # The execute-side boundary does not re-search.


def test_cli_plan_preserves_provider_failure_without_making_fake_donor(monkeypatch, tmp_path):
    from minecraft_mod_ai.pre_design_research_pipeline import _canonical_sha256
    from minecraft_mod_ai import pre_design_research_pipeline

    monkeypatch.setattr(
        pre_design_research_pipeline, "collect_source_candidate_handoff",
        lambda _router, _prompt: (_ for _ in ()).throw(OSError("offline")),
    )
    class Checkpoint:
        def __init__(self, **_kwargs):
            pass
        def load_request_scope(self):
            return {"requirements": []}
        def load_sections(self):
            return {"plan": "valid"}

    monkeypatch.setattr(complete_planner, "PlannerCheckpoint", Checkpoint)
    monkeypatch.setattr(complete_planner, "repository_revision", lambda: {"git_commit": "test"})
    monkeypatch.setattr(authored_structured_design, "normalize_structured_sections", lambda row: row)
    monkeypatch.setattr(authored_structured_design, "render_structured_sections", lambda _: "Design")
    monkeypatch.setattr(request_scope, "request_scope_context", lambda _: "Scope")
    monkeypatch.setattr(model_router, "ModelRouter", lambda **_kwargs: object())

    output = tmp_path / "offline.json"
    assert cli.main(["plan", "a new machine", "--save", str(output), "--json"]) == 0
    restored = AuthoredPlan.from_dict(json.loads(output.read_text(encoding="utf-8")))
    report = restored.content_design["_pre_design_source_handoff"]
    assert report["status"] == "provider_error"
    assert report["repository_candidates"] == []
    assert report["prompt_sha256"] == _canonical_sha256("a new machine")
