from __future__ import annotations

import json

from minecraft_mod_ai.authored_content_contract import content_request_catalog
from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_reuse_bridge import authored_capability_graph

PROMPT = "자원파밍 거래 우주선 조립 무기 선원 업그레이드 행성 여행 특수 광물 외계인 전투 식민지화 모드"
FEATURES = [
    ("Mine resources", "resource mining", "자원파밍"),
    ("Trade resources for currency", "economy trading", "거래"),
    ("Assemble ship parts", "spaceship construction", "우주선 조립"),
    ("Upgrade weapons and recruit crew", "equipment upgrades crew", "무기 선원 업그레이드"),
    ("Travel to other planets", "space travel", "행성 여행"),
    ("Collect special planetary minerals", "planet ore generation", "특수 광물"),
    ("Fight aliens", "alien combat", "외계인 전투"),
    ("Establish colonies", "colony settlement", "식민지화"),
]


def scope_fixture():
    return {
        "schema_version": "mmm/request-scope-v1",
        "requested_prompt": PROMPT,
        "requirements": [
            {"requirement_id": f"request_{i}", "statement": statement,
             "search_query": query, "source_span": {"text": quote},
             "implementation_role": "gameplay",
             "rag_references": []}
            for i, (statement, query, quote) in enumerate(FEATURES)
        ],
    }


def test_plan_round_trip_preserves_original_features():
    scope = scope_fixture()
    plan = AuthoredPlan(PROMPT, "Shipyard worksheet", request_scope=scope)
    saved = plan.to_dict()
    restored = AuthoredPlan.from_dict(saved)
    assert restored.request_scope == scope
    assert restored.calculate_hash() == plan.calculate_hash()
    assert "Establish colonies" in restored.production_prompt()
    saved["request_scope"]["requirements"].clear()
    assert len(plan.request_scope["requirements"]) == 8


def test_legacy_plan_stays_readable_without_scope():
    original = AuthoredPlan(PROMPT, "Legacy worksheet")
    restored = AuthoredPlan.from_dict(original.to_dict())
    assert restored.request_scope == {}
    assert restored.to_dict() == original.to_dict()


def test_content_discovery_preserves_features_missing_from_worksheet():
    catalog = content_request_catalog({}, requested_prompt=PROMPT,
                                      request_scope=scope_fixture())
    rows = catalog["requirements"]
    assert [r["statement"] for r in rows] == [r[0] for r in FEATURES]
    assert len({r["requirement_id"] for r in rows}) == 8
    assert rows[-1]["design_context"]["source_feature"]["source_span"]["text"] == "식민지화"


def test_scope_keeps_authored_transactions_as_implementation_work(monkeypatch):
    from minecraft_mod_ai import authored_content_contract as contract

    def concerns(sections, section):
        return {"atomic_mutations": [{"mutations": "spend ore and grant a hull"}]} if section == "algorithm" else {}

    monkeypatch.setattr(contract, "active_concern_records", concerns)
    catalog = contract.content_request_catalog({}, requested_prompt=PROMPT, request_scope=scope_fixture())
    rows = catalog["requirements"]
    assert len(rows) == 9
    assert rows[-1]["design_context"]["source_concern"] == "atomic_mutations"
    assert "spend ore" in rows[-1]["statement"]


def test_source_search_uses_feature_queries_without_whole_korean_request():
    graph = authored_capability_graph(PROMPT, {}, request_scope=scope_fixture())
    assert graph["nodes"] == [r[1] for r in FEATURES]
    assert {row["capability"] for row in graph["search_terms"]} == set(graph["nodes"])
    assert all(PROMPT not in term for row in graph["search_terms"] for term in row["terms"])


def test_planner_checkpoint_restores_scope_without_another_model_call(monkeypatch, tmp_path):
    from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner

    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    received = []

    def decompose(router, prompt, **kwargs):
        received.append("scope")
        return scope_fixture()

    def worksheet(router, prompt, **kwargs):
        received.append(prompt)
        return {}

    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope", decompose)
    monkeypatch.setattr("minecraft_mod_ai.authored_structured_design.author_structured_sections", worksheet)
    planner = CompleteGameDesignPlanner(object())
    first = planner.plan(PROMPT)
    second = planner.plan(PROMPT)
    assert first.request_scope == second.request_scope == scope_fixture()
    assert len(received) == 2
    assert "Establish colonies" in received[1]


def test_request_decomposition_retains_more_than_four_features(monkeypatch, tmp_path):
    from minecraft_mod_ai import request_scope

    monkeypatch.setenv("MMM_REQUEST_RAG_DIR", str(tmp_path))

    class Router:
        def __init__(self):
            self.messages = []
            self.ordinal = 0

        def generate_text(self, role, messages, **kwargs):
            self.messages.append(messages)
            fields = kwargs["response_schema"]["properties"]
            if "feature_count" in fields:
                return json.dumps({"feature_count": "8"})
            statement, query, quote = FEATURES[self.ordinal]
            self.ordinal += 1
            return json.dumps({"statement": statement, "search_query": query, "source_quote": quote,
                               "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, PROMPT)
    assert len(scope["requirements"]) == 8
    assert [r["statement"] for r in scope["requirements"]] == [r[0] for r in FEATURES]
    assert scope["requested_prompt"] == PROMPT
    assert all(r["source_span"]["text"] in PROMPT for r in scope["requirements"])
    assert any("RAG" in str(messages) for messages in router.messages)
    assert scope["requirements"][4]["rag_references"]


def test_dense_request_continues_after_a_full_feature_page(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)
    prompt = " ".join(f"f{i:02d}" for i in range(33))

    class Router:
        def __init__(self):
            self.counts = 0
            self.ordinal = 0
            self.inputs = []

        def generate_text(self, role, messages, **kwargs):
            self.inputs.append(str(messages))
            fields = kwargs["response_schema"]["properties"]
            if "feature_count" in fields:
                self.counts += 1
                assert "32+" in fields["feature_count"]["enum"]
                return json.dumps({"feature_count": "32+" if self.counts == 1 else "1"})
            token = f"f{self.ordinal:02d}"
            self.ordinal += 1
            return json.dumps({"statement": "Implement " + token, "search_query": token,
                               "source_quote": token, "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, prompt)
    assert router.counts == 2
    assert len(scope["requirements"]) == 33
    assert len({row["requirement_id"] for row in scope["requirements"]}) == 33
    assert [row["source_span"]["text"] for row in scope["requirements"]] == prompt.split()
    assert max(map(len, router.inputs)) < 5000


def test_repeating_continuation_has_bounded_recovery_without_duplicate_requirements(monkeypatch):
    import pytest

    from minecraft_mod_ai import request_scope
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)

    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            assert self.calls < 10, "Repeated output must not cause an endless continuation"
            if "feature_count" in kwargs["response_schema"]["properties"]:
                return json.dumps({"feature_count": "32+"})
            return json.dumps({"statement": "Mine ore", "search_query": "ore mining",
                               "source_quote": "mine ore", "implementation_role": "gameplay"})

    router = Router()
    with pytest.raises(StructuredOutputValidationError, match="REQUEST_SCOPE_NO_PROGRESS"):
        request_scope.author_request_scope(router, "mine ore, travel, build, trade")
    assert router.calls <= 6


def test_duplicate_recovery_preserves_distinct_features_sharing_one_clause(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)

    class Router:
        rows = 0

        def generate_text(self, role, messages, **kwargs):
            if "feature_count" in kwargs["response_schema"]["properties"]:
                return json.dumps({"feature_count": "2"})
            self.rows += 1
            statement = "Mine ore" if self.rows < 3 else "Trade ore"
            return json.dumps({"statement": statement, "search_query": statement,
                               "source_quote": "mine and trade ore", "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, "mine and trade ore")
    assert router.rows == 3
    assert [row["statement"] for row in scope["requirements"]] == ["Mine ore", "Trade ore"]
    assert len({row["requirement_id"] for row in scope["requirements"]}) == 2
