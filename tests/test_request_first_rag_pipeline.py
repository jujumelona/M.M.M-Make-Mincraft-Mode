from __future__ import annotations

import json

from minecraft_mod_ai.authored_content_contract import content_request_catalog
from minecraft_mod_ai.authored_plan import AuthoredPlan

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
    assert len(received) == 1 + len(FEATURES)
    assert all("ACTIVE REQUEST FEATURE:" in prompt for prompt in received[1:])
    assert "Establish colonies" in received[-1]


def test_request_decomposition_retains_more_than_four_features(monkeypatch, tmp_path):
    from minecraft_mod_ai import request_scope

    monkeypatch.setenv("MMM_REQUEST_RAG_DIR", str(tmp_path))

    class Router:
        def __init__(self):
            self.messages = []
            self.ordinal = 0

        def generate_text(self, role, messages, **kwargs):
            self.messages.append(messages)
            assert "feature_count" not in kwargs["response_schema"]["properties"]
            if self.ordinal == len(FEATURES):
                return json.dumps({"status": "complete", "statement": "", "search_query": "",
                                   "source_quote": "", "implementation_role": "gameplay"})
            statement, query, quote = FEATURES[self.ordinal]
            self.ordinal += 1
            return json.dumps({"status": "feature", "statement": statement, "search_query": query, "source_quote": quote,
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
            self.ordinal = 0
            self.inputs = []

        def generate_text(self, role, messages, **kwargs):
            self.inputs.append(str(messages))
            if self.ordinal == 33:
                return json.dumps({"status": "complete", "statement": "", "search_query": "",
                                   "source_quote": "", "implementation_role": "gameplay"})
            token = f"f{self.ordinal:02d}"
            self.ordinal += 1
            return json.dumps({"status": "feature", "statement": "Implement " + token, "search_query": token,
                               "source_quote": token, "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, prompt)
    assert router.ordinal == 33
    assert len(scope["requirements"]) == 33
    assert len({row["requirement_id"] for row in scope["requirements"]}) == 33
    assert [row["source_span"]["text"] for row in scope["requirements"]] == prompt.split()
    assert max(map(len, router.inputs)) < 5000


def test_repeating_continuation_has_bounded_recovery_without_duplicate_requirements(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)

    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            assert self.calls < 10, "Repeated output must not cause an endless continuation"
            return json.dumps({"status": "feature", "statement": "Mine ore", "search_query": "ore mining",
                               "source_quote": "mine ore", "implementation_role": "gameplay"})

    router = Router()
    prompt = "mine ore, travel, build, trade"
    scope = request_scope.author_request_scope(router, prompt)
    assert router.calls <= 4
    assert scope["requested_prompt"] == prompt
    assert sum(row["statement"] == "Mine ore" for row in scope["requirements"]) == 1
    assert any(row.get("source_recovery") for row in scope["requirements"])
    ledger = " ".join(row["statement"] for row in scope["requirements"])
    assert all(word in ledger for word in ("travel", "build", "trade"))
    assert all(prompt[row["source_span"]["start"]:row["source_span"]["end"]] == row["source_span"]["text"]
               for row in scope["requirements"])


def test_duplicate_recovery_preserves_distinct_features_sharing_one_clause(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)

    class Router:
        rows = 0

        def generate_text(self, role, messages, **kwargs):
            if self.rows == 3:
                return json.dumps({"status": "complete", "statement": "", "search_query": "",
                                   "source_quote": "", "implementation_role": "gameplay"})
            self.rows += 1
            statement = "Mine ore" if self.rows < 3 else "Trade ore"
            return json.dumps({"status": "feature", "statement": statement, "search_query": statement,
                               "source_quote": "mine and trade ore", "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, "mine and trade ore")
    assert router.rows == 3
    assert [row["statement"] for row in scope["requirements"]] == ["Mine ore", "Trade ore"]
    assert len({row["requirement_id"] for row in scope["requirements"]}) == 2


def test_repeated_last_feature_finishes_without_a_plan_error(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)

    class Router:
        calls = 0

        def generate_text(self, role, messages, **kwargs):
            self.calls += 1
            assert "feature_count" not in kwargs["response_schema"]["properties"]
            return json.dumps({"status": "feature", "statement": "Mine ore",
                               "search_query": "ore mining", "source_quote": "mine ore",
                               "implementation_role": "gameplay"})

    router = Router()
    scope = request_scope.author_request_scope(router, "mine ore")
    assert router.calls == 4
    assert len(scope["requirements"]) == 1
    assert not any(row.get("source_recovery") for row in scope["requirements"])


def test_failed_feature_extraction_preserves_original_source_and_continues(monkeypatch):
    from minecraft_mod_ai import request_scope
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)
    prompts = []

    def reject(_router, _role, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        raise StructuredOutputValidationError(output="{}", errors=("fixture schema failure",))

    monkeypatch.setattr(request_scope, "generate_fixed_template_value", reject)
    prompt = "mine, trade, explore\n" * 90
    scope = request_scope.author_request_scope(object(), prompt)
    assert len(prompts) > 1
    assert scope["requested_prompt"] == prompt
    assert all(row.get("source_recovery") for row in scope["requirements"])
    spans = [row["source_span"] for row in scope["requirements"]]
    assert all(prompt[span["start"]:span["end"]] == span["text"] for span in spans)
    assert len({row["requirement_id"] for row in scope["requirements"]}) == len(spans)
    assert "".join("".join(span["text"].split()) for span in spans) == "".join(prompt.split())


def test_constraints_only_do_not_force_a_gameplay_feature(monkeypatch):
    from minecraft_mod_ai import request_scope

    monkeypatch.setattr(request_scope, "_template_rag", lambda: None)
    router = type("Router", (), {"generate_text": lambda self, *args, **kwargs: json.dumps({
        "status": "complete", "statement": "", "search_query": "", "source_quote": "",
        "implementation_role": "infrastructure",
    })})()
    scope = request_scope.author_request_scope(router, "Fabric only; keep the original package name")
    assert scope["requirements"] == []
