from __future__ import annotations

import pytest

from minecraft_mod_ai.canonical_capability_ontology import atomic_capability_definitions
from minecraft_mod_ai.minecraft_template_catalog import semantic_capability_choices
from minecraft_mod_ai.request_requirements import (
    LeafAtomicityStatus,
    decompose_compound_leaf_host,
    detected_capability_clusters,
    filter_and_split_context,
    is_genre_context,
    is_pure_catch_all,
    validate_leaf_atomicity,
)
from minecraft_mod_ai.semantic_requirement_authority import _ground_source_anchor
from minecraft_mod_ai.semantic_source_fidelity import validate_semantic_source_partition


FULL_PROMPT = (
    "우주모드 인데 자원파밍 돈모으기 거래 등으로 우주선을 부위마다 만들어서 만들수있고 "
    "무기 선원 우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 있고 그렇게해서 "
    "우주로 나갈수있고 우주로 나가면 다른행성의 특수 광물 외게인과 싸움 식민지화등 여러가지가 가능한 모드"
)


def _clause(text: str = FULL_PROMPT, index: int = 0) -> dict[str, object]:
    return {
        "clause_index": index,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "text_sha256": "sha256:test",
    }


def test_validate_leaf_atomicity_rejects_compound_alien_and_colony() -> None:
    compound_leaf = {
        "source_clause_index": 0,
        "source_anchor": "외게인과 싸움 식민지화",
        "semantic_statement": "외게인과 싸움 및 행성 식민지화",
        "given": "외계인과 행성이 존재한다",
        "when": "외게인과 싸우고 행성을 식민지화한다",
        "then": "전투가 진행되고 식민지가 건설된다",
    }
    status, reason = validate_leaf_atomicity(compound_leaf)
    assert status == LeafAtomicityStatus.COMPOUND
    assert "alien_combat" in reason
    assert "colonization" in reason


def test_validate_leaf_atomicity_rejects_compound_weapons_crew_performance() -> None:
    compound_leaf = {
        "source_clause_index": 0,
        "source_anchor": "무기 선원 우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 있고",
        "semantic_statement": "무기, 선원, 우주선 성능 업그레이드 및 확장",
        "given": "우주선이 준비되어 있다",
        "when": "무기 선원 성능을 업그레이드한다",
        "then": "우주선의 모든 스펙이 강화된다",
    }
    status, reason = validate_leaf_atomicity(compound_leaf)
    assert status == LeafAtomicityStatus.COMPOUND
    assert "weapon_upgrade" in reason
    assert "crew_management" in reason
    assert "spaceship_performance" in reason


def test_validate_leaf_atomicity_rejects_compound_resource_money_trade() -> None:
    compound_leaf = {
        "source_clause_index": 0,
        "source_anchor": "자원파밍 돈모으기 거래 등으로",
        "semantic_statement": "자원파밍과 돈모으기 및 거래",
        "given": "자원이 존재한다",
        "when": "자원을 파밍하고 돈을 모아 거래한다",
        "then": "자원과 화폐가 증가하고 거래가 완료된다",
    }
    status, reason = validate_leaf_atomicity(compound_leaf)
    assert status == LeafAtomicityStatus.COMPOUND
    assert "resource_gathering" in reason
    assert "currency_economy" in reason
    assert "trading" in reason


def test_validate_leaf_atomicity_detects_context_and_catch_all() -> None:
    context_leaf = {
        "source_clause_index": 0,
        "source_anchor": "우주모드 인데",
        "semantic_statement": "우주모드 인데",
        "given": "마인크래프트 게임",
        "when": "모드를 로드한다",
        "then": "우주 테마가 적용된다",
    }
    status, reason = validate_leaf_atomicity(context_leaf)
    assert status == LeafAtomicityStatus.CONTEXT
    assert "genre/theme context" in reason

    catch_all_leaf = {
        "source_clause_index": 0,
        "source_anchor": "등 여러가지가 가능한 모드",
        "semantic_statement": "등 여러가지가 가능한 모드",
        "given": "우주에 도달함",
        "when": "기타 활동을 수행함",
        "then": "다양한 동작이 가능하다",
    }
    status, reason = validate_leaf_atomicity(catch_all_leaf)
    assert status == LeafAtomicityStatus.CATCH_ALL
    assert "catch-all" in reason


def test_filter_and_split_context_separates_all_four_categories() -> None:
    leaves = [
        {
            "source_clause_index": 0,
            "source_anchor": "우주모드 인데",
            "semantic_statement": "우주모드 인데",
            "given": "game",
            "when": "load",
            "then": "space theme",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "자원파밍",
            "semantic_statement": "자원 채굴 및 파밍",
            "given": "자원 존재",
            "when": "자원 채굴",
            "then": "인벤토리 획득",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "외게인과 싸움 식민지화",
            "semantic_statement": "외게인과 싸움 식민지화",
            "given": "외계인과 행성 존재",
            "when": "외게인과 싸우고 식민지화",
            "then": "전투 및 식민지화",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "등 여러가지가 가능한 모드",
            "semantic_statement": "등 여러가지가 가능한 모드",
            "given": "행성",
            "when": "기타활동",
            "then": "동작수행",
        },
    ]

    atomic, compound, context, catch_alls = filter_and_split_context(leaves)
    assert len(atomic) == 1
    assert atomic[0]["source_anchor"] == "자원파밍"
    assert len(compound) == 1
    assert compound[0]["source_anchor"] == "외게인과 싸움 식민지화"
    assert len(context) == 1
    assert context[0]["source_anchor"] == "우주모드 인데"
    assert len(catch_alls) == 1
    assert catch_alls[0]["source_anchor"] == "등 여러가지가 가능한 모드"


def test_full_prompt_decomposition_yields_11_atomic_leaves_and_context() -> None:
    clause = _clause(FULL_PROMPT)

    # Initial coarse decomposition representing typical LLM bundling
    coarse_leaves = [
        {
            "source_clause_index": 0,
            "source_anchor": "우주모드 인데",
            "semantic_statement": "우주모드 인데",
            "given": "game",
            "when": "load",
            "then": "space mode",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "자원파밍 돈모으기 거래 등으로 우주선을 부위마다 만들어서 만들수있고",
            "semantic_statement": "자원파밍 돈모으기 거래 등으로 우주선을 부위마다 만들기",
            "given": "materials exist",
            "when": "farm, earn, trade and craft ship",
            "then": "ship is created",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "무기 선원 우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 있고",
            "semantic_statement": "무기 선원 우주선 성능 업그레이드 확장",
            "given": "ship exists",
            "when": "upgrade weapons crew performance",
            "then": "ship is upgraded",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "그렇게해서 우주로 나갈수있고",
            "semantic_statement": "우주로 나가기",
            "given": "spaceship ready",
            "when": "launch to space",
            "then": "spaceship enters space",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "우주로 나가면 다른행성의 특수 광물",
            "semantic_statement": "다른 행성의 특수 광물 획득",
            "given": "planet exists",
            "when": "mine special minerals",
            "then": "minerals acquired",
        },
        {
            "source_clause_index": 0,
            "source_anchor": "외게인과 싸움 식민지화등 여러가지가 가능한 모드",
            "semantic_statement": "외게인과 싸움 식민지화등 여러가지가 가능한 모드",
            "given": "planet reached",
            "when": "fight aliens and colonize and other things",
            "then": "combat and colonization occur",
        },
    ]

    final_atomic_leaves: list[dict[str, object]] = []
    context: list[dict[str, object]] = []
    catch_alls: list[dict[str, object]] = []

    for c_leaf in coarse_leaves:
        status, _ = validate_leaf_atomicity(c_leaf)
        if status == LeafAtomicityStatus.CONTEXT:
            context.append(c_leaf)
        elif status == LeafAtomicityStatus.CATCH_ALL:
            catch_alls.append(c_leaf)
        elif status == LeafAtomicityStatus.COMPOUND:
            sub_leaves = decompose_compound_leaf_host(c_leaf, clause)
            sub_atomic, _, sub_ctx, sub_dropped = filter_and_split_context(sub_leaves)
            final_atomic_leaves.extend(sub_atomic)
            context.extend(sub_ctx)
            catch_alls.extend(sub_dropped)
        else:
            final_atomic_leaves.append(c_leaf)

    # Exactly 11 atomic leaves
    assert len(final_atomic_leaves) == 11, f"Expected 11 atomic leaves, got {len(final_atomic_leaves)}"

    # Check every leaf is atomic
    for leaf in final_atomic_leaves:
        status, _ = validate_leaf_atomicity(leaf)
        assert status == LeafAtomicityStatus.ATOMIC

    # Check catalog capabilities
    valid_capabilities = set(semantic_capability_choices())
    expected_capabilities = [
        "resource.farming",
        "economy.currency",
        "economy.trade",
        "spacecraft.component_construction",
        "spacecraft.weapon_upgrade",
        "crew.recruitment",
        "spacecraft.performance_upgrade",
        "space.launch",
        "planet.special_mineral",
        "alien.combat",
        "colony.colonization",
    ]

    for cap in expected_capabilities:
        assert cap in valid_capabilities

    # Check source partition with ignored_spans (context + catch-all)
    ignored_spans = [(0, 7), (134, 148)]
    nodes = []
    for idx, leaf in enumerate(final_atomic_leaves):
        if "source_start" not in leaf:
            grounding = _ground_source_anchor(clause, str(leaf["source_anchor"]))
            assert grounding is not None
            leaf = {**leaf, **grounding}
        nodes.append(
            {
                "source_clause_index": 0,
                "source_start": leaf["source_start"],
                "source_end": leaf["source_end"],
                "capability_id": expected_capabilities[idx],
            }
        )
    diagnostics = validate_semantic_source_partition(nodes, [clause], ignored_spans=ignored_spans)
    assert diagnostics == ()
