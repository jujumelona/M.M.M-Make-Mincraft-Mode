from __future__ import annotations

from minecraft_mod_ai.request_requirements import (
    LeafAtomicityStatus,
    detected_action_families,
    filter_and_split_context,
    validate_leaf_atomicity,
)


def _leaf(anchor: str, statement: str, *, then: str | None = None) -> dict[str, object]:
    return {
        "source_clause_index": 0,
        "source_anchor": anchor,
        "semantic_statement": statement,
        "given": "The relevant game state exists",
        "when": statement,
        "then": then or f"The observable result of {statement} occurs",
        "semantic_type": "gameplay_mechanic",
    }


def test_atomicity_rejects_compound_actions_across_domains() -> None:
    examples = (
        _leaf("농사 요리 판매", "농사를 하고 요리하고 음식을 판매한다"),
        _leaf("mine ore and build machines", "mine ore and build machines"),
        _leaf("무기 선원 성능 업그레이드 확장", "무기, 선원, 성능을 업그레이드하고 확장한다"),
    )
    for leaf in examples:
        status, reason = validate_leaf_atomicity(leaf)
        assert status == LeafAtomicityStatus.COMPOUND
        assert "action" in reason


def test_atomicity_accepts_one_observable_action() -> None:
    status, reason = validate_leaf_atomicity(
        _leaf("광석을 채굴한다", "광석을 채굴한다", then="광석이 인벤토리에 들어온다")
    )
    assert status == LeafAtomicityStatus.ATOMIC
    assert reason == ""


def test_atomicity_separates_context_and_catch_all_without_domain_capabilities() -> None:
    context = _leaf("우주모드 인데", "우주모드 인데")
    catch_all = _leaf("등 여러가지가 가능한 모드", "등 여러가지가 가능한 모드")

    context_status, _ = validate_leaf_atomicity(context)
    catch_status, _ = validate_leaf_atomicity(catch_all)

    assert context_status == LeafAtomicityStatus.CONTEXT
    assert catch_status == LeafAtomicityStatus.CATCH_ALL


def test_filter_and_split_context_has_no_semantic_synthesis() -> None:
    atomic = _leaf("광석 채굴", "광석을 채굴한다")
    compound = _leaf("채굴하고 건설", "광석을 채굴하고 기계를 건설한다")
    context = _leaf("산업 모드", "산업 모드")
    catch_all = _leaf("기타 활동", "기타 활동")
    original = [atomic, compound, context, catch_all]

    accepted, rejected, contexts, dropped = filter_and_split_context(original)

    assert accepted == [atomic]
    assert rejected[0]["source_anchor"] == compound["source_anchor"]
    assert contexts == [context]
    assert dropped == [catch_all]
    assert rejected[0]["semantic_statement"] == compound["semantic_statement"]


def test_action_detection_is_authority_neutral() -> None:
    families = detected_action_families("mine ore, cook food, and sell it")
    assert set(families) >= {"gather", "produce", "trade"}
    assert all("." not in family for family in families)
    assert detected_action_families("Travel to the selected destination") == ("travel",)
    assert detected_action_families("spacecraft.weapon_upgrade") == ()
