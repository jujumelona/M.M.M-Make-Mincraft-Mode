from __future__ import annotations

import pytest

from minecraft_mod_ai import ModAISession
from minecraft_mod_ai.planner import HeuristicPlanner, _proposal_from_model_data
from minecraft_mod_ai.webui import _explicit_focus, _project_scope


def _capabilities(prompt: str) -> set[str]:
    proposal = HeuristicPlanner().plan(prompt)
    return {request.capability for request in proposal.deferred_requests}


def _diagnostic_session(tmp_path, name: str) -> ModAISession:
    return ModAISession(output_root=tmp_path / name, planner=HeuristicPlanner())


def test_ascii_terms_require_token_or_phrase_boundaries() -> None:
    prompt = "Use a maple palette for a core exploration mechanic in a classic mobile style."
    proposal = HeuristicPlanner().plan(prompt)

    assert proposal.spec.mod_id == "maple_works"
    assert proposal.spec.contents == ()
    assert proposal.spec.boss is None
    assert _capabilities(prompt) == {"creative_brief"}
    assert _explicit_focus(prompt) == ()


@pytest.mark.parametrize(
    "removal",
    ("Remove the boss.", "The boss should not be included.", "보스는 넣지 마.", "보스는 만들지 마."),
)
def test_latest_removal_phrases_remove_boss(removal: str) -> None:
    proposal = HeuristicPlanner().plan(
        f"Create one frost item and a humanoid melee boss.\n{removal}"
    )

    assert len(proposal.spec.contents) == 1
    assert proposal.spec.boss is None
    assert "보스" not in _explicit_focus(proposal.requested_prompt)


def test_later_positive_intent_can_readd_a_removed_category() -> None:
    prompt = "Create a humanoid melee boss.\nRemove the boss.\nAdd a humanoid melee boss."
    proposal = HeuristicPlanner().plan(prompt)

    assert proposal.spec.boss is not None
    assert "보스" in _explicit_focus(prompt)


def test_removing_items_discards_an_earlier_item_count() -> None:
    proposal = HeuristicPlanner().plan("Create three frost items.\nRemove the items.")
    assert proposal.spec.contents == ()


def test_map_words_do_not_create_a_map_capability_or_focus() -> None:
    prompt = "Create one item with a village, field, dungeon, and arena."
    proposal = HeuristicPlanner().plan(prompt)
    assert len(proposal.spec.contents) == 1
    assert not _capabilities(prompt) & {
        "village_map",
        "field_map",
        "dungeon_map",
        "custom_map",
    }
    assert _explicit_focus(prompt) == ("아이템",)


@pytest.mark.parametrize(
    ("word", "expected"),
    (("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6), ("seven", 7), ("eight", 8)),
)
def test_english_number_words_one_through_eight(word: str, expected: int) -> None:
    assert len(HeuristicPlanner().plan(f"Create {word} frost items.").spec.contents) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    (("하나", 1), ("둘", 2), ("셋", 3), ("넷", 4), ("다섯", 5), ("여섯", 6), ("일곱", 7), ("여덟", 8)),
)
def test_korean_number_words_one_through_eight(word: str, expected: int) -> None:
    assert len(HeuristicPlanner().plan(f"서리 아이템 {word} 개를 만들어줘.").spec.contents) == expected


def test_dimensions_in_a_map_phrase_do_not_change_requested_item_count() -> None:
    proposal = HeuristicPlanner().plan("Create a 41x41 arena and one frost item.")
    assert len(proposal.spec.contents) == 1
    assert _explicit_focus(proposal.requested_prompt) == ("아이템",)


def test_large_explicit_item_count_has_no_legacy_eight_item_cap(tmp_path) -> None:
    reply = _diagnostic_session(tmp_path, "output").plan("Create 100 frost items.")
    assert len(reply.proposal.spec.contents) == 100
    assert "item_count_limit" not in {
        request.capability for request in reply.proposal.deferred_requests
    }


def test_model_output_cannot_exceed_requested_kind_cardinality() -> None:
    data = {
        "mod_id": "remote_cardinality",
        "mod_name": "Remote Cardinality",
        "package_name": "ai.minecraft.generated.remote_cardinality",
        "summary": "Remote candidate",
        "contents": [
            {
                "content_id": f"remote_block_{index}",
                "kind": "block",
                "display_name_en": f"Remote Block {index}",
                "display_name_ko": f"원격 블록 {index}",
                "color": "#00ff00",
                "recipe": True,
            }
            for index in range(5)
        ],
        "deferred_capabilities": [],
    }
    proposal = _proposal_from_model_data("Create exactly one green block.", data)
    assert len(proposal.spec.contents) == 1
    assert proposal.spec.contents[0].kind.value == "block"


def test_huge_content_does_not_set_huge_project_scope() -> None:
    prompt = "Create exactly one item. Make it a huge sword."
    proposal = HeuristicPlanner().plan(prompt)
    assert _project_scope(prompt, proposal) == ("현재 요청 기준 소규모 단일 기능", True)
