from __future__ import annotations

from minecraft_mod_ai.planning_authority import _host_causal_dependencies


def _req(rid: str, start: int, given: str, then: str, *, depends_on=()):
    return {
        "requirement_id": rid,
        "source_span": {"char_start": start},
        "depends_on": list(depends_on),
        "unlock_policy": {"required_requirement_refs": []},
        "observable_behavior": {
            "given": given,
            "when": "the player acts",
            "then": then,
        },
    }


def test_space_mode_observable_prerequisites_cannot_collapse_to_zero_edges():
    requirements = [
        _req(
            "resource",
            8,
            "The game is running in Space Mode.",
            "Resources are collected and added to the player's inventory.",
        ),
        _req(
            "ship",
            25,
            "The game is running in Space Mode and the player has gathered the necessary resources.",
            "A functional spacecraft is created from the assembled segments.",
        ),
        _req(
            "weapon",
            47,
            "The game is running in Space Mode and the player has a constructed spacecraft.",
            "The spacecraft is configured with the specified weapons and crew.",
        ),
        _req(
            "upgrade",
            53,
            "The game is running in Space Mode and the player has a spacecraft.",
            "The spacecraft's performance attributes are improved or expanded.",
        ),
        _req(
            "travel",
            87,
            "The game is running in Space Mode and the player has a functional spacecraft.",
            "The spacecraft travels from the planet to space.",
        ),
        _req(
            "explore",
            103,
            "The player has traveled to space and arrived at another planet.",
            "Special minerals are discovered and can be collected.",
        ),
        _req(
            "combat",
            124,
            "The player is on another planet and encounters an alien entity.",
            "Combat ensues, and the player can defeat or be defeated by the alien.",
        ),
        _req(
            "colony",
            132,
            "The player has established a presence on another planet.",
            "The planet becomes colonized, allowing for further development.",
        ),
    ]

    dependencies, provenance = _host_causal_dependencies(requirements)

    assert dependencies["ship"] == ["resource"]
    assert dependencies["weapon"] == ["ship"]
    assert dependencies["upgrade"] == ["ship"]
    assert dependencies["travel"] == ["ship"]
    assert dependencies["explore"] == ["travel"]
    assert dependencies["combat"] == ["travel"]
    assert dependencies["colony"] == ["travel"]
    assert len(provenance) == 7


def test_mention_order_alone_never_creates_dependency():
    requirements = [
        _req(
            "first",
            0,
            "The player has a spacecraft.",
            "The spacecraft is configured with a cosmetic emblem.",
        ),
        _req(
            "second",
            20,
            "The player has a spacecraft.",
            "The spacecraft is configured with a radio channel.",
        ),
    ]

    dependencies, provenance = _host_causal_dependencies(requirements)

    assert dependencies == {"first": [], "second": []}
    assert provenance == []


def test_declared_dependency_is_preserved_even_without_lexical_inference():
    requirements = [
        _req("foundation", 0, "A vanilla world exists.", "A station is created."),
        _req(
            "feature",
            50,
            "A separate authored condition exists.",
            "The feature becomes available.",
            depends_on=("foundation",),
        ),
    ]

    dependencies, _ = _host_causal_dependencies(requirements)

    assert dependencies["feature"] == ["foundation"]
