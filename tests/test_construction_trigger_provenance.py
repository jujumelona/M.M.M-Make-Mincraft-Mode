"""Source-derived input provenance across fresh, resumed and translated behavior text.

Tests target the *entire class* of keyboard/mouse triggers. They deliberately
do not special-case the user's gameplay theme, a particular button or key.
"""
from __future__ import annotations

import pytest

from minecraft_mod_ai.construction_trigger_provenance import (
    claim_is_physical_input,
    preserve_grounded_trigger,
)


@pytest.mark.parametrize("phrase", [
    "Space key pressed",
    "Press E to enter",
    "Mouse button clicked",
    "사용자가 키를 누르면",
    "마우스를 클릭하면",
])
def test_physical_input_claim_detection(phrase):
    assert claim_is_physical_input(phrase)


@pytest.mark.parametrize("source,claim", [
    ("Launch into orbit after a vehicle is built", "Keyboard shortcut pressed"),
    ("우주로 나가서 광물을 캔다", "Space key pressed"),
    ("The avatar travels after the gateway activates", "Mouse clicked"),
])
def test_unsourced_control_input_never_becomes_gameplay_requirement(source, claim):
    cleaned, error = preserve_grounded_trigger(
        claim, source_quote=source, source_text=source)
    assert cleaned == ""
    assert error["code"] == "CONSTRUCTION_TRIGGER_INPUT_NOT_REQUESTED"
    assert error["claimed_trigger"] == claim


@pytest.mark.parametrize("source,claim", [
    ("Press the J key to activate a generator", "J key pressed"),
    ("E키를 누르면 문이 열리게 해", "E key pressed"),
    ("Click the mouse button to fire", "Mouse button clicked"),
])
def test_requested_controls_remain_intact(source, claim):
    cleaned, error = preserve_grounded_trigger(
        claim, source_quote=source, source_text=source)
    assert cleaned == claim
    assert error is None


def test_world_condition_is_not_confused_with_physical_input():
    cleaned, error = preserve_grounded_trigger(
        "World time reaches daybreak",
        source_quote="When the sun rises, begin the encounter",
        source_text="When the sun rises, begin the encounter")
    assert cleaned == "World time reaches daybreak"
    assert error is None
