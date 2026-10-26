"""The host decides occurrence ownership before any MiMo hook/activation verdict."""
import json

import pytest

from minecraft_mod_ai.host_action_occurrence import (
    author_host_owned_player_occurrence, select_original_player_action,
)
from minecraft_mod_ai.source_owned_ui_event import (
    materialize_typed_ui_occurrence, read_verified_typed_ui_events,
)


TRADE = {
    "behavior_id": "behavior_7e44c67dc12bf6885bb9",
    "source_quote": "거래 등으로",
    "intent": "Trade and purchase goods to fund spacecraft construction and upgrades",
    "trigger": "Player has accumulated money and wants to build or upgrade a spacecraft",
    "effects": "Player trades mined resources for goods, then purchases spacecraft construction materials and upgrades",
    "runtime_state_reads": ["available goods", "spacecraft build requirements", "spacecraft upgrade requirements"],
    "runtime_state_writes": ["player inventory"],
}
SYMBOLS = {name: {"kind": "CLASS", "owner": name} for name in (
    "net.minecraft.commands.CommandSourceStack", "net.fabricmc.fabric.api.event.Event",
    "net.fabricmc.fabric.api.event.EventFactory",
    "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback",
)}


class SmallMiMo:
    def __init__(self):
        self.called = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.called.append(data_name)
        # The old production session returned 'unavailable' for a player
        # trade. The new host cannot request that decision at all.
        if data_name in {"source_owned_player_action_evidence",
                         "source_owned_ui_activation",
                         "select_inspected_source_producer_site",
                         "select_source_owned_observation_hook"}:
            raise AssertionError("MiMo cannot veto a host-owned action source")
        return {"ui_action_label": {"label": "Trade resources"},
                "ui_action_input_count": {"count": 0}}[data_name]


def test_original_trade_compiles_an_occurrence_without_asking_mimo_to_decide_event(tmp_path):
    router = SmallMiMo()
    chosen = select_original_player_action(TRADE)
    assert chosen["request_field"] == "effects"
    assert chosen["request_quote"].startswith("Player trades mined resources")
    plan = author_host_owned_player_occurrence(
        router, TRADE, symbols=SYMBOLS, package="example.generated",
        mod_id="trade_test", minecraft_version="26.2")
    assert router.called == ["ui_action_label", "ui_action_input_count"]
    assert plan["callback"]["parameters"] == ["net.minecraft.commands.CommandSourceStack"]
    manifest = tmp_path / "src/main/resources/fabric.mod.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"id":"trade_test"}')
    materialize_typed_ui_occurrence(plan, project_root=tmp_path, symbols=SYMBOLS)
    events = read_verified_typed_ui_events(
        project_root=tmp_path, symbols=SYMBOLS, behaviors=[TRADE],
        package="example.generated")
    assert events == [plan["callback"]]
    assert len(json.loads(manifest.read_text())["entrypoints"]["main"]) == 1


@pytest.mark.parametrize("trigger,effects", [
    ("Player reaches a new planet", "The server records the arrival"),
    ("Player takes damage", "The server subtracts health"),
    ("Player dies", "Drops are recorded automatically"),
    ("Player has enough credits", "The server purchases resources automatically"),
    ("Player wants to buy equipment", "Credits are already available"),
    ("Player selects a target automatically", "The target is processed"),
    ("Player must not purchase supplies", "Player purchases supplies"),
])
def test_passive_conditions_and_prohibitions_never_become_commands(trigger, effects):
    assert select_original_player_action({
        **TRADE, "trigger": trigger, "effects": effects, "source_quote": "test"}) is None


def test_actor_verb_not_in_allowlist_can_still_be_an_action():
    candidate = select_original_player_action({
        **TRADE, "trigger": "Player has enough materials",
        "effects": "Player commissions a new construction project"})
    assert candidate["request_quote"].startswith("Player commissions")


def test_korean_player_purchase_is_an_explicit_action():
    candidate = select_original_player_action({
        **TRADE, "trigger": "플레이어가 필요한 물품을 구매한다",
        "effects": "The server records it",
        "source_quote": "필요한 물품을 구매한다"})
    assert candidate is not None
    assert candidate["request_field"] == "trigger"


def test_the_canonical_scope_cannot_be_changed_after_planning():
    behavior = {**TRADE, "trigger": "Player does not purchase goods",
                "effects": "Player purchases goods"}
    assert select_original_player_action(behavior) is None


def test_host_action_with_automatic_server_side_effect_keeps_manual_producer():
    contract = select_original_player_action({
        **TRADE, "trigger": "Player has enough credits",
        "effects": "Player purchases supplies and the server automatically subtracts the price",
    })
    assert contract is not None
    assert contract["request_quote"].startswith("Player purchases supplies")


def test_production_trade_java_compiles_against_real_fabric_26_2_sdk(tmp_path):
    """Enabled only on the CI lane that provisions the actual 26.2 SDK."""
    import os
    import shutil
    import subprocess

    classpath = os.environ.get("MMM_ACTION_COMPILE_CLASSPATH")
    if not classpath:
        pytest.skip("real Minecraft 26.2 SDK lane required")
    javac = shutil.which("javac")
    assert javac, "Minecraft SDK test requires JDK 25"
    plan = author_host_owned_player_occurrence(
        SmallMiMo(), TRADE, symbols=SYMBOLS,
        package="example.generated", mod_id="trade_test",
        minecraft_version="26.2",
    )
    files = []
    for name, content in plan["files"].items():
        if not name.endswith(".java"):
            continue
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        files.append(str(target))
    result = subprocess.run(
        [javac, "-cp", classpath, "-d", str(tmp_path / "classes"), *files],
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert result.returncode == 0, result.stderr
