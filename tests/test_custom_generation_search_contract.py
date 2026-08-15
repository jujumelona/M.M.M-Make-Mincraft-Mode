from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator


@dataclass
class _Module:
    kind: str
    config: dict = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()


class _Router:
    def __init__(self) -> None:
        self.calls = []

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return "ok"


def test_custom_search_width_is_risk_adaptive_when_native_slots_exist(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    assert custom_search._width(_Module(kind="item")) == 1
    risky = _Module(
        kind="custom_java",
        config={"networking": "server authoritative", "persistence": True},
        depends_on=("state", "protocol"),
    )
    assert custom_search._width(risky) == 2


def test_custom_search_auto_never_serializes_candidates_on_one_slot(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "auto")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "3")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    risky = _Module(
        kind="custom_java",
        config={"networking": "server authoritative", "persistence": True},
        depends_on=("state", "protocol"),
    )
    assert custom_search._width(risky) == 1


def test_strategy_router_only_augments_coder_role() -> None:
    base = _Router()
    router = custom_search._StrategyRouter(
        base,
        strategy="api_contract_first",
        candidate_index=1,
        count=2,
    )
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "task"},
    ]
    router.generate_text("coder", messages, response_format="json")
    assert len(base.calls[0][1]) == 3
    assert "api_contract_first" in base.calls[0][1][1]["content"]

    router.generate_text("planner", messages, response_format="json")
    assert base.calls[1][1] == messages


def test_custom_generation_public_target_defaults_are_disabled() -> None:
    signature = inspect.signature(CustomModuleGenerator.generate)
    assert signature.parameters["minecraft_version"].default is None
    assert signature.parameters["loader"].default is None
    assert signature.parameters["mappings"].default is None


def test_target_values_fail_closed_without_complete_host_target() -> None:
    with pytest.raises(ValueError, match="must provide minecraft_version"):
        custom_search._target_values(
            {
                "minecraft_version": "1.20.1",
                "loader": "fabric",
            }
        )
