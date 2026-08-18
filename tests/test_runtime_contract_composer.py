from __future__ import annotations

from functools import wraps
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.runtime_contract_composer import (
    ContractCompositionError,
    ContractStage,
    call_shape,
    callable_boundary,
    compose_contract_stages,
    composition_state,
)


def test_successful_composition_is_receipted_and_not_replayed() -> None:
    state_owner = SimpleNamespace()
    calls: list[str] = []

    def first() -> None:
        calls.append("first")

    def second() -> None:
        calls.append("second")

    stages = (
        ContractStage("first", first),
        ContractStage("second", second),
    )
    receipts = compose_contract_stages(
        owner_name="unit-success",
        version=1,
        state_owner=state_owner,
        stages=stages,
    )
    repeated = compose_contract_stages(
        owner_name="unit-success",
        version=1,
        state_owner=state_owner,
        stages=stages,
    )

    assert calls == ["first", "second"]
    assert tuple(receipt.name for receipt in receipts) == ("first", "second")
    assert repeated == receipts
    state = composition_state(state_owner, "unit-success")
    assert state is not None
    assert state["version"] == 1
    assert state["graph_signature"][0] == ("first", "second")
    assert state["completed"] == ("first", "second")
    assert state["receipts"] == receipts
    assert state["active"] is None
    assert state["failed"] is None
    assert state["installed"] is True


def test_failed_stage_poison_prevents_partial_replay() -> None:
    state_owner = SimpleNamespace()
    calls: list[str] = []

    def first() -> None:
        calls.append("first")

    def failing() -> None:
        calls.append("failing")
        raise ValueError("boom")

    stages = (
        ContractStage("first", first),
        ContractStage("failing", failing),
    )
    with pytest.raises(ContractCompositionError, match="failed at stage 'failing'"):
        compose_contract_stages(
            owner_name="unit-failure",
            version=1,
            state_owner=state_owner,
            stages=stages,
        )

    with pytest.raises(ContractCompositionError, match="poisoned by prior failure"):
        compose_contract_stages(
            owner_name="unit-failure",
            version=1,
            state_owner=state_owner,
            stages=stages,
        )

    assert calls == ["first", "failing"]
    state = composition_state(state_owner, "unit-failure")
    assert state is not None
    assert state["completed"] == ("first",)
    assert state["failed"]["stage"] == "failing"
    assert state["failed"]["type"] == "ValueError"


def test_callable_boundary_cannot_be_destroyed_by_contract_stage() -> None:
    state_owner = SimpleNamespace()
    runtime = SimpleNamespace(handler=lambda: "ok")

    def destroy_handler() -> None:
        runtime.handler = None

    with pytest.raises(ContractCompositionError, match="destroyed callable boundary"):
        compose_contract_stages(
            owner_name="unit-boundary",
            version=1,
            state_owner=state_owner,
            stages=(ContractStage("bad-wrapper", destroy_handler),),
            boundaries=(
                callable_boundary("runtime.handler", runtime, "handler"),
            ),
        )

    state = composition_state(state_owner, "unit-boundary")
    assert state is not None
    assert state["failed"]["stage"] == "bad-wrapper"


def test_callable_boundary_rejects_signature_drift_without_executing_wrapper() -> None:
    state_owner = SimpleNamespace()

    def handler(left: object, right: object, *, mode: object) -> None:
        del left, right, mode

    runtime = SimpleNamespace(handler=handler)

    def narrow_wrapper() -> None:
        # functools.wraps normally makes inspect.signature() report the original
        # signature through __wrapped__. The real outer callable still accepts only
        # one argument, so the composer must validate follow_wrapped=False.
        @wraps(handler)
        def replacement(left: object) -> None:
            del left

        runtime.handler = replacement

    with pytest.raises(ContractCompositionError, match="changed call signature"):
        compose_contract_stages(
            owner_name="unit-signature",
            version=1,
            state_owner=state_owner,
            stages=(ContractStage("narrow-wrapper", narrow_wrapper),),
            boundaries=(
                callable_boundary(
                    "runtime.handler",
                    runtime,
                    "handler",
                    call_shapes=(call_shape(2, "mode"),),
                ),
            ),
        )

    state = composition_state(state_owner, "unit-signature")
    assert state is not None
    assert state["failed"]["stage"] == "narrow-wrapper"


def test_recursive_composition_is_rejected_and_poisoned() -> None:
    state_owner = SimpleNamespace()
    stages: tuple[ContractStage, ...]

    def recursive() -> None:
        compose_contract_stages(
            owner_name="unit-recursive",
            version=1,
            state_owner=state_owner,
            stages=stages,
        )

    stages = (ContractStage("recursive", recursive),)
    with pytest.raises(ContractCompositionError, match="re-entered while stage 'recursive'"):
        compose_contract_stages(
            owner_name="unit-recursive",
            version=1,
            state_owner=state_owner,
            stages=stages,
        )

    state = composition_state(state_owner, "unit-recursive")
    assert state is not None
    assert state["failed"]["stage"] == "recursive"


def test_successful_version_change_requires_clean_process() -> None:
    state_owner = SimpleNamespace()
    calls: list[int] = []

    def install() -> None:
        calls.append(len(calls) + 1)

    stages = (ContractStage("only", install),)
    compose_contract_stages(
        owner_name="unit-version",
        version=1,
        state_owner=state_owner,
        stages=stages,
    )
    with pytest.raises(
        ContractCompositionError,
        match="process restart is required before requesting version 2",
    ):
        compose_contract_stages(
            owner_name="unit-version",
            version=2,
            state_owner=state_owner,
            stages=stages,
        )

    assert calls == [1]
    state = composition_state(state_owner, "unit-version")
    assert state is not None
    assert state["version"] == 1
    assert state["completed"] == ("only",)
    assert state["installed"] is True


def test_version_bump_cannot_bypass_poisoned_process_state() -> None:
    state_owner = SimpleNamespace()
    calls: list[str] = []

    def failing() -> None:
        calls.append("failing")
        raise RuntimeError("partial mutation")

    stages = (ContractStage("failing", failing),)
    with pytest.raises(ContractCompositionError, match="failed at stage 'failing'"):
        compose_contract_stages(
            owner_name="unit-version-poison",
            version=1,
            state_owner=state_owner,
            stages=stages,
        )

    with pytest.raises(
        ContractCompositionError,
        match="process restart is required before requesting version 2",
    ):
        compose_contract_stages(
            owner_name="unit-version-poison",
            version=2,
            state_owner=state_owner,
            stages=stages,
        )

    assert calls == ["failing"]
    state = composition_state(state_owner, "unit-version-poison")
    assert state is not None
    assert state["version"] == 1
    assert state["failed"]["stage"] == "failing"


def test_same_version_graph_change_is_rejected() -> None:
    state_owner = SimpleNamespace()
    calls: list[str] = []

    def first() -> None:
        calls.append("first")

    def second() -> None:
        calls.append("second")

    compose_contract_stages(
        owner_name="unit-graph",
        version=7,
        state_owner=state_owner,
        stages=(ContractStage("first", first),),
    )
    with pytest.raises(ContractCompositionError, match="graph changed"):
        compose_contract_stages(
            owner_name="unit-graph",
            version=7,
            state_owner=state_owner,
            stages=(
                ContractStage("first", first),
                ContractStage("second", second),
            ),
        )

    assert calls == ["first"]
    state = composition_state(state_owner, "unit-graph")
    assert state is not None
    assert state["version"] == 7
    assert state["graph_signature"][0] == ("first",)


def test_duplicate_stage_names_are_rejected_before_install() -> None:
    state_owner = SimpleNamespace()
    calls = 0

    def install() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(ContractCompositionError, match="duplicate stages"):
        compose_contract_stages(
            owner_name="unit-duplicates",
            version=1,
            state_owner=state_owner,
            stages=(
                ContractStage("same", install),
                ContractStage("same", install),
            ),
        )

    assert calls == 0
