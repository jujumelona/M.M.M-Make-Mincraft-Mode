from __future__ import annotations

import json
import threading
import time
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import minecraft_mod_ai.complete_planner as planner_module
from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _ModuleCatalog,
    _ProductionBatch,
    _ProductionParts,
    _ProductionWaveConflict,
    _merge_production_batch_wave,
)
from minecraft_mod_ai.complete_spec import (
    AssetRequest,
    AudioRequest,
    ProductionModule,
)


_TRACE = ContextVar("mmm_test_production_batch_trace", default="missing")


class _Registry:
    def __init__(self, provider: str = "local") -> None:
        self.provider = provider

    def role(self, profile: str, role: str) -> SimpleNamespace:
        del profile
        return SimpleNamespace(
            role=role,
            provider=self.provider,
            adapter="llama_cpp",
            exclusive_gpu=True,
        )


def _module(module_id: str) -> dict[str, Any]:
    return {
        "module_id": module_id,
        "kind": "custom_java",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }


def _page(
    request: dict[str, Any],
    *,
    module_id: str | None = None,
    assets: list[dict[str, Any]] | None = None,
    audio: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    deliverable = request["current_target_deliverables"][0]
    return {
        "modules": [] if module_id is None else [_module(module_id)],
        "assets": list(assets or ()),
        "audio": list(audio or ()),
        "acceptance_tests": [f"{deliverable} works"],
        "completed_deliverables": [deliverable],
        "complete": True,
        "next_cursor": "",
    }


class _BatchRouter:
    profile = "test"

    def __init__(
        self,
        responder: Callable[[str, dict[str, Any]], dict[str, Any]],
        *,
        provider: str = "local",
        initial_barrier: bool = False,
        delays: dict[str, float] | None = None,
        fail_batch: str = "",
    ) -> None:
        self.registry = _Registry(provider)
        self.responder = responder
        self.initial_barrier = threading.Barrier(2) if initial_barrier else None
        self.delays = dict(delays or {})
        self.fail_batch = fail_batch
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.events: list[tuple[str, str, float]] = []
        self.requests: list[dict[str, Any]] = []
        self.traces: list[str] = []

    def generate_text(self, role: str, messages: Any, **kwargs: Any) -> str:
        del kwargs
        assert role == "planner"
        request = json.loads(messages[-1]["content"])
        batch_id = str(request["batch"]["batch_id"])
        retry = "parallel_wave_retry" in request["planning_context_receipt"]
        with self.lock:
            self.requests.append(request)
            self.traces.append(_TRACE.get())
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.events.append(("start", batch_id, time.monotonic()))
        try:
            if (
                self.initial_barrier is not None
                and not retry
                and batch_id in {"alpha", "beta"}
            ):
                self.initial_barrier.wait(timeout=5)
            time.sleep(self.delays.get(batch_id, 0.0))
            if batch_id == self.fail_batch:
                raise RuntimeError(f"forced failure for {batch_id}")
            return json.dumps(self.responder(batch_id, request))
        finally:
            with self.lock:
                self.events.append(("end", batch_id, time.monotonic()))
                self.active -= 1


def _batch(
    batch_id: str,
    *,
    dependencies: tuple[str, ...] = (),
    export: str = "",
) -> _ProductionBatch:
    return _ProductionBatch(
        batch_id=batch_id,
        scope=f"Implement {batch_id}.",
        depends_on_batches=dependencies,
        deliverables=(f"{batch_id}_deliverable",),
        exports=(() if not export else (export,)),
    )


def _configure_parallel(monkeypatch: pytest.MonkeyPatch, checkpoint_dir: Any) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(checkpoint_dir))


def _signature(parts: _ProductionParts) -> tuple[Any, ...]:
    return (
        tuple((item.module_id, item.depends_on) for item in parts.modules),
        tuple((item.asset_id, item.target_path) for item in parts.assets),
        tuple(item.sound_id for item in parts.audio),
        tuple(parts.acceptance_tests),
    )


def test_independent_batches_overlap_and_dependencies_wait_for_wave_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _configure_parallel(monkeypatch, tmp_path / "checkpoints")
    token = _TRACE.set("copied-context")
    try:
        router = _BatchRouter(
            lambda batch_id, request: _page(
                request,
                module_id=f"{batch_id}_module",
            ),
            initial_barrier=True,
            delays={"alpha": 0.08, "beta": 0.01, "gamma": 0.01},
        )
        parts = CompleteGameDesignPlanner(router)._expand_production_batches(
            batches=(
                _batch("beta", export="beta_module"),
                _batch("gamma", export="gamma_module"),
                _batch("alpha", export="alpha_module"),
                _batch(
                    "dependent",
                    dependencies=("alpha", "beta", "gamma"),
                    export="dependent_module",
                ),
            ),
            prompt="Build four systems.",
            game_design={"title": "Parallel"},
            media_paths=(),
            enforce_batch_dependencies=True,
        )
    finally:
        _TRACE.reset(token)

    assert router.max_active == 2
    assert router.traces == ["copied-context"] * 4
    starts = {batch: stamp for event, batch, stamp in router.events if event == "start"}
    ends = {batch: stamp for event, batch, stamp in router.events if event == "end"}
    assert starts["dependent"] >= max(
        ends["alpha"],
        ends["beta"],
        ends["gamma"],
    )
    assert [item.module_id for item in parts.modules] == [
        "alpha_module",
        "beta_module",
        "gamma_module",
        "dependent_module",
    ]
    assert parts.modules[-1].depends_on == (
        "alpha_module",
        "beta_module",
        "gamma_module",
    )


def test_completion_order_does_not_change_parallel_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    batches = (_batch("alpha"), _batch("beta"))
    signatures = []
    for index, delays in enumerate(
        ({"alpha": 0.04, "beta": 0.0}, {"alpha": 0.0, "beta": 0.04})
    ):
        _configure_parallel(monkeypatch, tmp_path / f"checkpoints-{index}")
        router = _BatchRouter(
            lambda batch_id, request: _page(
                request,
                module_id=f"{batch_id}_module",
            ),
            initial_barrier=True,
            delays=delays,
        )
        parts = CompleteGameDesignPlanner(router)._expand_production_batches(
            batches=batches,
            prompt="Build both systems.",
            game_design={"title": "Stable"},
            media_paths=(),
        )
        signatures.append(_signature(parts))

    assert signatures[0] == signatures[1]


@pytest.mark.parametrize(
    ("provider", "active_slots"),
    (("local", "1"), ("remote", "2")),
)
def test_one_slot_and_non_local_router_keep_serial_catalog_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    provider: str,
    active_slots: str,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", active_slots)
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path / provider))
    router = _BatchRouter(
        lambda batch_id, request: _page(
            request,
            module_id=f"{batch_id}_module",
        ),
        provider=provider,
        delays={"alpha": 0.01, "beta": 0.01},
    )

    CompleteGameDesignPlanner(router)._expand_production_batches(
        batches=(_batch("alpha"), _batch("beta")),
        prompt="Build both systems.",
        game_design={"title": "Serial"},
        media_paths=(),
    )

    assert router.max_active == 1
    assert [
        request["known_module_catalog"]["count"]
        for request in router.requests
    ] == [0, 1]


def test_serial_missing_export_is_committed_to_following_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "1")
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "off")
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))

    def respond(batch_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return _page(
            request,
            module_id=None if batch_id == "alpha" else "beta_module",
        )

    router = _BatchRouter(respond)
    parts = CompleteGameDesignPlanner(router)._expand_production_batches(
        batches=(
            _batch("alpha", export="alpha_export"),
            _batch("beta", export="beta_module"),
        ),
        prompt="Build both systems.",
        game_design={"title": "Exports"},
        media_paths=(),
    )

    assert [item.module_id for item in parts.modules] == [
        "alpha_export",
        "beta_module",
    ]
    assert router.requests[1]["known_module_catalog"]["count"] == 1


def test_worker_failure_never_enters_wave_merge_or_dependent_wave(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _configure_parallel(monkeypatch, tmp_path)
    router = _BatchRouter(
        lambda batch_id, request: _page(
            request,
            module_id=f"{batch_id}_module",
        ),
        initial_barrier=True,
        fail_batch="beta",
    )
    merge_calls = 0
    current_merge = planner_module._merge_production_batch_wave

    def counted_merge(**kwargs: Any):
        nonlocal merge_calls
        merge_calls += 1
        return current_merge(**kwargs)

    monkeypatch.setattr(planner_module, "_merge_production_batch_wave", counted_merge)
    with pytest.raises(RuntimeError, match="forced failure for beta"):
        CompleteGameDesignPlanner(router)._expand_production_batches(
            batches=(
                _batch("alpha"),
                _batch("beta"),
                _batch("dependent", dependencies=("alpha", "beta")),
            ),
            prompt="Build all systems.",
            game_design={"title": "Atomic"},
            media_paths=(),
        )

    assert merge_calls == 0
    assert "dependent" not in [batch for _, batch, _ in router.events]


def test_parallel_identity_collision_retries_once_against_staged_catalogs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _configure_parallel(monkeypatch, tmp_path)
    router = _BatchRouter(
        lambda batch_id, request: _page(request, module_id="shared_module"),
        initial_barrier=True,
    )

    parts = CompleteGameDesignPlanner(router)._expand_production_batches(
        batches=(_batch("alpha"), _batch("beta")),
        prompt="Build both systems.",
        game_design={"title": "Collision recovery"},
        media_paths=(),
    )

    retry_requests = [
        request
        for request in router.requests
        if "parallel_wave_retry" in request["planning_context_receipt"]
    ]
    assert len(router.requests) == 4
    assert [item.module_id for item in parts.modules] == [
        "shared_module",
        "shared_module_2",
    ]
    assert [request["known_module_catalog"]["count"] for request in retry_requests] == [0, 1]
    assert {
        request["planning_context_receipt"]["parallel_wave_retry"]["attempt"]
        for request in retry_requests
    } == {1}


def test_asset_path_collision_fails_closed_after_one_serial_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _configure_parallel(monkeypatch, tmp_path)

    def respond(batch_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return _page(
            request,
            assets=[
                {
                    "asset_id": f"{batch_id}_asset",
                    "kind": "item",
                    "prompt": f"Texture for {batch_id}",
                    "target_path": "assets/mmm/textures/item/shared.png",
                    "width": 16,
                    "height": 16,
                }
            ],
        )

    router = _BatchRouter(respond, initial_barrier=True)
    with pytest.raises(_ProductionWaveConflict, match="asset target path"):
        CompleteGameDesignPlanner(router)._expand_production_batches(
            batches=(_batch("alpha"), _batch("beta")),
            prompt="Build both systems.",
            game_design={"title": "Path collision"},
            media_paths=(),
        )

    assert len(router.requests) == 4


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("module", "duplicate module ID"),
        ("asset_id", "asset ID"),
        ("asset_path", "asset target path"),
        ("audio", "audio ID.*audio target path"),
    ),
)
def test_wave_conflicts_validate_all_ids_and_derived_paths_atomically(
    kind: str,
    message: str,
) -> None:
    first = _ProductionParts([], [], [], [])
    second = _ProductionParts([], [], [], [])
    if kind == "module":
        first.modules.append(ProductionModule("shared", "custom_java"))
        second.modules.append(ProductionModule("shared", "custom_java"))
    elif kind == "asset_id":
        first.assets.append(AssetRequest("shared", "item", "one", "one.png"))
        second.assets.append(AssetRequest("shared", "item", "two", "two.png"))
    elif kind == "asset_path":
        first.assets.append(AssetRequest("first", "item", "one", "same/path.png"))
        second.assets.append(AssetRequest("second", "item", "two", "same\\path.png"))
    else:
        first.audio.append(AudioRequest("shared", "effect", 1.0))
        second.audio.append(AudioRequest("shared", "effect", 1.0))

    parts = _ProductionParts([], [], [], [])
    module_catalog = _ModuleCatalog()
    asset_catalog = _ModuleCatalog()
    audio_catalog = _ModuleCatalog()
    with pytest.raises(_ProductionWaveConflict, match=message):
        _merge_production_batch_wave(
            parts=parts,
            module_catalog=module_catalog,
            asset_catalog=asset_catalog,
            audio_catalog=audio_catalog,
            test_catalog=set(),
            wave_results=((_batch("beta"), second), (_batch("alpha"), first)),
        )

    assert parts == _ProductionParts([], [], [], [])
    assert module_catalog.receipt()["count"] == 0
    assert asset_catalog.receipt()["count"] == 0
    assert audio_catalog.receipt()["count"] == 0
