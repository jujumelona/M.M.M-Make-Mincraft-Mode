from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable


class ReentrantReadWriteLock:
    """Writer-reentrant lock with shared readers and writer preference."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._writer_thread: int | None = None
        self._writer_depth = 0
        self._readers: dict[int, int] = {}
        self._waiting_writers = 0

    def acquire(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._writer_depth += 1
                return True
            self._waiting_writers += 1
            try:
                while self._writer_thread is not None or self._other_reader_count(owner):
                    self._condition.wait()
                self._writer_thread = owner
                self._writer_depth = 1
                return True
            finally:
                self._waiting_writers -= 1

    def release(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread != owner or self._writer_depth <= 0:
                raise RuntimeError("cannot release unowned GPU write lock")
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer_thread = None
                self._condition.notify_all()

    def acquire_read(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            if self._writer_thread == owner:
                self._readers[owner] = self._readers.get(owner, 0) + 1
                return True
            while self._writer_thread is not None or self._waiting_writers > 0:
                self._condition.wait()
            self._readers[owner] = self._readers.get(owner, 0) + 1
            return True

    def release_read(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            count = self._readers.get(owner, 0)
            if count <= 0:
                raise RuntimeError("cannot release unowned GPU read lock")
            if count == 1:
                self._readers.pop(owner, None)
            else:
                self._readers[owner] = count - 1
            if not self._readers:
                self._condition.notify_all()

    def _other_reader_count(self, owner: int) -> int:
        return sum(
            count
            for thread_id, count in self._readers.items()
            if thread_id != owner
        )

    def __enter__(self) -> ReentrantReadWriteLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    @contextmanager
    def shared(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()


class ReentrantCapacityGate:
    """Bound concurrent callers to a dynamic capacity without blocking re-entry."""

    def __init__(self, capacity: Callable[[], int]) -> None:
        self._capacity = capacity
        self._condition = threading.Condition(threading.RLock())
        self._owners: dict[int, int] = {}

    def acquire(self) -> bool:
        owner = threading.get_ident()
        with self._condition:
            depth = self._owners.get(owner, 0)
            if depth:
                self._owners[owner] = depth + 1
                return True
            while len(self._owners) >= max(1, int(self._capacity())):
                self._condition.wait()
            self._owners[owner] = 1
            return True

    def release(self) -> None:
        owner = threading.get_ident()
        with self._condition:
            depth = self._owners.get(owner, 0)
            if depth <= 0:
                raise RuntimeError("cannot release unowned llama inference slot")
            if depth == 1:
                self._owners.pop(owner, None)
                self._condition.notify_all()
            else:
                self._owners[owner] = depth - 1

    def __enter__(self) -> ReentrantCapacityGate:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _planner_parallel_capacity(router: Any, width: int) -> int:
    capacity = min(max(1, int(width)), _active_parallelism())
    if capacity <= 1:
        return 1
    try:
        config = router.registry.role(router.profile, "planner")
    except Exception:
        return 1
    if not bool(getattr(config, "exclusive_gpu", False)):
        return 1
    if str(getattr(config, "provider", "")) != "local":
        return 1
    if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
        return 1
    return capacity


def _install_router(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    if getattr(cls.generate_text, "_mmm_llama_shared_slots", False):
        return

    model_router_module._GPU_EXCLUSIVE_LOCK = ReentrantReadWriteLock()
    model_router_module._LLAMA_INFERENCE_SLOTS = ReentrantCapacityGate(
        _active_parallelism
    )

    @contextmanager
    def generation_session(self: Any, role: str):
        config = self.registry.role(self.profile, role)
        adapter = self._new_text_adapter(config, role=role)
        with self._generation_lock:
            if self._active_generation_adapter is not None:
                raise model_router_module.ModelConfigurationError(
                    "A generation session is already active for role "
                    f"{self._active_generation_role!r}."
                )
            self._active_generation_role = role
            self._active_generation_adapter = adapter

        session_factory = getattr(adapter, "generation_session", None)
        try:
            if callable(session_factory):
                with session_factory():
                    yield self
            else:
                try:
                    yield self
                finally:
                    adapter.close()
        finally:
            with self._generation_lock:
                if self._active_generation_adapter is adapter:
                    self._active_generation_adapter = None
                    self._active_generation_role = None

    generation_session._mmm_llama_shared_slots = True  # type: ignore[attr-defined]

    def generate_text(
        self: Any,
        role: str,
        messages: Any,
        *,
        media_paths: Any = (),
        response_format: str = "text",
        tool_stage: str | None = None,
        enable_tools: bool = True,
    ) -> str:
        # Keep router state selection protected, but release the router mutex before
        # native decode. Native llama/vLLM concurrency is bounded independently by the
        # server slot gate below. This preserves generation-session safety without
        # serializing independent Best-of-N candidates.
        with self._generation_lock:
            config = self.registry.role(self.profile, role)
            if self._active_generation_adapter is not None:
                if role != self._active_generation_role:
                    raise model_router_module.ModelConfigurationError(
                        "Generation session for role "
                        f"{self._active_generation_role!r} cannot serve role "
                        f"{role!r}."
                    )
                adapter = self._active_generation_adapter
            else:
                adapter = self._new_text_adapter(config, role=role)

        # Preserve the complete ModelRouter tool contract. The parallel runtime layer
        # is a lock/scheduling policy only; it must not silently remove Qwen's stage-
        # scoped MCP tools, tool choice, or parallel tool-call capability.
        stage = (tool_stage or model_router_module._ROLE_TOOL_STAGE.get(role, "")).strip().lower()
        runtime = None
        tools: tuple[Any, ...] = ()
        if self._tools_enabled(
            enable_tools=enable_tools,
            stage=stage,
            adapter_name=config.adapter,
        ):
            runtime = self._tool_runtime()
            tools = tuple(runtime.tool_schemas(stage))

        request = model_router_module.GenerationRequest(
            messages=messages,
            media_paths=tuple(Path(path) for path in media_paths),
            response_format=response_format,
            tools=tools,
            tool_choice="auto" if tools else None,
            parallel_tool_calls=True,
        )

        def run_generation() -> str:
            if runtime is not None and tools:
                return self._generate_with_tools(
                    adapter=adapter,
                    request=request,
                    runtime=runtime,
                    stage=stage,
                )
            return adapter.generate(request)

        shared_llama = (
            bool(config.exclusive_gpu)
            and str(config.provider) == "local"
            and str(config.adapter) in {"llama_cpp", "vllm"}
            and _active_parallelism() > 1
        )
        if shared_llama:
            # Take a native-server slot before the shared GPU lock. This keeps
            # scheduler-external callers out of llama-server's internal queue while
            # preserving writer preference for image/speech GPU handoff.
            with model_router_module._LLAMA_INFERENCE_SLOTS:
                with model_router_module._GPU_EXCLUSIVE_LOCK.shared():
                    return run_generation()
        with self._gpu_scope(config.exclusive_gpu):
            return run_generation()

    generate_text._mmm_llama_shared_slots = True  # type: ignore[attr-defined]
    generate_text._mmm_preserves_agent_tools = True  # type: ignore[attr-defined]

    cls.generation_session = generation_session
    cls.generate_text = generate_text


def _install_scheduler(scheduler_module: Any) -> None:
    current = scheduler_module._capacities
    if getattr(current, "_mmm_dynamic_llama_slots", False):
        return

    @wraps(current)
    def capacities() -> dict[str, int]:
        values = dict(current())
        values["llm"] = _active_parallelism()
        return values

    capacities._mmm_dynamic_llama_slots = True  # type: ignore[attr-defined]
    scheduler_module._capacities = capacities


def _install_planner_search_parallelism() -> None:
    # Agentic search owns candidate policy/scoring. This layer only maps independent
    # candidates onto the native server slots selected by autotuning.
    from . import agentic_optimization_contract as agentic_module
    from . import complete_planner as complete_planner_module

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_parallel_plan_search", False):
        return
    if not getattr(current, "_mmm_verifier_plan_search", False):
        return
    base = getattr(current, "__wrapped__", None)
    if not callable(base):
        return

    @wraps(current)
    def generate_with_parallel_search(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Any,
        expected_contracts: Any,
        stage: str,
    ) -> dict[str, Any]:
        width = agentic_module._planner_candidate_count(request, stage)
        parallel = _planner_parallel_capacity(router, width)
        if width <= 1 or parallel <= 1:
            return current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )

        def run_candidate(candidate_index: int):
            candidate_system = (
                system_prompt
                + "\n\nHOST SEARCH CANDIDATE: independently solve this page. Candidate "
                + str(candidate_index + 1)
                + " of "
                + str(width)
                + ". Preserve the exact contract; do not mention candidate search."
            )
            page = base(
                router,
                system_prompt=candidate_system,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )
            score, verifier = agentic_module._score_plan_page(page)
            return score, candidate_index, page, verifier

        candidates: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        errors: dict[int, BaseException] = {}
        with ThreadPoolExecutor(
            max_workers=parallel,
            thread_name_prefix="mmm_plan_search",
        ) as pool:
            futures = [pool.submit(run_candidate, index) for index in range(width)]
            # Consume in candidate order so failure selection and tie behavior remain
            # deterministic even though native decode runs concurrently.
            for candidate_index, future in enumerate(futures):
                try:
                    candidates.append(future.result())
                except BaseException as exc:
                    errors[candidate_index] = exc

        if not candidates:
            if errors:
                raise errors[max(errors)]
            raise complete_planner_module.SpecValidationError(
                f"{stage} produced no verified planning candidate."
            )
        candidates.sort(key=lambda item: (-item[0], item[1]))
        winner = candidates[0]
        print(
            "planner search:",
            f"stage={stage}",
            f"candidates={len(candidates)}",
            f"parallel={parallel}",
            f"winner={winner[1] + 1}",
            f"score={winner[0]:.3f}",
            flush=True,
        )
        return winner[2]

    generate_with_parallel_search._mmm_parallel_plan_search = True  # type: ignore[attr-defined]
    generate_with_parallel_search._mmm_verifier_plan_search = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_with_parallel_search


def install(model_router_module: Any, scheduler_module: Any) -> None:
    _install_router(model_router_module)
    _install_scheduler(scheduler_module)
    _install_planner_search_parallelism()


__all__ = [
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "_active_parallelism",
    "_planner_parallel_capacity",
    "install",
]
