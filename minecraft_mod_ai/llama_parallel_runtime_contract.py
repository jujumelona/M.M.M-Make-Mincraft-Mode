from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Mapping

_ROUTER_CONTRACT_VERSION = 3
_RESEARCH_DESIGN_CAPACITY_VERSION = 1


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


def _install_router(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    installed_version = int(
        getattr(cls.generate_text, "_mmm_parallel_router_contract_version", 0) or 0
    )
    if installed_version >= _ROUTER_CONTRACT_VERSION:
        return

    model_router_module._GPU_EXCLUSIVE_LOCK = ReentrantReadWriteLock()
    model_router_module._LLAMA_INFERENCE_SLOTS = ReentrantCapacityGate(_active_parallelism)

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
        response_schema: Mapping[str, Any] | None = None,
        tool_stage: str | None = None,
        enable_tools: bool = True,
    ) -> str:
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

        stage, runtime, tools, request = self._prepare_generation_request(
            role,
            messages,
            config=config,
            media_paths=media_paths,
            response_format=response_format,
            response_schema=response_schema,
            tool_stage=tool_stage,
            enable_tools=enable_tools,
        )

        def run_generation() -> str:
            if runtime is not None and tools:
                return self._generate_with_tools(
                    adapter=adapter,
                    request=request,
                    runtime=runtime,
                    stage=stage,
                    role=role,
                )
            return adapter.generate(request)

        shared_llama = (
            bool(config.exclusive_gpu)
            and str(config.provider) == "local"
            and str(config.adapter) in {"llama_cpp", "vllm"}
            and _active_parallelism() > 1
        )
        if shared_llama:
            with model_router_module._LLAMA_INFERENCE_SLOTS:
                with model_router_module._GPU_EXCLUSIVE_LOCK.shared():
                    return run_generation()
        with self._gpu_scope(config.exclusive_gpu):
            return run_generation()

    generate_text._mmm_llama_shared_slots = True  # type: ignore[attr-defined]
    generate_text._mmm_preserves_agent_tools = True  # type: ignore[attr-defined]
    generate_text._mmm_preserves_response_schema = True  # type: ignore[attr-defined]
    generate_text._mmm_uses_canonical_request_preparation = True  # type: ignore[attr-defined]
    generate_text._mmm_parallel_router_contract_version = _ROUTER_CONTRACT_VERSION  # type: ignore[attr-defined]
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


def _install_research_design_capacity_policy(model_router_module: Any) -> None:
    """Apply managed-runtime limits only to routers that own the llama process."""
    from . import central_intelligence_amplifier as central_module

    current = central_module._research_domain_worker_count
    installed_version = int(
        getattr(current, "_mmm_managed_research_capacity_version", 0) or 0
    )
    if installed_version >= _RESEARCH_DESIGN_CAPACITY_VERSION:
        return

    @wraps(current)
    def research_design_capacity(router: Any, width: int) -> int:
        requested = min(max(1, int(width)), central_module._worker_count())
        if isinstance(router, model_router_module.ModelRouter):
            return current(router, width)
        try:
            config = router.registry.role(router.profile, "planner")
        except Exception:
            return requested
        if not bool(getattr(config, "exclusive_gpu", False)):
            return 1
        if str(getattr(config, "provider", "")) != "local":
            return 1
        if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
            return 1
        return requested

    research_design_capacity._mmm_managed_research_capacity_version = (  # type: ignore[attr-defined]
        _RESEARCH_DESIGN_CAPACITY_VERSION
    )
    research_design_capacity.__wrapped__ = current  # type: ignore[attr-defined]
    central_module._research_domain_worker_count = research_design_capacity


def install(model_router_module: Any, scheduler_module: Any) -> None:
    """Install runtime slot sharing; Planner search is not a runtime concern."""
    _install_router(model_router_module)
    _install_scheduler(scheduler_module)
    _install_research_design_capacity_policy(model_router_module)


__all__ = [
    "ReentrantCapacityGate",
    "ReentrantReadWriteLock",
    "_active_parallelism",
    "install",
]
