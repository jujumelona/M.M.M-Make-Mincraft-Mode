from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class PlanValidationError(ValueError):
    """Raised when a semantic planning operation would corrupt host plan state."""


class PlanTerminal(str, Enum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    INFEASIBLE = "infeasible"
    FAILED = "failed"


class DeltaKind(str, Enum):
    ADD_TASK = "add_task"
    BLOCKED = "blocked"
    INFEASIBLE = "infeasible"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PlanTask:
    """One host-owned executable planning node.

    A task may cover zero requirements when it is a dependency introduced only to
    make another requirement executable. Dependencies must already exist when the
    task is added, which keeps incremental planning topological and makes every
    accepted delta immediately actionable when its dependencies are satisfied.
    """

    task_id: str
    intent: str
    action: Mapping[str, Any]
    covers: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()

    def normalized(self) -> "PlanTask":
        task_id = self.task_id.strip()
        intent = self.intent.strip()
        covers = tuple(dict.fromkeys(item.strip() for item in self.covers if item.strip()))
        depends_on = tuple(
            dict.fromkeys(item.strip() for item in self.depends_on if item.strip())
        )
        return PlanTask(
            task_id=task_id,
            intent=intent,
            action=dict(self.action),
            covers=covers,
            depends_on=depends_on,
        )


@dataclass(frozen=True, slots=True)
class SemanticDelta:
    """A single small semantic operation proposed by the model.

    There is intentionally no COMPLETE operation. Completion is a host-derived
    predicate over requirements, dependencies and executable task definitions.
    """

    kind: DeltaKind
    task: PlanTask | None = None
    reason: str = ""

    @classmethod
    def add_task(cls, task: PlanTask) -> "SemanticDelta":
        return cls(kind=DeltaKind.ADD_TASK, task=task)

    @classmethod
    def blocked(cls, reason: str) -> "SemanticDelta":
        return cls(kind=DeltaKind.BLOCKED, reason=reason)

    @classmethod
    def infeasible(cls, reason: str) -> "SemanticDelta":
        return cls(kind=DeltaKind.INFEASIBLE, reason=reason)


@dataclass(frozen=True, slots=True)
class DeltaResult:
    task_id: str | None
    newly_ready: tuple[PlanTask, ...]
    semantic_complete: bool
    terminal: PlanTerminal


@dataclass(slots=True)
class PlanState:
    """Authoritative host-side state for incremental semantic planning."""

    requirements: tuple[str, ...]
    tasks: dict[str, PlanTask] = field(default_factory=dict)
    task_status: dict[str, TaskStatus] = field(default_factory=dict)
    terminal: PlanTerminal = PlanTerminal.ACTIVE
    terminal_reason: str = ""

    def __post_init__(self) -> None:
        normalized = tuple(
            dict.fromkeys(item.strip() for item in self.requirements if item.strip())
        )
        if not normalized:
            raise PlanValidationError("plan requires at least one semantic requirement")
        self.requirements = normalized

    @property
    def covered_requirements(self) -> frozenset[str]:
        return frozenset(
            requirement
            for task in self.tasks.values()
            for requirement in task.covers
        )

    @property
    def uncovered_requirements(self) -> tuple[str, ...]:
        covered = self.covered_requirements
        return tuple(item for item in self.requirements if item not in covered)

    @property
    def dependency_valid(self) -> bool:
        return all(
            dependency in self.tasks and dependency != task.task_id
            for task in self.tasks.values()
            for dependency in task.depends_on
        ) and not self._contains_cycle()

    @property
    def executable_ready(self) -> bool:
        """Whether every planned node has a concrete executable definition."""

        return bool(self.tasks) and all(
            task.intent and bool(task.action) for task in self.tasks.values()
        )

    @property
    def semantic_complete(self) -> bool:
        """Host-derived plan completion; model output cannot directly set it."""

        return (
            self.terminal is PlanTerminal.ACTIVE
            and not self.uncovered_requirements
            and self.dependency_valid
            and self.executable_ready
        )

    def ready_tasks(self) -> tuple[PlanTask, ...]:
        if self.terminal is not PlanTerminal.ACTIVE:
            return ()
        return tuple(
            task
            for task_id, task in self.tasks.items()
            if self.task_status[task_id] is TaskStatus.PENDING
            and all(
                self.task_status[dependency] is TaskStatus.SUCCEEDED
                for dependency in task.depends_on
            )
        )

    def apply(self, delta: SemanticDelta) -> DeltaResult:
        """Validate and atomically apply one semantic model operation."""

        if self.terminal is not PlanTerminal.ACTIVE:
            raise PlanValidationError(
                f"cannot mutate terminal plan state: {self.terminal.value}"
            )

        ready_before = {task.task_id for task in self.ready_tasks()}

        if delta.kind is DeltaKind.ADD_TASK:
            task = self._validate_new_task(delta.task)
            self.tasks[task.task_id] = task
            self.task_status[task.task_id] = TaskStatus.PENDING
            task_id: str | None = task.task_id
        elif delta.kind in {DeltaKind.BLOCKED, DeltaKind.INFEASIBLE}:
            reason = delta.reason.strip()
            if not reason:
                raise PlanValidationError(f"{delta.kind.value} requires a reason")
            if delta.task is not None:
                raise PlanValidationError(f"{delta.kind.value} cannot include a task")
            self.terminal = (
                PlanTerminal.BLOCKED
                if delta.kind is DeltaKind.BLOCKED
                else PlanTerminal.INFEASIBLE
            )
            self.terminal_reason = reason
            task_id = None
        else:  # pragma: no cover - protects callers bypassing Enum construction.
            raise PlanValidationError(f"unsupported semantic delta: {delta.kind!r}")

        newly_ready = tuple(
            task for task in self.ready_tasks() if task.task_id not in ready_before
        )
        return DeltaResult(
            task_id=task_id,
            newly_ready=newly_ready,
            semantic_complete=self.semantic_complete,
            terminal=self.terminal,
        )

    def mark_running(self, task_id: str) -> None:
        self._require_status(task_id, TaskStatus.PENDING)
        if task_id not in {task.task_id for task in self.ready_tasks()}:
            raise PlanValidationError(f"task is not dependency-ready: {task_id}")
        self.task_status[task_id] = TaskStatus.RUNNING

    def mark_succeeded(self, task_id: str) -> tuple[PlanTask, ...]:
        """Record success and publish nodes unblocked by this exact transition."""

        before = {task.task_id for task in self.ready_tasks()}
        current = self.task_status.get(task_id)
        if current not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            raise PlanValidationError(
                f"task cannot succeed from {current.value if current else 'unknown'}: {task_id}"
            )
        if current is TaskStatus.PENDING and task_id not in before:
            raise PlanValidationError(f"task is not dependency-ready: {task_id}")
        self.task_status[task_id] = TaskStatus.SUCCEEDED
        return tuple(task for task in self.ready_tasks() if task.task_id not in before)

    def mark_failed(self, task_id: str, reason: str) -> None:
        current = self.task_status.get(task_id)
        if current not in {TaskStatus.PENDING, TaskStatus.RUNNING}:
            raise PlanValidationError(
                f"task cannot fail from {current.value if current else 'unknown'}: {task_id}"
            )
        detail = reason.strip()
        if not detail:
            raise PlanValidationError("failed task requires a reason")
        self.task_status[task_id] = TaskStatus.FAILED
        self.terminal = PlanTerminal.FAILED
        self.terminal_reason = f"{task_id}: {detail}"

    def _validate_new_task(self, task: PlanTask | None) -> PlanTask:
        if task is None:
            raise PlanValidationError("add_task requires a task")
        normalized = task.normalized()
        if not normalized.task_id:
            raise PlanValidationError("task_id cannot be empty")
        if normalized.task_id in self.tasks:
            raise PlanValidationError(f"duplicate task_id: {normalized.task_id}")
        if not normalized.intent:
            raise PlanValidationError(f"task intent cannot be empty: {normalized.task_id}")
        if not normalized.action:
            raise PlanValidationError(f"task action cannot be empty: {normalized.task_id}")
        unknown_coverage = set(normalized.covers).difference(self.requirements)
        if unknown_coverage:
            raise PlanValidationError(
                "task covers unknown requirements: " + ", ".join(sorted(unknown_coverage))
            )
        unknown_dependencies = set(normalized.depends_on).difference(self.tasks)
        if unknown_dependencies:
            raise PlanValidationError(
                "dependencies must already exist: "
                + ", ".join(sorted(unknown_dependencies))
            )
        if normalized.task_id in normalized.depends_on:
            raise PlanValidationError(f"task cannot depend on itself: {normalized.task_id}")
        return normalized

    def _require_status(self, task_id: str, expected: TaskStatus) -> None:
        current = self.task_status.get(task_id)
        if current is not expected:
            raise PlanValidationError(
                f"task {task_id} must be {expected.value}, got "
                f"{current.value if current else 'unknown'}"
            )

    def _contains_cycle(self) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> bool:
            if task_id in visited:
                return False
            if task_id in visiting:
                return True
            visiting.add(task_id)
            for dependency in self.tasks[task_id].depends_on:
                if dependency in self.tasks and visit(dependency):
                    return True
            visiting.remove(task_id)
            visited.add(task_id)
            return False

        return any(visit(task_id) for task_id in self.tasks)
