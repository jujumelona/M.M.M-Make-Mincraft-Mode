from __future__ import annotations

import hashlib
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

_DEFAULT_MAX_RENDERED_GROUPS = 20
_DEFAULT_TEXT_LIMIT = 1600
_MAX_RENDERED_FALLBACKS = 8


class FailureCategory(str, Enum):
    """Stable operator-facing failure classes.

    Categories describe why a boundary failed, not which fallback happened next.
    Callers should classify at the boundary where enough context exists rather than
    guessing from message text.
    """

    INPUT = "INPUT"
    DEPENDENCY = "DEPENDENCY"
    TRANSIENT = "TRANSIENT"
    VALIDATION = "VALIDATION"
    INTERNAL = "INTERNAL"


class FailureStatus(str, Enum):
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class FailureEvent:
    stage: str
    operation: str
    category: FailureCategory
    cause_type: str
    cause: str
    retryable: bool
    final_status: FailureStatus
    fallback: str | None = None
    affected_artifact: str | None = None
    debug_traceback: str | None = None
    deduplication_key: str | None = None

    @property
    def fingerprint(self) -> str:
        """Return a stable causal fingerprint without guessing from message text.

        By default the full cause participates in the fingerprint. Boundaries that
        know repeated attempts can produce varying human-readable messages may pass
        an explicit ``deduplication_key`` instead.
        """

        payload = "\x1f".join(
            (
                self.stage,
                self.operation,
                self.category.value,
                self.cause_type,
                self.deduplication_key if self.deduplication_key is not None else self.cause,
                self.affected_artifact or "",
            )
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["category"] = self.category.value
        value["final_status"] = self.final_status.value
        value["fingerprint"] = self.fingerprint
        if not include_debug:
            value.pop("debug_traceback", None)
        if self.deduplication_key is None:
            value.pop("deduplication_key", None)
        return value


@dataclass
class FailureGroup:
    event: FailureEvent
    attempts: int = 1
    fallbacks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.event.fallback:
            self.fallbacks.append(self.event.fallback)

    def add(self, event: FailureEvent) -> None:
        if event.fingerprint != self.event.fingerprint:
            raise ValueError("cannot merge different failure fingerprints")
        self.attempts += 1
        if event.fallback and event.fallback not in self.fallbacks:
            self.fallbacks.append(event.fallback)
        # Preserve the first causal description, but make terminal/retry state reflect
        # the latest attempt. Keep the latest traceback when one is available.
        self.event = FailureEvent(
            stage=self.event.stage,
            operation=self.event.operation,
            category=self.event.category,
            cause_type=self.event.cause_type,
            cause=self.event.cause,
            retryable=event.retryable,
            final_status=event.final_status,
            fallback=event.fallback or self.event.fallback,
            affected_artifact=self.event.affected_artifact,
            debug_traceback=event.debug_traceback or self.event.debug_traceback,
            deduplication_key=self.event.deduplication_key,
        )

    def to_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        value = self.event.to_dict(include_debug=include_debug)
        value["attempts"] = self.attempts
        value["fallbacks"] = list(self.fallbacks)
        return value


class DiagnosticCollector:
    """Deduplicate repeated symptoms while retaining one causal diagnostic."""

    def __init__(self) -> None:
        self._groups: dict[str, FailureGroup] = {}
        self._order: list[str] = []

    def clear(self) -> None:
        self._groups.clear()
        self._order.clear()

    def record(self, event: FailureEvent) -> FailureGroup:
        fingerprint = event.fingerprint
        group = self._groups.get(fingerprint)
        if group is None:
            group = FailureGroup(event=event)
            self._groups[fingerprint] = group
            self._order.append(fingerprint)
        else:
            group.add(event)
        return group

    def record_exception(
        self,
        exc: BaseException,
        *,
        stage: str,
        operation: str,
        category: FailureCategory,
        retryable: bool,
        final_status: FailureStatus,
        fallback: str | None = None,
        affected_artifact: str | None = None,
        include_debug_traceback: bool | None = None,
        sanitize: Callable[[str], str] | None = None,
        deduplication_key: str | None = None,
    ) -> FailureGroup:
        clean = sanitize or (lambda value: value)
        cause = clean(str(exc))
        capture_debug = (
            category is FailureCategory.INTERNAL
            if include_debug_traceback is None
            else include_debug_traceback
        )
        debug = None
        if capture_debug:
            debug = clean("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        return self.record(
            FailureEvent(
                stage=stage,
                operation=operation,
                category=category,
                cause_type=type(exc).__name__,
                cause=cause,
                retryable=retryable,
                final_status=final_status,
                fallback=fallback,
                affected_artifact=affected_artifact,
                debug_traceback=debug,
                deduplication_key=(
                    clean(deduplication_key) if deduplication_key is not None else None
                ),
            )
        )

    def groups(self) -> tuple[FailureGroup, ...]:
        return tuple(self._groups[key] for key in self._order)

    def to_dicts(self, *, include_debug: bool = False) -> list[dict[str, Any]]:
        return [group.to_dict(include_debug=include_debug) for group in self.groups()]


def _compact_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value).split())
    if not text:
        return "none"
    if len(text) <= limit:
        return text
    if limit <= 1:
        return text[:limit]
    return text[: limit - 1] + "…"


def _render_fallbacks(fallbacks: list[str], *, text_limit: int) -> str:
    if not fallbacks:
        return "none"
    shown = [
        _compact_text(item, limit=min(text_limit, 400))
        for item in fallbacks[:_MAX_RENDERED_FALLBACKS]
    ]
    omitted = len(fallbacks) - len(shown)
    if omitted:
        shown.append(f"+{omitted} more")
    return ", ".join(shown)


def render_failure_summary(
    groups: Iterable[FailureGroup],
    *,
    max_groups: int = _DEFAULT_MAX_RENDERED_GROUPS,
    text_limit: int = _DEFAULT_TEXT_LIMIT,
) -> str:
    """Render bounded root-cause diagnostics without dumping tracebacks.

    Full diagnostic evidence remains available through ``FailureGroup.to_dict``.
    The compact representation intentionally caps distinct roots and line lengths so
    one pathological exception cannot recreate CI/user-facing log floods.
    """

    if max_groups < 1:
        raise ValueError("max_groups must be at least 1")
    if text_limit < 32:
        raise ValueError("text_limit must be at least 32")

    items = tuple(groups)
    if not items:
        return "FINAL STATUS\nPASS"

    lines: list[str] = []
    shown_items = items[:max_groups]
    for index, group in enumerate(shown_items, start=1):
        event = group.event
        if index > 1:
            lines.append("")
        lines.extend(
            (
                f"ROOT FAILURE {index}",
                (
                    f"{_compact_text(event.stage, limit=240)} / "
                    f"{_compact_text(event.operation, limit=400)} "
                    f"[{event.category.value}]"
                ),
                "CAUSE",
                (
                    f"{_compact_text(event.cause_type, limit=160)}: "
                    f"{_compact_text(event.cause, limit=text_limit)}"
                ),
                "ATTEMPTS",
                str(group.attempts),
                "FALLBACK",
                _render_fallbacks(group.fallbacks, text_limit=text_limit),
                "FINAL STATUS",
                event.final_status.value,
            )
        )

    omitted = len(items) - len(shown_items)
    if omitted:
        lines.extend(
            (
                "",
                "ROOT FAILURES OMITTED",
                f"{omitted} additional distinct root failures are retained in debug evidence",
            )
        )
    return "\n".join(lines)
