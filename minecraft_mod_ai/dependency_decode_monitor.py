from __future__ import annotations

"""Decode-time dependency admission for the production coder.

This module strengthens the repository research engine's finite dependency monitor at
its actual generation boundary. The monitor is active only while a research-grounded
coder turn is running. Local llama-server output is checked incrementally, so an
unknown dependency coordinate or Java import aborts the stream as soon as the token
sequence is complete; the research router then retries with the exact violation fed
back as host evidence. Non-streaming backends retain the same final patch admission
check through the enhanced DependencyMonitor.
"""

import json
import re
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterable, Mapping, Sequence


_ACTIVE_MONITOR: ContextVar[Any | None] = ContextVar(
    "mmm_active_dependency_decode_monitor",
    default=None,
)
_STREAM_TEXT: ContextVar[str] = ContextVar(
    "mmm_dependency_decode_stream_text",
    default="",
)

# A coordinate is considered complete only when the current stream exposes a
# terminator after the version. If it ends exactly at the latest chunk boundary we
# defer admission, preventing false positives on versions split across SSE deltas.
_STREAM_COORD = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+):([A-Za-z0-9][A-Za-z0-9.+_-]*)"
)
_STREAM_IMPORT = re.compile(
    r"\bimport\s+(?:static\s+)?"
    r"([A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$*][A-Za-z0-9_$*]*)+)\s*;"
)
_STREAM_URL = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")
_COORD_TERMINATORS = frozenset("\"'\\\n\r\t ),;}]")
_JAVA_ROOT = re.compile(
    r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$"
)

# These namespaces are owned by the selected Java/Minecraft/Fabric toolchain, not
# inferred from the language model. Fabric is deliberately narrow: arbitrary
# ``net.fabricmc.*`` is not admitted.
_PLATFORM_IMPORT_PREFIXES = (
    "java.",
    "javax.",
    "jdk.",
    "net.minecraft.",
    "com.mojang.",
    "org.jetbrains.annotations.",
    "net.fabricmc.api.",
    "net.fabricmc.loader.api.",
    "net.fabricmc.fabric.api.",
)


class DependencyDecodeAdmissionError(RuntimeError):
    """Raised when completed generated tokens leave the finite admission set."""

    def __init__(self, violations: Sequence[Any]) -> None:
        self.violations = tuple(violations)
        rendered = ", ".join(
            f"{getattr(item, 'kind', 'dependency')}={getattr(item, 'value', item)}"
            for item in self.violations
        )
        super().__init__(
            "PackMonitor blocked generated dependency/package tokens outside the "
            f"authoritative admission set: {rendered}"
        )


def _java_package_prefix(import_name: str) -> str:
    value = str(import_name).strip().removesuffix(".*")
    if not value or "." not in value:
        return ""
    parent = value.rsplit(".", 1)[0]
    return parent + "." if _JAVA_ROOT.fullmatch(parent) else ""


def _maven_group_prefix(group: str) -> str:
    value = str(group).strip()
    return value + "." if _JAVA_ROOT.fullmatch(value) else ""


def _operation_text(research: Any, raw: Mapping[str, Any]) -> str:
    helper = getattr(research, "_operation_text", None)
    if callable(helper):
        return str(helper(raw))
    content = raw.get("content")
    if isinstance(content, str):
        return content
    replacements = raw.get("replacements")
    if not isinstance(replacements, list):
        return ""
    return "\n".join(
        str(item.get("new", ""))
        for item in replacements
        if isinstance(item, Mapping)
    )


def _extract_operations(research: Any, text: str) -> list[Mapping[str, Any]]:
    helper = getattr(research, "_extract_json_object", None)
    payload = helper(text) if callable(helper) else None
    if not isinstance(payload, Mapping):
        try:
            value = json.loads(text)
        except Exception:
            return []
        payload = value if isinstance(value, Mapping) else None
    if not isinstance(payload, Mapping):
        return []
    for key in ("operations", "patch_operations", "patches"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _unique(violations: Iterable[Any]) -> tuple[Any, ...]:
    by_key: dict[tuple[str, str, str], Any] = {}
    for item in violations:
        key = (
            str(getattr(item, "kind", "")),
            str(getattr(item, "value", "")),
            str(getattr(item, "path", "")),
        )
        by_key[key] = item
    return tuple(by_key[key] for key in sorted(by_key))


def _install_enhanced_monitor() -> None:
    from . import research_code_context as research

    base = research.DependencyMonitor
    if getattr(base, "_mmm_decode_time_packmonitor", False):
        return

    class DecodeTimeDependencyMonitor(base):
        _mmm_decode_time_packmonitor = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.allowed_import_prefixes: set[str] = set(_PLATFORM_IMPORT_PREFIXES)
            self.project_package_prefixes: set[str] = set()
            super().__init__(*args, **kwargs)
            self._seed_java_source()
            self._admit_maven_group_prefixes()

        def _admit_maven_group_prefixes(self) -> None:
            for coordinate in tuple(self.allowed_coordinates):
                group = coordinate.split(":", 1)[0]
                prefix = _maven_group_prefix(group)
                if prefix:
                    self.allowed_import_prefixes.add(prefix)

        def _seed_java_source(self) -> None:
            for source_root in (
                self.root / "src/main/java",
                self.root / "src/test/java",
                self.root / "src/gametest",
            ):
                if not source_root.is_dir() or source_root.is_symlink():
                    continue
                for path in source_root.rglob("*.java"):
                    if not path.is_file() or path.is_symlink():
                        continue
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    self._admit_java_text(text)

        def _admit_java_text(self, text: str) -> None:
            package_re = getattr(research, "_PACKAGE")
            import_re = getattr(research, "_IMPORT")
            for package in package_re.findall(text):
                value = str(package).strip()
                if _JAVA_ROOT.fullmatch(value):
                    prefix = value + "."
                    self.project_package_prefixes.add(prefix)
                    self.allowed_import_prefixes.add(prefix)
            for imported in import_re.findall(text):
                prefix = _java_package_prefix(str(imported))
                if prefix:
                    self.allowed_import_prefixes.add(prefix)

        def admit_text(self, text: str, *, code_owned: bool) -> None:
            super().admit_text(text, code_owned=code_owned)
            if not code_owned:
                return
            self._admit_java_text(text)
            maven_re = getattr(research, "_MAVEN_COORD")
            for group, _artifact, _version in maven_re.findall(text):
                prefix = _maven_group_prefix(group)
                if prefix:
                    self.allowed_import_prefixes.add(prefix)

        def _import_allowed(self, imported: str) -> bool:
            value = str(imported).strip().removesuffix(".*")
            return bool(value) and any(
                value == prefix[:-1] or value.startswith(prefix)
                for prefix in self.allowed_import_prefixes
            )

        def _java_violations(self, text: str, *, path: str) -> tuple[Any, ...]:
            violation_cls = research.DependencyViolation
            result: list[Any] = []
            for imported in getattr(research, "_IMPORT").findall(text):
                value = str(imported).strip()
                if value and not self._import_allowed(value):
                    result.append(violation_cls("java_import", value, path))
            return _unique(result)

        def validate_model_output(self, text: str) -> tuple[Any, ...]:
            violations = list(super().validate_model_output(text))
            for raw in _extract_operations(research, text):
                path = str(raw.get("path", "")).replace("\\", "/")
                if not path.casefold().endswith(".java"):
                    continue
                violations.extend(
                    self._java_violations(_operation_text(research, raw), path=path)
                )
            return _unique(violations)

        def stream_violations(self, text: str, *, final: bool) -> tuple[Any, ...]:
            """Check only token sequences complete at this stream boundary."""

            violation_cls = research.DependencyViolation
            result: list[Any] = []
            for match in _STREAM_COORD.finditer(text):
                if match.end() == len(text) and not final:
                    continue
                if (
                    match.end() < len(text)
                    and text[match.end()] not in _COORD_TERMINATORS
                ):
                    continue
                group, artifact, version = match.groups()
                package = f"{group}:{artifact}"
                coordinate = f"{package}:{version}"
                if package not in self.allowed_packages:
                    result.append(
                        violation_cls("package", package, "<decode-stream>")
                    )
                elif coordinate not in self.allowed_coordinates:
                    result.append(
                        violation_cls("coordinate", coordinate, "<decode-stream>")
                    )
            for match in _STREAM_IMPORT.finditer(text):
                imported = match.group(1)
                if not self._import_allowed(imported):
                    result.append(
                        violation_cls("java_import", imported, "<decode-stream>")
                    )
            for match in _STREAM_URL.finditer(text):
                if match.end() == len(text) and not final:
                    continue
                value = match.group(0).rstrip("\\\"'.,;)")
                is_dependency_url = (
                    "maven" in value.casefold() or "repo" in value.casefold()
                )
                if is_dependency_url and value not in self.allowed_repositories:
                    result.append(
                        violation_cls("repository", value, "<decode-stream>")
                    )
            return _unique(result)

        def enforce_stream(self, text: str, *, final: bool) -> None:
            violations = self.stream_violations(text, final=final)
            if violations:
                raise DependencyDecodeAdmissionError(violations)

        def receipt(self) -> dict[str, Any]:
            payload = dict(super().receipt())
            payload.update(
                {
                    "schema_version": "mmm/dependency-monitor-v3",
                    "policy": "finite_authoritative_decode_time_package_admission",
                    "allowed_import_prefix_count": len(self.allowed_import_prefixes),
                    "allowed_import_prefixes_sha256": research._sha(
                        sorted(self.allowed_import_prefixes)
                    ),
                    "decode_time_monitoring": True,
                    "java_import_admission": True,
                    "partial_token_deferral": True,
                }
            )
            return payload

    research.DependencyMonitor = DecodeTimeDependencyMonitor


def active_dependency_monitor() -> Any | None:
    return _ACTIVE_MONITOR.get()


def _install_research_router_scope() -> None:
    from . import custom_generation_search_contract as generation

    cls = generation._ResearchEvidenceRouter
    original = cls.generate_text
    if getattr(original, "_mmm_dependency_decode_scope", False):
        return

    @wraps(original)
    def generate_text(
        self: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        if role != "coder":
            return original(self, role, messages, **kwargs)
        engine = self._engine()
        seen: set[tuple[tuple[str, str, str], ...]] = set()
        current = [dict(message) for message in messages]
        while True:
            token = _ACTIVE_MONITOR.set(engine.monitor)
            try:
                return original(self, role, current, **kwargs)
            except DependencyDecodeAdmissionError as exc:
                key = tuple(
                    sorted(
                        (
                            str(getattr(item, "kind", "")),
                            str(getattr(item, "value", "")),
                            str(getattr(item, "path", "")),
                        )
                        for item in exc.violations
                    )
                )
                if key in seen or len(seen) >= 3:
                    raise RuntimeError(
                        "Decode-time dependency admission made no progress after "
                        "research-grounded correction."
                    ) from exc
                seen.add(key)
                current = [
                    *current,
                    {
                        "role": "system",
                        "content": (
                            "PackMonitor stopped the previous decode before completion. "
                            "Regenerate from the beginning using only the authoritative "
                            "dependency/import admission set in research_code_context. "
                            "Blocked values: "
                            + json.dumps(
                                [
                                    {
                                        "kind": getattr(item, "kind", ""),
                                        "value": getattr(item, "value", ""),
                                    }
                                    for item in exc.violations
                                ],
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        ),
                    },
                ]
            finally:
                _ACTIVE_MONITOR.reset(token)

    generate_text._mmm_dependency_decode_scope = True
    cls.generate_text = generate_text


def _install_llama_stream_hook() -> None:
    from . import llama_server_hardware_policy as hardware

    original_strict = hardware._strict_server_generate
    original_delta = hardware._stream_delta_parts
    if getattr(original_strict, "_mmm_dependency_decode_monitor", False):
        return

    @wraps(original_delta)
    def stream_delta_parts(choice: dict[str, Any]) -> tuple[str, str]:
        reasoning, content = original_delta(choice)
        monitor = active_dependency_monitor()
        if monitor is not None and content:
            cumulative = _STREAM_TEXT.get() + content
            _STREAM_TEXT.set(cumulative)
            monitor.enforce_stream(cumulative, final=False)
        return reasoning, content

    @wraps(original_strict)
    def strict_server_generate(adapter: Any, request: Any, server_url: str) -> str:
        token = _STREAM_TEXT.set("")
        try:
            try:
                result = original_strict(adapter, request, server_url)
            except Exception as exc:
                # Native hardware policy wraps stream failures in ModelBackendError.
                # Preserve the PackMonitor signal for the research correction loop.
                cause = getattr(exc, "cause", None)
                if isinstance(cause, DependencyDecodeAdmissionError):
                    raise cause from exc
                raise
            monitor = active_dependency_monitor()
            if monitor is not None:
                monitor.enforce_stream(result, final=True)
            return result
        finally:
            _STREAM_TEXT.reset(token)

    stream_delta_parts._mmm_dependency_decode_monitor = True
    strict_server_generate._mmm_dependency_decode_monitor = True
    hardware._stream_delta_parts = stream_delta_parts
    hardware._strict_server_generate = strict_server_generate


def activate_dependency_decode_monitor() -> None:
    """Install the PackMonitor boundary exactly once."""

    _install_enhanced_monitor()
    _install_research_router_scope()
    _install_llama_stream_hook()


__all__ = [
    "DependencyDecodeAdmissionError",
    "activate_dependency_decode_monitor",
    "active_dependency_monitor",
]
