"""Typed JSON resource boundary; unknown extension fields remain lossless.

This is deliberately not a complete Fabric schema/version validator. It validates
the object and entrypoint shapes that our editors operate on, while preserving
other loader and third-party fields.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, TypeAlias
from urllib.parse import urlparse

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
ResourceKind = Literal["json", "config", "lang", "fabric"]


class ResourceValidationError(ValueError):
    """The input cannot represent an unambiguous resource of the requested kind."""


def _object_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ResourceValidationError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ResourceValidationError(f"Nonfinite JSON number: {value}")


def _validate_json(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResourceValidationError("JSON object keys must be strings.")
            _validate_json(item)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ResourceValidationError("JSON numbers must be finite.")
    elif value is not None and not isinstance(value, (str, bool, int)):
        raise ResourceValidationError(f"Unsupported JSON value: {type(value).__name__}")


def _require_object(value: JsonValue, kind: ResourceKind) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ResourceValidationError(f"{kind} resource must be an object.")
    return value


def _validate_language(value: dict[str, JsonValue]) -> None:
    if any(not isinstance(item, str) for item in value.values()):
        raise ResourceValidationError("Language resource values must be strings.")


def _valid_fabric_entry(entry: JsonValue) -> bool:
    if isinstance(entry, str):
        return bool(entry)
    if not isinstance(entry, dict):
        return False
    value = entry.get("value")
    adapter = entry.get("adapter")
    return isinstance(value, str) and bool(value) and (
        "adapter" not in entry or isinstance(adapter, str)
    )


def _validate_fabric(value: dict[str, JsonValue]) -> None:
    entrypoints = value.get("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise ResourceValidationError("Fabric entrypoints must be an object.")
    for entries in entrypoints.values():
        if not isinstance(entries, list):
            raise ResourceValidationError("Fabric entrypoints must be lists.")
        if any(not _valid_fabric_entry(entry) for entry in entries):
            raise ResourceValidationError("Invalid Fabric entrypoint.")


@dataclass(frozen=True)
class JsonResource:
    value: JsonValue
    kind: ResourceKind = "json"

    @classmethod
    def parse(cls, text: str, *, kind: ResourceKind = "json") -> JsonResource:
        try:
            value = json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise ResourceValidationError(f"Invalid JSON resource: {exc}") from exc
        resource = cls(value, kind)
        resource.validate()
        return resource

    def validate(self) -> None:
        _validate_json(self.value)
        if self.kind not in ("fabric", "lang", "config"):
            return
        value = _require_object(self.value, self.kind)
        if self.kind == "lang":
            _validate_language(value)
        elif self.kind == "fabric":
            _validate_fabric(value)

    def serialize(self) -> str:
        # Validate again: callers may have edited nested containers since parsing.
        self.validate()
        return json.dumps(self.value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def resource_kind(path: str) -> ResourceKind:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if parts and parts[-1] == "fabric.mod.json":
        return "fabric"
    if "assets" in parts and len(parts) >= 2 and parts[-2] == "lang":
        return "lang"
    if "config" in parts:
        return "config"
    return "json"


@dataclass(frozen=True)
class GradleDependency:
    configuration: str
    group: str
    artifact: str
    version: str
    repository_name: str
    repository_url: str

    @classmethod
    def parse_legacy(cls, repository: str, dependency: str) -> GradleDependency:
        # This compatibility grammar accepts literal Maven declarations only.
        dep = re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\(\s*["\']([\w.-]+):([\w.-]+):([\w.+-]+)["\']\s*\)\s*', dependency)
        repo = re.fullmatch(r'\s*maven\s*\{\s*name\s*=\s*["\']([\w.-]+)["\']\s*url\s*=\s*["\']([^"\'\s$]+)["\']\s*\}\s*', repository)
        if dep is None or repo is None:
            raise ResourceValidationError('Dependency requires literal module coordinates and Maven repository')
        url = urlparse(repo[2])
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ResourceValidationError('Maven repository must be an HTTPS URL without credentials')
        return cls(*dep.groups(), *repo.groups())

    def serialize(self) -> str:
        return ('repositories {\n    maven {\n        name = ' + json.dumps(self.repository_name)
                + '\n        url = uri(' + json.dumps(self.repository_url) + ')\n    }\n}\n'
                + 'dependencies {\n    ' + self.configuration + '(' + json.dumps(
                    f'{self.group}:{self.artifact}:{self.version}') + ')\n}\n')


def _skip_line_comment(text: str, index: int) -> int | None:
    if not text.startswith('//', index):
        return None
    end = text.find('\n', index)
    return len(text) if end < 0 else end + 1


def _skip_block_comment(text: str, index: int) -> int | None:
    if not text.startswith('/*', index):
        return None
    end = text.find('*/', index + 2)
    if end < 0:
        raise ResourceValidationError('Unterminated Gradle comment')
    return end + 2


def _skip_quoted_string(text: str, index: int) -> int | None:
    char = text[index]
    if char not in {'"', "'"}:
        return None
    quote = char * 3 if text.startswith(char * 3, index) else char
    cursor = index + len(quote)
    while cursor < len(text) and not text.startswith(quote, cursor):
        cursor += 2 if text[cursor] == '\\' else 1
    if cursor >= len(text):
        raise ResourceValidationError('Unterminated Gradle string')
    return cursor + len(quote)


def _update_delimiter_stack(stack: list[str], char: str) -> None:
    if char in '{[(':
        stack.append(char)
        return
    closing = {'}': '{', ']': '[', ')': '('}
    expected = closing.get(char)
    if expected is not None and (not stack or stack.pop() != expected):
        raise ResourceValidationError('Unbalanced Gradle delimiters')


def validate_gradle_append(text: str) -> None:
    """Reject unterminated lexical constructs before appending a top-level script."""
    stack: list[str] = []
    index = 0
    while index < len(text):
        next_index = _skip_line_comment(text, index)
        if next_index is None:
            next_index = _skip_block_comment(text, index)
        if next_index is None:
            next_index = _skip_quoted_string(text, index)
        if next_index is not None:
            index = next_index
            continue
        _update_delimiter_stack(stack, text[index])
        index += 1
    if stack:
        raise ResourceValidationError('Unclosed Gradle block')