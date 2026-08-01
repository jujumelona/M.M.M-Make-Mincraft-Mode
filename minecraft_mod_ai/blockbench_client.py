from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_FACES = frozenset({"north", "south", "east", "west", "up", "down"})
_INTERPOLATIONS = frozenset({"linear", "step", "catmullrom", "bezier"})

_ALLOWED = frozenset(
    {
        "open_project",
        "create_cube",
        "set_texture",
        "set_uv",
        "create_animation",
        "render_preview",
        "validate_uv",
        "export_bbmodel",
        "export_geckolib",
        "close_project",
    }
)


class BlockbenchMCPError(RuntimeError):
    pass


def _closed_arguments(
    operation: str,
    arguments: dict[str, Any],
    *,
    required: frozenset[str] = frozenset(),
    optional: frozenset[str] = frozenset(),
) -> None:
    keys = set(arguments)
    if not all(isinstance(key, str) for key in keys):
        raise BlockbenchMCPError(
            f"Blockbench {operation!r} argument names must be strings."
        )
    unknown = keys - required - optional
    if unknown:
        raise BlockbenchMCPError(
            f"Blockbench {operation!r} received unsupported arguments: "
            + ", ".join(sorted(unknown))
        )
    missing = required - keys
    if missing:
        raise BlockbenchMCPError(
            f"Blockbench {operation!r} is missing required arguments: "
            + ", ".join(sorted(missing))
        )


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise BlockbenchMCPError(f"Blockbench argument {field!r} must be a string.")
    if "\x00" in value:
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} must not contain a NUL byte."
        )
    if not allow_empty and not value.strip():
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} must not be empty."
        )
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise BlockbenchMCPError(f"Blockbench argument {field!r} must be boolean.")
    return value


def _finite_number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlockbenchMCPError(f"Blockbench argument {field!r} must be numeric.")
    if not math.isfinite(float(value)):
        raise BlockbenchMCPError(f"Blockbench argument {field!r} must be finite.")
    return value


def _positive_integer(
    value: Any,
    field: str,
    *,
    minimum: int = 1,
    maximum: int = 16_384,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BlockbenchMCPError(f"Blockbench argument {field!r} must be an integer.")
    if not minimum <= value <= maximum:
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} must be between {minimum} and {maximum}."
        )
    return value


def _vector(value: Any, field: str, *, lengths: tuple[int, ...] = (3,)) -> None:
    if not isinstance(value, (list, tuple)) or len(value) not in lengths:
        expected = " or ".join(str(length) for length in lengths)
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} must contain exactly {expected} numbers."
        )
    for index, item in enumerate(value):
        _finite_number(item, f"{field}[{index}]")


def _faces(value: Any, field: str) -> None:
    if not isinstance(value, (list, tuple)) or not value:
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} must be a non-empty face list."
        )
    invalid = [
        face for face in value if not isinstance(face, str) or face not in _FACES
    ]
    if invalid:
        raise BlockbenchMCPError(
            f"Blockbench argument {field!r} contains an unsupported face."
        )


def _validate_open_project(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "open_project",
        arguments,
        required=frozenset({"path"}),
        optional=frozenset({"read_only"}),
    )
    _string(arguments["path"], "path")
    if "read_only" in arguments:
        _boolean(arguments["read_only"], "read_only")


def _validate_create_cube(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "create_cube",
        arguments,
        required=frozenset({"name", "from", "to"}),
        optional=frozenset(
            {
                "origin",
                "rotation",
                "inflate",
                "mirror",
                "texture",
                "uv",
                "parent",
            }
        ),
    )
    _string(arguments["name"], "name")
    _vector(arguments["from"], "from")
    _vector(arguments["to"], "to")
    for field in ("origin", "rotation"):
        if field in arguments:
            _vector(arguments[field], field)
    if "inflate" in arguments:
        _finite_number(arguments["inflate"], "inflate")
    if "mirror" in arguments:
        _boolean(arguments["mirror"], "mirror")
    for field in ("texture", "parent"):
        if field in arguments:
            _string(arguments[field], field)
    if "uv" in arguments:
        uv = arguments["uv"]
        if isinstance(uv, dict):
            invalid_faces = set(uv) - _FACES
            if invalid_faces:
                raise BlockbenchMCPError(
                    "Blockbench argument 'uv' contains an unsupported face."
                )
            for face, coordinates in uv.items():
                _vector(coordinates, f"uv.{face}", lengths=(4,))
        else:
            _vector(uv, "uv", lengths=(2, 4))


def _validate_set_texture(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "set_texture",
        arguments,
        required=frozenset({"path"}),
        optional=frozenset(
            {
                "name",
                "element",
                "element_id",
                "texture",
                "texture_id",
                "faces",
            }
        ),
    )
    _string(arguments["path"], "path")
    for field in ("name", "element", "element_id", "texture", "texture_id"):
        if field in arguments:
            _string(arguments[field], field)
    if "faces" in arguments:
        _faces(arguments["faces"], "faces")


def _validate_set_uv(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "set_uv",
        arguments,
        required=frozenset({"uv"}),
        optional=frozenset({"element", "element_id", "face", "faces", "rotation"}),
    )
    if not ({"element", "element_id"} & set(arguments)):
        raise BlockbenchMCPError(
            "Blockbench 'set_uv' requires either 'element' or 'element_id'."
        )
    if {"element", "element_id"} <= set(arguments):
        raise BlockbenchMCPError(
            "Blockbench 'set_uv' accepts only one of 'element' or 'element_id'."
        )
    for field in ("element", "element_id"):
        if field in arguments:
            _string(arguments[field], field)
    if "face" in arguments:
        face = _string(arguments["face"], "face")
        if face not in _FACES:
            raise BlockbenchMCPError("Blockbench argument 'face' is unsupported.")
    if "faces" in arguments:
        _faces(arguments["faces"], "faces")
    _vector(arguments["uv"], "uv", lengths=(4,))
    if "rotation" in arguments:
        rotation = _finite_number(arguments["rotation"], "rotation")
        if rotation not in {0, 90, 180, 270}:
            raise BlockbenchMCPError(
                "Blockbench argument 'rotation' must be 0, 90, 180, or 270."
            )


def _validate_create_animation(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "create_animation",
        arguments,
        required=frozenset({"name", "bones"}),
        optional=frozenset(
            {"loop", "animation_length", "particle_effects"}
        ),
    )
    _string(arguments["name"], "name")
    if "loop" in arguments:
        _boolean(arguments["loop"], "loop")
    if "animation_length" in arguments:
        length = _finite_number(arguments["animation_length"], "animation_length")
        if length < 0:
            raise BlockbenchMCPError(
                "Blockbench argument 'animation_length' must not be negative."
            )
    bones = arguments["bones"]
    if not isinstance(bones, dict):
        raise BlockbenchMCPError("Blockbench argument 'bones' must be an object.")
    for bone_name, keyframes in bones.items():
        _string(bone_name, "bones key")
        if not isinstance(keyframes, list):
            raise BlockbenchMCPError(
                f"Blockbench bone {bone_name!r} keyframes must be a list."
            )
        for index, keyframe in enumerate(keyframes):
            field = f"bones.{bone_name}[{index}]"
            if not isinstance(keyframe, dict):
                raise BlockbenchMCPError(
                    f"Blockbench argument {field!r} must be an object."
                )
            _closed_arguments(
                field,
                keyframe,
                required=frozenset({"time"}),
                optional=frozenset(
                    {"position", "rotation", "scale", "interpolation"}
                ),
            )
            _finite_number(keyframe["time"], f"{field}.time")
            for vector_field in ("position", "rotation"):
                if vector_field in keyframe:
                    _vector(
                        keyframe[vector_field],
                        f"{field}.{vector_field}",
                    )
            if "scale" in keyframe:
                scale = keyframe["scale"]
                if isinstance(scale, (list, tuple)):
                    _vector(scale, f"{field}.scale")
                else:
                    _finite_number(scale, f"{field}.scale")
            if "interpolation" in keyframe:
                interpolation = _string(
                    keyframe["interpolation"],
                    f"{field}.interpolation",
                )
                if interpolation not in _INTERPOLATIONS:
                    raise BlockbenchMCPError(
                        f"Blockbench argument {field!r} has unsupported interpolation."
                    )
    if "particle_effects" in arguments:
        effects = arguments["particle_effects"]
        if not isinstance(effects, dict):
            raise BlockbenchMCPError(
                "Blockbench argument 'particle_effects' must be an object."
            )
        for timestamp, effect in effects.items():
            _string(timestamp, "particle_effects key")
            _string(effect, f"particle_effects.{timestamp}")


def _validate_render_preview(arguments: dict[str, Any]) -> None:
    _closed_arguments(
        "render_preview",
        arguments,
        required=frozenset({"output_path"}),
        optional=frozenset({"width", "height", "background"}),
    )
    _string(arguments["output_path"], "output_path")
    for field in ("width", "height"):
        if field in arguments:
            _positive_integer(arguments[field], field)
    if "background" in arguments:
        _string(arguments["background"], "background")


def _validate_no_arguments(
    operation: str, arguments: dict[str, Any]
) -> None:
    _closed_arguments(operation, arguments)


def _validate_export(
    operation: str,
    arguments: dict[str, Any],
    *,
    extra_paths: frozenset[str] = frozenset(),
    multiple_paths: bool = False,
) -> None:
    path_fields = frozenset({"path", "output_path"}) | extra_paths
    _closed_arguments(
        operation,
        arguments,
        optional=path_fields
        | frozenset({"namespace", "model_name", "overwrite"}),
    )
    supplied = path_fields & set(arguments)
    if not supplied or (not multiple_paths and len(supplied) != 1):
        requirement = "at least one" if multiple_paths else "exactly one"
        raise BlockbenchMCPError(
            f"Blockbench {operation!r} requires {requirement} output path."
        )
    for field in supplied:
        _string(arguments[field], field)
    for field in ("namespace", "model_name"):
        if field in arguments:
            _string(arguments[field], field)
    if "overwrite" in arguments:
        _boolean(arguments["overwrite"], "overwrite")


def _validate_operation_arguments(
    operation: str, arguments: dict[str, Any]
) -> None:
    if not isinstance(arguments, dict):
        raise BlockbenchMCPError("Blockbench operation arguments must be an object.")
    if operation == "open_project":
        _validate_open_project(arguments)
    elif operation == "create_cube":
        _validate_create_cube(arguments)
    elif operation == "set_texture":
        _validate_set_texture(arguments)
    elif operation == "set_uv":
        _validate_set_uv(arguments)
    elif operation == "create_animation":
        _validate_create_animation(arguments)
    elif operation == "render_preview":
        _validate_render_preview(arguments)
    elif operation in {"validate_uv", "close_project"}:
        _validate_no_arguments(operation, arguments)
    elif operation == "export_bbmodel":
        _validate_export(operation, arguments)
    elif operation == "export_geckolib":
        _validate_export(
            operation,
            arguments,
            extra_paths=frozenset(
                {
                    "model_output_path",
                    "animation_output_path",
                    "texture_output_path",
                }
            ),
            multiple_paths=True,
        )
    else:
        raise BlockbenchMCPError(
            f"Blockbench operation is not allowlisted: {operation}"
        )


_INPUT_PATHS: dict[str, frozenset[str]] = {
    "open_project": frozenset({"path"}),
    "set_texture": frozenset({"path"}),
}
_OUTPUT_PATHS: dict[str, frozenset[str]] = {
    "render_preview": frozenset({"output_path"}),
    "export_bbmodel": frozenset({"path", "output_path"}),
    "export_geckolib": frozenset(
        {
            "path",
            "output_path",
            "model_output_path",
            "animation_output_path",
            "texture_output_path",
        }
    ),
}


def _validate_local_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise BlockbenchMCPError("Blockbench MCP URL must be a non-empty string.")
    candidate = url.strip()
    if any(character.isspace() or character == "\\" for character in candidate):
        raise BlockbenchMCPError("Blockbench MCP URL contains an invalid character.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise BlockbenchMCPError("Blockbench MCP URL is malformed.") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise BlockbenchMCPError("Blockbench MCP URL must use HTTP or HTTPS.")
    if not parsed.netloc or parsed.hostname is None:
        raise BlockbenchMCPError("Blockbench MCP URL must include a host.")
    if parsed.username is not None or parsed.password is not None:
        raise BlockbenchMCPError("Blockbench MCP URL must not contain userinfo.")
    if parsed.hostname.lower() not in _LOOPBACK_HOSTS:
        raise BlockbenchMCPError(
            "Blockbench MCP must use the literal loopback host "
            "127.0.0.1, ::1, or localhost."
        )
    if parsed.fragment:
        raise BlockbenchMCPError("Blockbench MCP URL must not contain a fragment.")
    if port is not None and not 1 <= port <= 65_535:
        raise BlockbenchMCPError("Blockbench MCP URL contains an invalid port.")
    return candidate


class BlockbenchMCPClient:
    """Restricted streamable-HTTP MCP client for the Blockbench plugin.

    This client deliberately exposes only reviewed modeling operations. It never
    forwards arbitrary script, shell or unrestricted file tools.
    """

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: int = 60,
        workspace_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.url = _validate_local_url(
            url
            or os.environ.get("MMM_BLOCKBENCH_MCP_URL", "").strip()
            or "http://127.0.0.1:3000/bb-mcp"
        )
        configured_root = (
            workspace_root
            or os.environ.get("MMM_BLOCKBENCH_WORKSPACE_ROOT", "").strip()
            or Path.cwd()
        )
        try:
            self.workspace_root = Path(configured_root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise BlockbenchMCPError(
                "Blockbench workspace root must be an existing directory."
            ) from exc
        if not self.workspace_root.is_dir():
            raise BlockbenchMCPError(
                "Blockbench workspace root must be an existing directory."
            )
        self.timeout_seconds = timeout_seconds
        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        )
        self.session_id: str | None = None
        self._request_id = 1

    def initialize(self) -> dict[str, Any]:
        result, headers = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "mmm-blockbench-restricted", "version": "1"},
                },
            }
        )
        self.session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
        self._post(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        if self.session_id is None:
            self.initialize()
        result, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            }
        )
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [
            tool
            for tool in tools
            if isinstance(tool, dict) and tool.get("name") in _ALLOWED
        ]

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(operation, str) or operation not in _ALLOWED:
            raise BlockbenchMCPError(f"Blockbench operation is not allowlisted: {operation}")
        _validate_operation_arguments(operation, arguments)
        safe_arguments = self._normalize_paths(operation, arguments)
        if self.session_id is None:
            self.initialize()
        available = {tool["name"] for tool in self.list_tools()}
        if operation not in available:
            raise BlockbenchMCPError(
                f"Connected Blockbench server does not expose reviewed tool {operation!r}."
            )
        result, _ = self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": operation, "arguments": safe_arguments},
            }
        )
        return {
            "schema_version": "mmm/blockbench-call-result-v1",
            "operation": operation,
            "result": result,
        }

    def close(self) -> None:
        self.client.close()

    def _next_id(self) -> int:
        value = self._request_id
        self._request_id += 1
        return value

    def _normalize_paths(
        self, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = dict(arguments)
        for field in _INPUT_PATHS.get(operation, frozenset()):
            if field in normalized:
                normalized[field] = str(
                    self._workspace_path(normalized[field], field, must_exist=True)
                )
        for field in _OUTPUT_PATHS.get(operation, frozenset()):
            if field in normalized:
                normalized[field] = str(
                    self._workspace_path(normalized[field], field, must_exist=False)
                )
        return normalized

    def _workspace_path(
        self, value: Any, field: str, *, must_exist: bool
    ) -> Path:
        raw = _string(value, field)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise BlockbenchMCPError(
                f"Blockbench path {field!r} cannot be resolved."
            ) from exc
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise BlockbenchMCPError(
                f"Blockbench path {field!r} escapes the configured workspace root."
            ) from exc
        if must_exist:
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(self.workspace_root)
            except ValueError as exc:
                raise BlockbenchMCPError(
                    f"Blockbench path {field!r} escapes the configured workspace root."
                ) from exc
            except (OSError, RuntimeError) as exc:
                raise BlockbenchMCPError(
                    f"Blockbench input path {field!r} does not exist or cannot be resolved."
                ) from exc
            if not resolved.is_file():
                raise BlockbenchMCPError(
                    f"Blockbench input path {field!r} must be a regular file."
                )
        return resolved

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = self.client.post(self.url, json=payload, headers=headers)
        response.raise_for_status()
        parsed = _parse_response(response)
        if "error" in parsed:
            raise BlockbenchMCPError(json.dumps(parsed["error"], ensure_ascii=False))
        result = parsed.get("result", {})
        if not isinstance(result, dict):
            result = {"value": result}
        return result, dict(response.headers)


def _parse_response(response: httpx.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        value = response.json()
        if not isinstance(value, dict):
            raise BlockbenchMCPError("Blockbench MCP returned a non-object response.")
        return value
    messages: list[dict[str, Any]] = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw:
            continue
        value = json.loads(raw)
        if isinstance(value, dict):
            messages.append(value)
    if not messages:
        raise BlockbenchMCPError("Blockbench MCP returned an empty event stream.")
    return messages[-1]


def allowed_blockbench_operations() -> tuple[str, ...]:
    return tuple(sorted(_ALLOWED))
