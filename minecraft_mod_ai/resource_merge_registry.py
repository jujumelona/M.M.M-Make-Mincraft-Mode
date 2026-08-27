from __future__ import annotations

"""Canonical path normalization and semantic merging for mod resources."""

import json
import re
from pathlib import PurePosixPath


_RESOURCE_NAMESPACE_RE = re.compile(
    r"^(src/main/resources/(?:assets|data)/)([^/]+)/(.+)$"
)
_MIXIN_CONFIG_RE = re.compile(r"\.mixins?\.json$", re.IGNORECASE)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")


def _safe_relative_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or _WINDOWS_DRIVE_RE.match(raw):
        raise ValueError("RESOURCE_PATH_MUST_BE_WORKSPACE_RELATIVE")
    parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    if not parts or ".." in parts:
        raise ValueError("RESOURCE_PATH_TRAVERSAL_FORBIDDEN")
    return "/".join(parts)


def _json_object(path: str, content: str) -> tuple[dict[str, object] | None, str]:
    try:
        value = json.loads(content)
    except Exception as exc:
        return None, f"JSON_PARSE_ERROR in {path}: {exc}"
    if not isinstance(value, dict):
        return None, f"JSON_OBJECT_REQUIRED: {path}"
    return value, ""


class ResourceMergeRegistry:
    """Single authority for resource paths, normalization, and collisions."""

    @classmethod
    def canonical_path(cls, path: str, *, target_modid: str) -> str:
        normalized = _safe_relative_path(path)
        match = _RESOURCE_NAMESPACE_RE.match(normalized)
        if match:
            prefix, rest = match.group(1), match.group(3)
            return f"{prefix}{target_modid}/{rest}"

        resource = PurePosixPath(normalized)
        if resource.parent.as_posix() == "src/main/resources":
            if _MIXIN_CONFIG_RE.search(resource.name):
                return f"src/main/resources/{target_modid}.mixins.json"
            if resource.name.endswith(".accesswidener"):
                return f"src/main/resources/{target_modid}.accesswidener"
        return normalized

    @classmethod
    def can_merge(cls, path: str) -> bool:
        normalized = _safe_relative_path(path)
        name = PurePosixPath(normalized).name
        return (
            normalized == "src/main/resources/fabric.mod.json"
            or bool(_MIXIN_CONFIG_RE.search(name))
            or name.endswith(".accesswidener")
            or (
                name.endswith(".json")
                and ("/assets/" in f"/{normalized}" or "/data/" in f"/{normalized}")
            )
        )

    @classmethod
    def normalize(
        cls,
        path: str,
        content: str,
        *,
        target_modid: str,
    ) -> tuple[str, bool, str]:
        """Validate one resource and apply target-owned metadata names."""
        normalized = _safe_relative_path(path)
        name = PurePosixPath(normalized).name

        if name.endswith(".accesswidener"):
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            headers = [line for line in lines if line.startswith("accessWidener")]
            if len(headers) != 1:
                return content, False, f"ACCESS_WIDENER_HEADER_REQUIRED: {normalized}"
            return content, True, ""

        if not name.endswith(".json"):
            return content, True, ""

        value, error = _json_object(normalized, content)
        if value is None:
            return content, False, error

        if normalized == "src/main/resources/fabric.mod.json":
            value["id"] = target_modid
            value["name"] = target_modid.replace("_", " ").title()

            entrypoints = value.get("entrypoints")
            if isinstance(entrypoints, dict):
                for key, entries in tuple(entrypoints.items()):
                    if isinstance(entries, list):
                        entrypoints[key] = list(dict.fromkeys(entries))

            mixins = value.get("mixins")
            if isinstance(mixins, list):
                canonical = f"{target_modid}.mixins.json"
                value["mixins"] = list(
                    dict.fromkeys(
                        canonical
                        if isinstance(item, str) and _MIXIN_CONFIG_RE.search(item)
                        else item
                        for item in mixins
                    )
                )
            if isinstance(value.get("accessWidener"), str):
                value["accessWidener"] = f"{target_modid}.accesswidener"

        return json.dumps(value, indent=2), True, ""

    @classmethod
    def merge(
        cls,
        path: str,
        existing: str,
        incoming: str,
        *,
        target_modid: str = "",
    ) -> tuple[str, bool, str]:
        """Return ``(content, valid, error)`` for a typed collision."""
        normalized = _safe_relative_path(path)
        name = PurePosixPath(normalized).name

        if name.endswith(".accesswidener"):
            return cls._merge_access_widener(normalized, existing, incoming)

        a, error = _json_object(normalized, existing)
        if a is None:
            return existing, False, error
        b, error = _json_object(normalized, incoming)
        if b is None:
            return existing, False, error

        if normalized == "src/main/resources/fabric.mod.json":
            return cls._merge_fabric_metadata(
                normalized, a, b, target_modid=target_modid
            )

        if _MIXIN_CONFIG_RE.search(name):
            return cls._merge_mixin_config(normalized, a, b)

        parts = PurePosixPath(normalized).parts

        if "tags" in parts:
            values_a = a.get("values", [])
            values_b = b.get("values", [])
            if not isinstance(values_a, list) or not isinstance(values_b, list):
                return existing, False, f"TAG_VALUES_ARRAY_REQUIRED: {normalized}"
            a["values"] = list(dict.fromkeys([*values_a, *values_b]))
            if "replace" in b:
                a["replace"] = b["replace"]
            return json.dumps(a, indent=2), True, ""

        if "lang" in parts:
            for key, value in b.items():
                if key in a and a[key] != value:
                    return existing, False, (
                        f"LANG_KEY_CONFLICT: '{key}' has conflicting translations "
                        f"in {normalized}"
                    )
                a[key] = value
            return json.dumps(a, indent=2), True, ""

        if name == "sounds.json":
            for event_name, event_data in b.items():
                if event_name in a and a[event_name] != event_data:
                    return existing, False, (
                        f"SOUND_EVENT_CONFLICT: '{event_name}' in {normalized}"
                    )
                a[event_name] = event_data
            return json.dumps(a, indent=2), True, ""

        if "recipes" in parts:
            if a != b:
                return existing, False, f"DUPLICATE_RECIPE_ID_CONFLICT: {normalized}"
            return existing, True, ""

        if "loot_tables" in parts:
            pools_a = a.get("pools")
            pools_b = b.get("pools")
            if isinstance(pools_a, list) and isinstance(pools_b, list):
                a["pools"] = pools_a + [pool for pool in pools_b if pool not in pools_a]
                return json.dumps(a, indent=2), True, ""
            if a != b:
                return existing, False, f"LOOT_TABLE_CONFLICT: {normalized}"
            return existing, True, ""

        if "models" in parts:
            if a != b:
                return existing, False, f"MODEL_DEFINITION_CONFLICT: {normalized}"
            return existing, True, ""

        if "blockstates" in parts:
            variants_a = a.get("variants")
            variants_b = b.get("variants")
            if isinstance(variants_a, dict) and isinstance(variants_b, dict):
                for key, value in variants_b.items():
                    if key in variants_a and variants_a[key] != value:
                        return existing, False, (
                            f"BLOCKSTATE_VARIANT_CONFLICT: '{key}' in {normalized}"
                        )
                    variants_a[key] = value
                a["variants"] = variants_a
                return json.dumps(a, indent=2), True, ""

        if a == b:
            return existing, True, ""
        return existing, False, f"UNRESOLVED_RESOURCE_CONFLICT: {normalized}"

    @classmethod
    def _merge_fabric_metadata(
        cls,
        path: str,
        existing: dict[str, object],
        incoming: dict[str, object],
        *,
        target_modid: str,
    ) -> tuple[str, bool, str]:
        merged = dict(existing)
        if target_modid:
            merged["id"] = target_modid
            merged["name"] = target_modid.replace("_", " ").title()

        entrypoints = (
            dict(merged.get("entrypoints", {}))
            if isinstance(merged.get("entrypoints"), dict)
            else {}
        )
        incoming_entrypoints = incoming.get("entrypoints", {})
        if not isinstance(incoming_entrypoints, dict):
            return json.dumps(existing, indent=2), False, (
                f"FABRIC_ENTRYPOINTS_OBJECT_REQUIRED: {path}"
            )
        for key, value in incoming_entrypoints.items():
            current = entrypoints.get(key, [])
            current_values = (
                current if isinstance(current, list) else ([current] if current else [])
            )
            incoming_values = (
                value if isinstance(value, list) else ([value] if value else [])
            )
            entrypoints[key] = list(
                dict.fromkeys([*current_values, *incoming_values])
            )
        if entrypoints:
            merged["entrypoints"] = entrypoints

        current_mixins = merged.get("mixins", [])
        incoming_mixins = incoming.get("mixins", [])
        current_values = (
            current_mixins
            if isinstance(current_mixins, list)
            else ([current_mixins] if current_mixins else [])
        )
        incoming_values = (
            incoming_mixins
            if isinstance(incoming_mixins, list)
            else ([incoming_mixins] if incoming_mixins else [])
        )
        if current_values or incoming_values:
            canonical = f"{target_modid}.mixins.json" if target_modid else ""
            merged["mixins"] = list(
                dict.fromkeys(
                    canonical
                    if canonical and isinstance(item, str) and _MIXIN_CONFIG_RE.search(item)
                    else item
                    for item in [*current_values, *incoming_values]
                )
            )

        depends = (
            dict(merged.get("depends", {}))
            if isinstance(merged.get("depends"), dict)
            else {}
        )
        incoming_depends = incoming.get("depends", {})
        if not isinstance(incoming_depends, dict):
            return json.dumps(existing, indent=2), False, (
                f"FABRIC_DEPENDS_OBJECT_REQUIRED: {path}"
            )
        for dep_id, constraint in incoming_depends.items():
            if dep_id in depends and depends[dep_id] != constraint:
                return json.dumps(existing, indent=2), False, (
                    f"FABRIC_DEPENDENCY_CONFLICT: '{dep_id}' in {path}"
                )
            depends[dep_id] = constraint
        if depends:
            merged["depends"] = depends

        if target_modid and (
            "accessWidener" in merged or "accessWidener" in incoming
        ):
            merged["accessWidener"] = f"{target_modid}.accesswidener"
        for key, value in incoming.items():
            if key not in merged:
                merged[key] = value
        return json.dumps(merged, indent=2), True, ""

    @classmethod
    def _merge_mixin_config(
        cls,
        path: str,
        existing: dict[str, object],
        incoming: dict[str, object],
    ) -> tuple[str, bool, str]:
        merged = dict(existing)
        for key in ("mixins", "client", "server"):
            current = merged.get(key, [])
            addition = incoming.get(key, [])
            if not isinstance(current, list) or not isinstance(addition, list):
                return json.dumps(existing, indent=2), False, (
                    f"MIXIN_CLASS_ARRAY_REQUIRED: '{key}' in {path}"
                )
            values = list(dict.fromkeys([*current, *addition]))
            if values:
                merged[key] = values

        for key, value in incoming.items():
            if key in {"mixins", "client", "server"}:
                continue
            if key in merged and merged[key] != value:
                return json.dumps(existing, indent=2), False, (
                    f"MIXIN_CONFIG_CONFLICT: '{key}' in {path}"
                )
            merged[key] = value
        return json.dumps(merged, indent=2), True, ""

    @classmethod
    def _merge_access_widener(
        cls,
        path: str,
        existing: str,
        incoming: str,
    ) -> tuple[str, bool, str]:
        def split(content: str) -> tuple[str, list[str]]:
            header = ""
            declarations: list[str] = []
            for raw_line in content.splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("accessWidener"):
                    if not header:
                        header = line
                    continue
                declarations.append(line)
            return header, declarations

        header_a, declarations_a = split(existing)
        header_b, declarations_b = split(incoming)
        if not header_a or not header_b:
            return existing, False, f"ACCESS_WIDENER_HEADER_REQUIRED: {path}"
        if header_a != header_b:
            return existing, False, f"ACCESS_WIDENER_HEADER_CONFLICT: {path}"
        declarations = list(dict.fromkeys([*declarations_a, *declarations_b]))
        return header_a + "\n" + "\n".join(declarations) + "\n", True, ""
