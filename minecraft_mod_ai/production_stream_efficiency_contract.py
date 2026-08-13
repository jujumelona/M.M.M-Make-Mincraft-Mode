from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Sequence


_ARRAY_FIELDS = (
    "modules",
    "assets",
    "audio",
    "acceptance_tests",
    "completed_deliverables",
)
_ITEM_KIND = {"modules": "module", "assets": "asset", "audio": "audio"}
_KIND_FIELD = {value: key for key, value in _ITEM_KIND.items()}
_FULL_PAGE_DECODE_LIMIT = 2


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _skip_ws(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _parse_array_prefix(text: str, position: int) -> tuple[list[Any], int, str]:
    """Return strict complete array elements plus the exact unresolved tail.

    No brace/bracket is invented and no malformed element is accepted.  A completed
    sibling can therefore be committed even when the following child was cut by the
    model output limit.
    """

    decoder = json.JSONDecoder()
    values: list[Any] = []
    position += 1
    while True:
        position = _skip_ws(text, position)
        if position >= len(text):
            return values, position, ""
        if text[position] == "]":
            return values, position + 1, ""
        if text[position] == ",":
            position = _skip_ws(text, position + 1)
            if position >= len(text):
                return values, position, ""
        start = position
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return values, len(text), text[start:].strip()
        values.append(value)
        position = _skip_ws(text, int(end))
        if position < len(text) and text[position] not in {",", "]"}:
            return values, len(text), text[position:].strip()


def _scan_root_candidate(text: str, start: int) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    values: dict[str, list[Any]] = {field: [] for field in _ARRAY_FIELDS}
    seen: list[str] = []
    incomplete_kind = ""
    incomplete_fragment = ""
    position = start + 1

    while True:
        position = _skip_ws(text, position)
        if position >= len(text) or text[position] == "}":
            break
        if text[position] == ",":
            position = _skip_ws(text, position + 1)
        if position >= len(text):
            break
        try:
            key, key_end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(key, str):
            break
        position = _skip_ws(text, int(key_end))
        if position >= len(text) or text[position] != ":":
            break
        position = _skip_ws(text, position + 1)
        if position >= len(text):
            break

        if key in _ARRAY_FIELDS and text[position] == "[":
            parsed, end, fragment = _parse_array_prefix(text, position)
            values[key].extend(parsed)
            seen.append(key)
            position = end
            if fragment:
                incomplete_kind = _ITEM_KIND.get(key, "")
                incomplete_fragment = fragment
                break
            continue

        try:
            _ignored, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        position = int(end)

    return {
        "start": start,
        "seen": seen,
        "values": values,
        "incomplete_kind": incomplete_kind,
        "incomplete_fragment": incomplete_fragment,
    }


def _best_truncated_root(text: str) -> dict[str, Any] | None:
    # The intended structured response root appears before any nested config object.
    # Prefer the earliest candidate that actually exposes a production top-level array;
    # this avoids mistaking nested config keys such as "assets" for the page root.
    for index, character in enumerate(text):
        if character != "{":
            continue
        candidate = _scan_root_candidate(text, index)
        if candidate["seen"]:
            return candidate
    return None


def _stream_event_path(stage: str, request: dict[str, Any]) -> Path:
    from .production_page_durable_contract import page_checkpoint_path

    page_path = page_checkpoint_path(stage, request)
    return page_path.with_name(page_path.name + ".stream.jsonl")


def _latest_saved_stream(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        return ""
    latest = ""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    break
                if isinstance(value, dict) and isinstance(value.get("text"), str):
                    latest = value["text"]
    except OSError:
        return ""
    return latest


def _append_stream_event(
    *,
    stage: str,
    request: dict[str, Any],
    round_index: int,
    text: str,
    diagnostic: str,
) -> Path:
    path = _stream_event_path(stage, request)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "round": round_index,
        "sha256": _fingerprint(text),
        "diagnostic": diagnostic,
        "text": text,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(event))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(_canonical(value))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _item_parser(module: Any, kind: str):
    name = {"module": "_module", "asset": "_asset", "audio": "_audio"}[kind]
    return getattr(module, name)


def _repair_truncated_item(
    module: Any,
    runtime: Any,
    router: Any,
    *,
    stage: str,
    request: dict[str, Any],
    kind: str,
    fragment: str,
    targets: Sequence[str],
) -> dict[str, Any]:
    """Repair only one syntactically incomplete child; backend errors propagate."""

    from .planner_strict_json_contract import _extract_one_complete_object
    from .production_page_durable_contract import page_checkpoint_path

    fingerprint = _fingerprint(fragment)
    page_path = page_checkpoint_path(stage, request)
    state_path = page_path.with_name(
        page_path.name + f".truncated-{kind}-{fingerprint[:20]}.json"
    )
    saved = _read_json(state_path)
    if saved.get("fingerprint") == fingerprint and isinstance(saved.get("resolved"), dict):
        candidate = dict(saved["resolved"])
        _item_parser(module, kind)(candidate)
        return candidate

    schema = {
        "type": "object",
        "properties": {
            "target_fingerprint": {"type": "string"},
            "replacement": {"type": "object", "additionalProperties": True},
        },
        "required": ["target_fingerprint", "replacement"],
        "additionalProperties": False,
    }
    round_index = int(saved.get("round", 0)) if saved.get("fingerprint") == fingerprint else 0
    previous_error = str(saved.get("error", "")) if saved else ""
    target_set = set(targets)

    while True:
        round_index += 1
        _atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "status": "repairing",
                "round": round_index,
                "kind": kind,
                "fragment": fragment,
                "targets": list(targets),
                "error": previous_error,
            },
        )
        system = (
            f"Repair exactly ONE truncated Minecraft production {kind}. Complete siblings are "
            "already saved: never emit a page or sibling. Return exactly target_fingerprint "
            "and replacement. Reconstruct only this child from saved_truncated_fragment and "
            "preserve every recoverable value. replacement must include a non-empty "
            "implements_deliverables array using only exact remaining_deliverables. JSON only."
        )
        user = {
            "target_fingerprint": fingerprint,
            "item_kind": kind,
            "saved_truncated_fragment": fragment,
            "remaining_deliverables": list(targets),
            "previous_validation_error": previous_error,
        }
        token = runtime._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": _canonical(user)},
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime._JSON_SCHEMA.reset(token)

        # Only semantic/format errors loop here. router/backend exceptions occur above
        # this block and propagate, leaving the fsynced fragment/state for next run.
        try:
            patch = _extract_one_complete_object(text)
            if set(patch) != {"target_fingerprint", "replacement"}:
                raise ValueError("truncated-item repair fields are invalid")
            if patch.get("target_fingerprint") != fingerprint:
                raise ValueError("truncated-item fingerprint mismatch")
            replacement = patch.get("replacement")
            if not isinstance(replacement, dict):
                raise ValueError("replacement must be an object")
            claims = replacement.get("implements_deliverables")
            if not isinstance(claims, list) or not claims:
                raise ValueError("replacement must declare implements_deliverables")
            normalized = [
                str(value).strip()
                for value in claims
                if isinstance(value, str) and str(value).strip()
            ]
            if not normalized or any(value not in target_set for value in normalized):
                raise ValueError("replacement attribution is outside the host target")
            replacement["implements_deliverables"] = normalized
            _item_parser(module, kind)(replacement)
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"
            continue

        _atomic_json(
            state_path,
            {
                "fingerprint": fingerprint,
                "status": "resolved",
                "round": round_index,
                "kind": kind,
                "fragment": fragment,
                "resolved": replacement,
            },
        )
        return replacement


def _dedupe(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _fingerprint(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _repair_missing_attribution(
    runtime: Any,
    router: Any,
    *,
    items: list[tuple[str, dict[str, Any]]],
    targets: Sequence[str],
) -> None:
    """Patch attribution metadata only; never rewrite salvaged production objects."""

    if len(targets) <= 1 or not items:
        return
    target_set = set(targets)
    unresolved: list[tuple[str, dict[str, Any], str]] = []
    for kind, item in items:
        claims = item.get("implements_deliverables")
        valid = [
            str(value).strip()
            for value in claims
            if isinstance(value, str) and str(value).strip() in target_set
        ] if isinstance(claims, list) else []
        if valid:
            item["implements_deliverables"] = valid
        else:
            unresolved.append((kind, item, _fingerprint(item)))
    if not unresolved:
        return

    schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fingerprint": {"type": "string"},
                        "implements_deliverables": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["fingerprint", "implements_deliverables"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }
    from .planner_strict_json_contract import _extract_one_complete_object

    previous_error = ""
    while True:
        user = {
            "remaining_deliverables": list(targets),
            "saved_objects": [
                {"kind": kind, "fingerprint": fingerprint, "object": item}
                for kind, item, fingerprint in unresolved
            ],
            "previous_validation_error": previous_error,
        }
        token = runtime._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {
                        "role": "system",
                        "content": (
                            "Return attribution patches only; never rewrite production objects. "
                            "For every saved fingerprint assign one or more exact host remaining "
                            "deliverables it implements. Return one JSON object with assignments."
                        ),
                    },
                    {"role": "user", "content": _canonical(user)},
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime._JSON_SCHEMA.reset(token)

        try:
            raw = _extract_one_complete_object(text)
            if set(raw) != {"assignments"} or not isinstance(raw["assignments"], list):
                raise ValueError("attribution response fields are invalid")
            expected = {fingerprint for _kind, _item, fingerprint in unresolved}
            assignments: dict[str, list[str]] = {}
            for value in raw["assignments"]:
                if not isinstance(value, dict) or set(value) != {
                    "fingerprint",
                    "implements_deliverables",
                }:
                    raise ValueError("attribution assignment is invalid")
                fingerprint = str(value["fingerprint"])
                claims = value["implements_deliverables"]
                if fingerprint not in expected or fingerprint in assignments:
                    raise ValueError("unknown or duplicate attribution fingerprint")
                if not isinstance(claims, list) or not claims:
                    raise ValueError("attribution must be non-empty")
                normalized = [str(item).strip() for item in claims if isinstance(item, str)]
                if not normalized or any(item not in target_set for item in normalized):
                    raise ValueError("attribution contains a non-host deliverable")
                assignments[fingerprint] = normalized
            if set(assignments) != expected:
                raise ValueError("attribution omitted a saved production object")
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"
            continue

        for _kind, item, fingerprint in unresolved:
            item["implements_deliverables"] = assignments[fingerprint]
        return


def _merge_complete_production_pages(
    module: Any,
    runtime: Any,
    text: str,
    request: dict[str, Any],
) -> tuple[dict[str, list[Any]], int]:
    from .planner_strict_json_contract import _outermost_complete_json_containers

    merged: dict[str, list[Any]] = {field: [] for field in _ARRAY_FIELDS}
    last_end = 0
    for container in _outermost_complete_json_containers(text):
        if not isinstance(container.value, dict):
            continue
        try:
            page = runtime._extract_production_page_with_host_bookkeeping(
                module,
                _canonical(container.value),
                request,
            )
            runtime._validate_production_progress(
                module,
                page,
                request,
                (frozenset(module._PRODUCTION_PAGE_CONTRACT),),
            )
        except Exception:
            continue
        for field in _ARRAY_FIELDS:
            merged[field].extend(page.get(field, []))
        last_end = max(last_end, int(container.end))
    return merged, last_end


def _salvage_production_stream(
    module: Any,
    runtime: Any,
    router: Any,
    *,
    text: str,
    request: dict[str, Any],
    stage: str,
) -> dict[str, Any] | None:
    merged, complete_end = _merge_complete_production_pages(module, runtime, text, request)
    tail = text[complete_end:] if complete_end else text
    prefix = _best_truncated_root(tail)
    prefix_items: list[tuple[str, dict[str, Any]]] = []

    if prefix is not None:
        values = prefix["values"]
        for field in _ARRAY_FIELDS:
            merged[field].extend(values[field])
        for field, kind in _ITEM_KIND.items():
            for value in values[field]:
                if isinstance(value, dict):
                    prefix_items.append((kind, value))

        kind = str(prefix.get("incomplete_kind", ""))
        fragment = str(prefix.get("incomplete_fragment", ""))
        targets = runtime._target_names(request)
        if kind and fragment and targets:
            repaired = _repair_truncated_item(
                module,
                runtime,
                router,
                stage=stage,
                request=request,
                kind=kind,
                fragment=fragment,
                targets=targets,
            )
            merged[_KIND_FIELD[kind]].append(repaired)
            prefix_items.append((kind, repaired))

    for field in _ARRAY_FIELDS:
        merged[field] = _dedupe(merged[field])

    targets = runtime._target_names(request)
    _repair_missing_attribution(runtime, router, items=prefix_items, targets=targets)

    modules = [value for value in merged["modules"] if isinstance(value, dict)]
    assets = [value for value in merged["assets"] if isinstance(value, dict)]
    audio = [value for value in merged["audio"] if isinstance(value, dict)]
    tests = [
        str(value).strip()
        for value in merged["acceptance_tests"]
        if isinstance(value, str) and str(value).strip()
    ]
    candidate = {
        "modules": modules,
        "assets": assets,
        "audio": audio,
        "acceptance_tests": tests,
        "completed_deliverables": [
            str(value).strip()
            for value in merged["completed_deliverables"]
            if isinstance(value, str) and str(value).strip()
        ],
    }
    completed = runtime._derive_completed_deliverables(
        candidate,
        targets=targets,
        modules=modules,
        assets=assets,
        audio=audio,
        acceptance_tests=tests,
    )
    if targets and not completed:
        return None
    if not (modules or assets or audio or tests):
        return None

    remaining = runtime._remaining_names(request)
    completed_set = set(completed)
    still_remaining = [value for value in remaining if value not in completed_set]
    page = {
        "modules": modules,
        "assets": assets,
        "audio": audio,
        "acceptance_tests": tests,
        "completed_deliverables": completed,
        "complete": not still_remaining,
        "next_cursor": "" if not still_remaining else f"host_remaining_{len(still_remaining)}",
    }
    runtime._validate_production_progress(
        module,
        page,
        request,
        (frozenset(module._PRODUCTION_PAGE_CONTRACT),),
    )
    return page


def install(complete_planner_module: Any) -> None:
    """Make production JSON streaming lossless without increasing GPU concurrency."""

    current = complete_planner_module._generate_json_page_with_repair
    if getattr(current, "_mmm_lossless_production_stream", False):
        return

    @wraps(current)
    def generate_json_page_lossless(
        router: Any,
        *,
        system_prompt: str,
        request: dict[str, Any] | str,
        media_paths: Sequence[Any],
        expected_contracts: Sequence[frozenset[str]],
        stage: str,
    ) -> dict[str, Any]:
        from . import planner_json_runtime_contract as runtime

        production = (
            isinstance(request, dict)
            and len(expected_contracts) == 1
            and expected_contracts[0]
            == frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT)
        )
        if not production:
            return current(
                router,
                system_prompt=system_prompt,
                request=request,
                media_paths=media_paths,
                expected_contracts=expected_contracts,
                stage=stage,
            )

        saved_text = _latest_saved_stream(_stream_event_path(stage, request))
        if saved_text:
            saved_page = _salvage_production_stream(
                complete_planner_module,
                runtime,
                router,
                text=saved_text,
                request=request,
                stage=stage,
            )
            if saved_page is not None:
                return saved_page

        view = runtime._contract_view(request, expected_contracts)
        schema = runtime._schema_for_contract(view) if view is not None else None
        contract_text = (
            json.dumps(view, ensure_ascii=False, separators=(",", ":"))
            if view is not None
            else "required top-level fields: "
            + ", ".join(sorted(complete_planner_module._PRODUCTION_PAGE_CONTRACT))
        )
        previous_diagnostic = ""
        round_index = 0

        while round_index < _FULL_PAGE_DECODE_LIMIT:
            prompt = (
                system_prompt
                + "\n\nHOST JSON CONTRACT: Return production JSON with these top-level fields and "
                + "compatible value types: "
                + contract_text
                + ". The host imposes NO fixed item/deliverable width. Choose any coherent "
                + "non-empty subset of remaining work that you can finish as valid JSON. "
                + "Finish the current child before starting another; when budget is tight, stop "
                + "after a complete child/page instead of starting one you cannot finish. Never "
                + "repeat IDs already present in host catalogs."
            )
            if round_index:
                prompt += (
                    "\nCORRECTION: the previous raw stream was durably saved but contained no "
                    "host-verifiable object after strict lossless salvage: "
                    + previous_diagnostic
                    + ". Return a clean page; do not pad, explain, or echo the request."
                )

            token = runtime._JSON_SCHEMA.set(schema)
            try:
                text = router.generate_text(
                    "planner",
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    ],
                    media_paths=media_paths if round_index == 0 else (),
                    response_format="json",
                )
            finally:
                runtime._JSON_SCHEMA.reset(token)

            try:
                page = runtime._extract_production_page_with_host_bookkeeping(
                    complete_planner_module,
                    text,
                    request,
                )
                runtime._validate_production_progress(
                    complete_planner_module,
                    page,
                    request,
                    expected_contracts,
                )
                return page
            except Exception as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
                _append_stream_event(
                    stage=stage,
                    request=request,
                    round_index=round_index,
                    text=text,
                    diagnostic=diagnostic,
                )

            # Do not catch exceptions from salvage. Semantic repair errors are handled
            # inside the child/attribution loops; backend/process failures propagate so
            # the fsynced stream remains the restart point instead of triggering a new
            # expensive full-page decode.
            salvaged = _salvage_production_stream(
                complete_planner_module,
                runtime,
                router,
                text=text,
                request=request,
                stage=stage,
            )
            if salvaged is not None:
                return salvaged

            previous_diagnostic = diagnostic + "; salvage=no verified production item"
            round_index += 1

        raise complete_planner_module.SpecValidationError(
            "Production page failed after one page-local repair; durable stream/item "
            "repair remains authoritative."
        )

    generate_json_page_lossless._mmm_lossless_production_stream = True  # type: ignore[attr-defined]
    generate_json_page_lossless._mmm_saved_stream_resume = True  # type: ignore[attr-defined]
    generate_json_page_lossless._mmm_bounded_full_page_decode = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_json_page_lossless


__all__ = [
    "install",
    "_append_stream_event",
    "_salvage_production_stream",
    "_stream_event_path",
]
