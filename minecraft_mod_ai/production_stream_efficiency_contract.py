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
_ITEM_KIND = {
    "modules": "module",
    "assets": "asset",
    "audio": "audio",
}


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


def _parse_array_prefix(
    text: str,
    position: int,
) -> tuple[list[Any], int, str]:
    """Parse only already-complete JSON elements from one array prefix.

    ``position`` points at ``[``.  No closing token is invented.  If the last
    element is truncated, its exact raw fragment is returned separately.
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
        position = int(end)
        position = _skip_ws(text, position)
        if position < len(text) and text[position] not in {",", "]"}:
            # A complete value followed by malformed/truncated material. Preserve the
            # complete value and expose only the unresolved suffix.
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

        # Scalar bookkeeping or unrelated fields are skipped only if Python's strict
        # decoder can already parse the complete value. A truncated unknown value ends
        # this candidate; it is never repaired implicitly.
        try:
            _value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        position = int(end)

    score = (
        len(seen) * 1000
        + sum(len(values[field]) for field in _ARRAY_FIELDS) * 10
        + (1 if incomplete_fragment else 0)
    )
    return {
        "score": score,
        "start": start,
        "seen": seen,
        "values": values,
        "incomplete_kind": incomplete_kind,
        "incomplete_fragment": incomplete_fragment,
    }


def _best_truncated_root(text: str) -> dict[str, Any] | None:
    candidates = [
        _scan_root_candidate(text, index)
        for index, character in enumerate(text)
        if character == "{"
    ]
    candidates = [candidate for candidate in candidates if candidate["seen"]]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate["score"], candidate["start"]))


def _stream_event_path(stage: str, request: dict[str, Any]) -> Path:
    from .production_page_durable_contract import page_checkpoint_path

    page_path = page_checkpoint_path(stage, request)
    return page_path.with_name(page_path.name + ".stream.jsonl")


def _append_stream_event(
    *,
    stage: str,
    request: dict[str, Any],
    round_index: int,
    text: str,
    diagnostic: str,
) -> Path:
    path = _stream_event_path(stage, request)
    event = {
        "round": round_index,
        "sha256": _fingerprint(text),
        "diagnostic": diagnostic,
        "text": text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return getattr(module, {"module": "_module", "asset": "_asset", "audio": "_audio"}[kind])


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
    """Replace only the one syntactically incomplete child object.

    A partial JSON object has no safe field-level representation yet, so replacing this
    child is the smallest sound repair unit. The exact fragment remains durably stored.
    """

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
        prompt = (
            f"Repair exactly ONE truncated Minecraft production {kind}. The host has already "
            "saved every complete sibling object, so NEVER emit a page or sibling. Return "
            "exactly target_fingerprint and replacement. Reconstruct only this incomplete "
            "child from its saved fragment. Preserve every recoverable field/value. The "
            "replacement must include implements_deliverables as a non-empty array containing "
            "only exact names from remaining_deliverables that this object implements. Return "
            "JSON only, no Markdown or explanation."
        )
        user = {
            "target_fingerprint": fingerprint,
            "item_kind": kind,
            "saved_truncated_fragment": fragment,
            "remaining_deliverables": list(targets),
            "previous_validation_error": previous_error,
            "output_contract": {
                "target_fingerprint": fingerprint,
                "replacement": {
                    "same_item_reconstructed": True,
                    "implements_deliverables": ["exact remaining deliverable name"],
                },
            },
        }
        token = runtime._JSON_SCHEMA.set(schema)
        try:
            text = router.generate_text(
                "planner",
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": _canonical(user)},
                ],
                media_paths=(),
                response_format="json",
            )
        finally:
            runtime._JSON_SCHEMA.reset(token)

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
            target_set = set(targets)
            normalized_claims = [
                str(value).strip()
                for value in claims
                if isinstance(value, str) and str(value).strip()
            ]
            if not normalized_claims or any(value not in target_set for value in normalized_claims):
                raise ValueError("replacement deliverable attribution is outside the host target")
            replacement["implements_deliverables"] = normalized_claims
            _item_parser(module, kind)(replacement)
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
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"


def _dedupe(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = _fingerprint(value)
        if key in seen:
            continue
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
    """Patch only attribution metadata on salvaged complete objects when needed."""

    if len(targets) <= 1 or not items:
        return
    target_set = set(targets)
    unresolved: list[tuple[str, dict[str, Any], str]] = []
    for kind, item in items:
        claims = item.get("implements_deliverables")
        normalized = [
            str(value).strip()
            for value in claims
            if isinstance(value, str) and str(value).strip() in target_set
        ] if isinstance(claims, list) else []
        if normalized:
            item["implements_deliverables"] = normalized
            continue
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
        prompt = (
            "Return attribution patches only. Do NOT rewrite any production object. For every "
            "saved object fingerprint, choose the exact remaining_deliverables it implements. "
            "Every assignment must use only host-provided fingerprints and exact deliverable "
            "names. Return one JSON object with assignments only."
        )
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
                    {"role": "system", "content": prompt},
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
                if not isinstance(value, dict) or set(value) != {"fingerprint", "implements_deliverables"}:
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
            for _kind, item, fingerprint in unresolved:
                item["implements_deliverables"] = assignments[fingerprint]
            return
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"


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
            field = {value: key for key, value in _ITEM_KIND.items()}[kind]
            merged[field].append(repaired)
            prefix_items.append((kind, repaired))

    for field in _ARRAY_FIELDS:
        merged[field] = _dedupe(merged[field])

    targets = runtime._target_names(request)
    _repair_missing_attribution(
        runtime,
        router,
        items=prefix_items,
        targets=targets,
    )

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
            and expected_contracts[0] == frozenset(complete_planner_module._PRODUCTION_PAGE_CONTRACT)
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

        view = runtime._contract_view(request, expected_contracts)
        schema = runtime._schema_for_contract(view) if view is not None else None
        contract_text = (
            json.dumps(view, ensure_ascii=False, separators=(",", ":"))
            if view is not None
            else "required top-level fields: " + ", ".join(sorted(complete_planner_module._PRODUCTION_PAGE_CONTRACT))
        )
        previous_diagnostic = ""
        round_index = 0
        while True:
            prompt = (
                system_prompt
                + "\n\nHOST JSON CONTRACT: Return production JSON with these top-level fields and "
                + "compatible value types: "
                + contract_text
                + ". The host imposes NO fixed item/deliverable width. Choose any coherent "
                + "non-empty subset of the remaining work that you can finish as valid JSON. "
                + "Finish the current child object before starting another one; if output budget "
                + "is getting tight, stop after a complete child/page instead of beginning an "
                + "object you cannot finish. Do not repeat host catalog IDs."
            )
            if round_index:
                prompt += (
                    "\nCORRECTION: the previous raw stream was durably saved but contained no "
                    "host-verifiable production progress after lossless salvage: "
                    + previous_diagnostic
                    + ". Return a clean production page now. Do not pad, explain, or echo the request."
                )
            request_text = json.dumps(request, ensure_ascii=False)
            token = runtime._JSON_SCHEMA.set(schema)
            try:
                text = router.generate_text(
                    "planner",
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": request_text},
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
                try:
                    salvaged = _salvage_production_stream(
                        complete_planner_module,
                        runtime,
                        router,
                        text=text,
                        request=request,
                        stage=stage,
                    )
                except Exception as salvage_exc:
                    previous_diagnostic = (
                        diagnostic
                        + "; salvage="
                        + f"{type(salvage_exc).__name__}: {salvage_exc}"
                    )
                    round_index += 1
                    continue
                if salvaged is not None:
                    return salvaged
                previous_diagnostic = diagnostic + "; salvage=no verified production item"
                round_index += 1

    generate_json_page_lossless._mmm_lossless_production_stream = True  # type: ignore[attr-defined]
    complete_planner_module._generate_json_page_with_repair = generate_json_page_lossless


__all__ = ["install"]
