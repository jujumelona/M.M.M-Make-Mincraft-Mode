"""Pass a saved design to the implementation agent without planning it again."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .complete_spec import (
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .planning_pipeline import PlanningPipeline
from .spec import ModSpec, Proposal, ProposalStatus
from .target_contract import TargetContractError, target_coordinates_from_mapping

_TARGET_KEYS = ("minecraft_version", "loader", "mappings")
_AUTHORED_EXECUTION_SCHEMA = "mmm/authored-execution-manifest-v2"
_GENERIC_AUTHORED_CONTAINER_TITLES = frozenset({
    "design",
    "design document",
    "game design",
    "spec",
    "specification",
    "requirements",
    "features",
    "feature design",
    "architecture",
    "implementation",
    "설계",
    "설계 문서",
    "게임 설계",
    "요구사항",
    "기능",
    "기능 목록",
    "아키텍처",
    "구현",
})


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _main_class_name(mod_id: str) -> str:
    """Match the canonical Fabric template provider's host-owned entrypoint name."""

    return "".join(part.capitalize() for part in str(mod_id).split("_")) + "Mod"


def _authored_heading_records(text: str) -> tuple[list[str], tuple[tuple[int, int, str], ...]]:
    """Parse Markdown headings outside fences while preserving exact source lines."""

    lines = text.splitlines(keepends=True)
    heading = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$")
    records: list[tuple[int, int, str]] = []
    fence = ""
    for index, line in enumerate(lines):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end():].strip()
            ):
                fence = ""
            continue
        if marker:
            fence = marker[1]
            continue
        match = heading.match(line)
        if match:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match[2]).strip("*_` ")
            records.append((index, len(match[1]), title))
    return lines, tuple(records)


def _generic_authored_container(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(title or "").strip()).casefold()
    return normalized in _GENERIC_AUTHORED_CONTAINER_TITLES


def _document_preamble_title(title: str) -> bool:
    """Recognize document-level wrappers without hard-coding a project name."""

    normalized = re.sub(r"\s+", " ", str(title or "").strip()).casefold()
    return bool(
        re.search(
            r"(?:^|\s)(?:game\s+)?(?:mod\s+)?design\s+document$"
            r"|(?:^|\s)design\s+spec(?:ification)?$"
            r"|(?:^|\s)requirements(?:\s+document)?$"
            r"|(?:^|\s)설계\s*문서$"
            r"|(?:^|\s)기획서$",
            normalized,
        )
    )


def _document_context_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(title or "").strip()).casefold()
    return normalized in {
        "intro",
        "introduction",
        "overview",
        "document overview",
        "project overview",
        "metadata",
        "project metadata",
        "summary",
        "about",
        "소개",
        "개요",
        "문서 개요",
        "프로젝트 개요",
        "메타데이터",
        "요약",
    }


def _metadata_only_preamble(lines: list[str], start: int, end: int) -> bool:
    """Return whether a leading section contains document metadata, not behavior."""

    meaningful = []
    for line in lines[start:end]:
        value = line.strip()
        if not value or re.fullmatch(r"[-*_]{3,}", value):
            continue
        if value.startswith("#"):
            continue
        meaningful.append(value)
    if not meaningful:
        return False
    metadata = re.compile(
        r"^(?:[-*+]\s+)?(?:\*\*)?[^:]{1,80}:(?:\*\*)?\s*\S.*$"
    )
    return all(metadata.match(value) for value in meaningful)


def _semantic_authored_blocks(text: str) -> tuple[str, ...]:
    """Split approved prose into semantic implementation units without losing bytes.

    Split points select implementation headings only. The actual slices are built as a
    lossless partition of the original line stream: every byte before the first selected
    feature is attached to that first feature, and every later block begins exactly at
    the preceding block's end. Document wrappers and metadata therefore remain context
    without ever becoming standalone AuthoredFeature tasks.
    """

    if not text:
        return ("",)
    lines, records = _authored_heading_records(text)
    if not records:
        return (text,)

    shallowest = min(depth for _index, depth, _title in records)
    split_level = shallowest
    shallow = [record for record in records if record[1] == shallowest]
    if (
        len(shallow) == 1
        and (
            _generic_authored_container(shallow[0][2])
            or _document_preamble_title(shallow[0][2])
        )
    ):
        for depth in sorted({record[1] for record in records if record[1] > shallowest}):
            peers = [record for record in records if record[1] == depth]
            if len(peers) >= 2:
                split_level = depth
                break

    split_records = [
        record for record in records if record[1] == split_level
    ]
    starts = [record[0] for record in split_records]
    if not starts:
        return (text,)

    if split_level > shallowest and len(starts) >= 2:
        record_by_start = {record[0]: record for record in split_records}
        while len(starts) >= 2:
            first_start = starts[0]
            first_end = starts[1]
            first_record = record_by_start.get(first_start)
            first_title = first_record[2] if first_record is not None else ""
            if not (
                _document_context_title(first_title)
                or _metadata_only_preamble(lines, first_start + 1, first_end)
            ):
                break
            starts = starts[1:]

    # At the document's own split level, a leading title/metadata section is context,
    # not an implementation unit. Remove only that boundary; slicing below will retain
    # every byte of the removed preamble in the first real feature block.
    if split_level == shallowest and len(starts) >= 2:
        first_start = starts[0]
        first_end = starts[1]
        first_title = split_records[0][2]
        if (
            _document_preamble_title(first_title)
            or _metadata_only_preamble(lines, first_start + 1, first_end)
        ):
            starts = starts[1:]

    if not starts:
        return (text,)

    heading_indexes = {index for index, _depth, _title in records}
    blocks: list[str] = []
    pending = ""
    for position, start_line in enumerate(starts):
        end_line = starts[position + 1] if position + 1 < len(starts) else len(lines)
        # The first implementation block owns *all* preceding source bytes. This is
        # what makes the partition lossless for leading prose, blank lines, BOM text,
        # document wrappers, metadata and horizontal rules.
        segment_start = 0 if position == 0 else start_line
        segment = "".join(lines[segment_start:end_line])
        feature_has_body = any(
            line.strip() and index not in heading_indexes
            for index, line in enumerate(lines[start_line:end_line], start_line)
        )
        if not feature_has_body:
            pending += segment
            continue
        blocks.append(pending + segment)
        pending = ""

    if pending:
        if blocks:
            blocks[-1] += pending
        else:
            blocks.append(pending)

    if "".join(blocks) != text:
        raise ValueError("Authored semantic block parsing changed approved design text.")
    return tuple(blocks)


def _authored_block_section(block: str) -> str:
    lines, records = _authored_heading_records(block)
    if not records:
        return ""
    first_index, first_level, first_title = records[0]
    if _generic_authored_container(first_title) or _document_preamble_title(first_title):
        descendants = [
            record for record in records[1:]
            if record[1] > first_level
        ]
        for position, (child_index, child_level, child_title) in enumerate(descendants):
            next_index = len(lines)
            for later_index, later_level, _later_title in descendants[position + 1:]:
                if later_level == child_level:
                    next_index = later_index
                    break
            if (
                _document_context_title(child_title)
                or _metadata_only_preamble(lines, child_index + 1, next_index)
            ):
                continue
            return child_title
        if descendants:
            return descendants[-1][2]
    if len(records) >= 2:
        second_index, second_level, second_title = records[1]
        if (
            second_level == first_level
            and (
                _document_preamble_title(first_title)
                or _metadata_only_preamble(lines, first_index + 1, second_index)
            )
        ):
            return second_title
    return first_title


def _implementation_authored_plan(plan: AuthoredPlan) -> tuple[AuthoredPlan, dict[str, Any] | None]:
    """Project a delimited final design without interpreting or rewriting its content.

    Legacy saved responses sometimes contain an explicitly labelled reasoning prefix.
    Only a leading envelope or a labelled prefix followed by a final-design boundary
    is recoverable here. Mentions in prose, examples and ambiguous drafts stay intact.
    """

    text = plan.text
    start = 0
    # These are response envelopes, not tags embedded in a design or fenced example.
    envelope = re.compile(
        r"\A\s*<(think|analysis)>.*?</\1>[ \t]*(?:\r?\n)*", re.IGNORECASE | re.DOTALL
    )
    while match := envelope.match(text[start:]):
        start += match.end()
    remainder = text[start:]
    labelled_reasoning = re.match(
        r"\A\s*(?:#{1,6}[ \t]+)?(?:\*\*)?"
        r"(?:thinking process|사고 과정|생각 과정)[ \t]*:?(?:\*\*)?[ \t]*\r?\n",
        remainder, re.IGNORECASE,
    )
    if labelled_reasoning is not None:
        # Ignore fenced examples when looking for the actual final document.
        fence = ""
        offset = 0
        for line in remainder.splitlines(keepends=True):
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if fence:
                if (marker and marker[1][0] == fence[0]
                        and len(marker[1]) >= len(fence)
                        and not line[marker.end():].strip()):
                    fence = ""
            elif marker:
                fence = marker[1]
            else:
                heading = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)\s*$", line)
                if heading:
                    title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[1]).strip("*_` ")
                    final_heading = re.fullmatch(
                        r"(?:behavior[ _-]+contract|행동[ _-]*계약|동작[ _-]*계약|"
                        r"(?:.+[ \t]+)?(?:design|설계)(?:[ \t]+(?:document|문서))?)",
                        title, re.IGNORECASE,
                    )
                    prefix = remainder[:offset].casefold()
                    has_analysis = any(token in prefix for token in (
                        "analyze the request", "deconstruct the template", "drafting content",
                        "요청 분석", "요청을 분석",
                    ))
                    if final_heading and has_analysis:
                        start += offset
                        break
            offset += len(line)
    if not start or not text[start:].strip():
        return plan, None
    prefix = text[:start]
    implementation_text = text[start:]
    projected = AuthoredPlan(
        requested_prompt=plan.requested_prompt,
        text=implementation_text,
        existing_input_sha256=plan.existing_input_sha256,
        media_paths=plan.media_paths,
    )
    provenance = {
        "schema_version": "mmm/authored-source-projection-v1",
        "policy": "strip_leaked_model_reasoning_prefix_only",
        # Keep the source recoverable; modules receive only the projected exact suffix.
        "source_plan": plan.to_dict(),
        "source_text_sha256": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "implementation_text_sha256": "sha256:"
        + hashlib.sha256(implementation_text.encode("utf-8")).hexdigest(),
        "stripped_prefix_bytes": len(prefix.encode("utf-8")),
    }
    return projected, provenance


def _authored_execution_units(text: str) -> tuple[dict[str, Any], ...]:
    """Lower saved prose to semantic Markdown-section obligations.

    Byte length never creates a new class or implementation task. Generic document
    wrappers may expose peer child features, but one semantic feature remains one unit
    even when its prose is long.
    """

    encoded = text.encode("utf-8")
    if not encoded:
        return ({
            "index": 1,
            "start_byte": 0,
            "end_byte": 0,
            "text": "",
            "section": "",
            "text_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
        },)

    chunks = [
        (block, _authored_block_section(block))
        for block in _semantic_authored_blocks(text)
    ]

    if "".join(chunk for chunk, _section in chunks) != text:
        raise ValueError("Authored execution lowering changed the approved design text.")

    units: list[dict[str, Any]] = []
    start = 0
    for index, (chunk, section) in enumerate(chunks, start=1):
        raw = chunk.encode("utf-8")
        end = start + len(raw)
        units.append({
            "index": index,
            "start_byte": start,
            "end_byte": end,
            "text": chunk,
            "section": section,
            "text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        })
        start = end
    return tuple(units)


def _task_sha(task: Mapping[str, Any]) -> str:
    payload = dict(task)
    payload.pop("task_sha256", None)
    return _sha256_json(payload)


def _exact_authored_task(
    *,
    task_id: str,
    path: str,
    symbol: str,
    target: Mapping[str, Any],
    obligation: str,
    semantic_outcome: str,
    depends_on: tuple[str, ...],
    consumes: tuple[str, ...],
    provides: tuple[str, ...],
    worksheet: Mapping[str, Any],
    required_gates: tuple[str, ...],
    target_status: str = "existing",
) -> dict[str, Any]:
    anchor = {
        "kind": "symbol",
        "locator": f"{path}#{symbol}",
        # Fresh authored projects materialize this exact skeleton before coder decode.
        # The coder therefore modifies an existing host-owned file and never chooses a path.
        "status": target_status,
        "ownership": "host_exact_authored_lowering",
        "module_id": task_id,
        "source_set": "main",
    }
    task: dict[str, Any] = {
        "task_id": task_id,
        "task_sha256": "",
        "execution_role": "coder",
        "semantic_outcome": semantic_outcome,
        "implementation_obligations": [obligation],
        "engineering_worksheet": dict(worksheet),
        "target_cell": dict(target),
        "owned_anchors": [anchor],
        "production_bindings": [{
            "task_ref": task_id,
            "reuse_action": "fresh",
            "owned_anchors": [dict(anchor)],
        }],
        "depends_on": list(depends_on),
        "consumes": list(consumes),
        "provides": list(provides),
        "required_gates": list(required_gates),
        "acceptance": [
            "Only the exact host-owned target is mutated.",
            "The generated Java passes host verification for the selected platform.",
        ],
    }
    task["task_sha256"] = _task_sha(task)
    return task


def _compile_new_authored_modules(
    plan: AuthoredPlan,
    *,
    mod_id: str,
    package_name: str,
    target: Mapping[str, Any],
) -> tuple[tuple[ProductionModule, ...], dict[str, Any]]:
    """Compile a saved design into exact-path tasks small coders can execute independently."""

    units = _authored_execution_units(plan.text)
    package_path = package_name.replace(".", "/")
    modules: list[ProductionModule] = []
    manifest_units: list[dict[str, Any]] = []
    previous_id = ""
    previous_provide = ""
    for unit in units:
        index = int(unit["index"])
        task_id = f"authored_feature_{index:03d}"
        symbol = f"AuthoredFeature{index:03d}"
        path = f"src/main/java/{package_path}/{symbol}.java"
        provide = f"{task_id}_ready"
        depends_on = (previous_id,) if previous_id else ()
        consumes = (previous_provide,) if previous_provide else ()
        exact_text = str(unit["text"])
        section = str(unit.get("section") or "").strip()
        target_summary = (
            f"Minecraft {target.get('minecraft_version', '')}, "
            f"loader {target.get('loader', '')}, mappings {target.get('mappings', '')}"
        )
        obligation = (
            f"Implement approved authored design unit {index}/{len(units)} only in "
            f"{symbol}. The exact class must be public final {symbol} in package "
            f"{package_name} and expose public static void initialize(). Do not put "
            "side-only annotations on that class or initialize(): the host invokes it on "
            "both client and server. Keep side-specific behavior in guarded helpers. "
            "Replace the MMM_AUTHORED_FEATURE_BODY marker with the approved behavior; "
            "a placeholder or initialization flag alone is not an implementation. Do not implement "
            "ModInitializer or ClientModInitializer, do not create another entrypoint, and "
            "do not create or edit sibling files. Additional helpers/state needed for this "
            "unit must stay inside this exact class. Earlier authored units, when present, are "
            "already compiled in the same staged workspace: inspect and reuse their public or "
            "package-visible API/state when this requirement depends on them instead of duplicating "
            "shared state. The host-selected target is authoritative "
            f"({target_summary}); adapt stale version/API examples in the authored prose to "
            "that target without changing gameplay semantics. Preserve the approved gameplay "
            "requirements in this unit as the semantic source of truth:\n\n" + exact_text
        )
        task = _exact_authored_task(
            task_id=task_id,
            path=path,
            symbol=symbol,
            target=target,
            obligation=obligation,
            semantic_outcome=(
                f"Approved authored design unit {index}/{len(units)} is implemented behind "
                f"{symbol}.initialize() without inventing project architecture."
            ),
            depends_on=depends_on,
            consumes=consumes,
            provides=(provide,),
            worksheet={
                "objective": "Implement exactly one host-scheduled authored design section.",
                "authored_unit": {
                    "index": index,
                    "section": section,
                    "count": len(units),
                    "source_text_sha256": unit["text_sha256"],
                    "start_byte": unit["start_byte"],
                    "end_byte": unit["end_byte"],
                    "text": exact_text,
                },
                "java_contract": {
                    "status": "applicable",
                    "requirements": [
                        f"Exact target: {path}#{symbol}",
                        f"Exact package: {package_name}",
                        f"Exact top-level type: public final class {symbol}",
                        "Required host integration surface: public static void initialize()",
                        "The feature class and initialize() must exist on both client and server; no side-only annotations on either.",
                        "Replace the host body marker with approved behavior; no placeholder-only implementation.",
                        "Forbidden: ModInitializer, ClientModInitializer, alternate entrypoints, sibling-file writes.",
                        "Do not require private implementation APIs from sibling feature classes; cross-feature activation is host-owned.",
                    ],
                },
            },
            required_gates=("target_compile",),
        )
        modules.append(ProductionModule(
            module_id=task_id,
            kind="custom_java",
            config={
                "implementation": "custom",
                "evidence_task": task,
                **dict(target),
            },
            depends_on=depends_on,
            required_gates=("target_compile",),
        ))
        manifest_units.append({
            "module_id": task_id,
            "path": path,
            "symbol": symbol,
            "start_byte": unit["start_byte"],
            "end_byte": unit["end_byte"],
            "text_sha256": unit["text_sha256"],
            "depends_on": list(depends_on),
            "consumes": list(consumes),
            "provides": provide,
            "section": section,
        })
        previous_id = task_id
        previous_provide = provide

    main_symbol = _main_class_name(mod_id)
    main_path = f"src/main/java/{package_path}/{main_symbol}.java"
    feature_symbols = [str(item["symbol"]) for item in manifest_units]

    source_sha = "sha256:" + hashlib.sha256(plan.text.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "source_text_sha256": source_sha,
        "source_bytes": len(plan.text.encode("utf-8")),
        "unit_count": len(manifest_units),
        "policy": "host_exact_task_queue_no_coder_file_planning",
        "units": manifest_units,
        "entrypoint": {
            "owner": "host_scaffold",
            "path": main_path,
            "symbol": main_symbol,
            # feature_symbols is the single host-owned integration source of truth.
            # Entry-point calls are derived from it during scaffold materialization.
            "feature_symbols": feature_symbols,
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return tuple(modules), manifest


def _compile_existing_authored_modules(
    plan: AuthoredPlan,
    *,
    target: Mapping[str, Any],
) -> tuple[tuple[ProductionModule, ...], dict[str, Any]]:
    """Split existing-project authored work into small semantic localization/edit tasks.

    Existing source files may be shared by multiple authored sections, so units are serialized.
    Each unit localizes a minimal exact target set immediately before editing.
    """

    units = _authored_execution_units(plan.text)
    modules: list[ProductionModule] = []
    manifest_units: list[dict[str, Any]] = []
    previous_id = ""
    for unit in units:
        index = int(unit["index"])
        task_id = f"authored_existing_{index:03d}"
        depends_on = (previous_id,) if previous_id else ()
        unit_plan = AuthoredPlan(
            requested_prompt=plan.requested_prompt,
            text=str(unit["text"]),
            existing_input_sha256=plan.existing_input_sha256,
            media_paths=plan.media_paths,
        )
        modules.append(
            ProductionModule(
                module_id=task_id,
                kind="custom_java",
                config={
                    "implementation": "custom",
                    "authored_plan": unit_plan.to_dict(),
                    "authored_localization_required": True,
                    "authored_unit": {
                        "index": index,
                        "count": len(units),
                        "section": str(unit.get("section") or ""),
                        "start_byte": int(unit["start_byte"]),
                        "end_byte": int(unit["end_byte"]),
                        "text_sha256": str(unit["text_sha256"]),
                    },
                    **dict(target),
                },
                depends_on=depends_on,
                required_gates=("target_compile",),
            )
        )
        manifest_units.append(
            {
                "module_id": task_id,
                "start_byte": int(unit["start_byte"]),
                "end_byte": int(unit["end_byte"]),
                "text_sha256": str(unit["text_sha256"]),
                "section": str(unit.get("section") or ""),
                "depends_on": list(depends_on),
            }
        )
        previous_id = task_id

    raw = plan.text.encode("utf-8")
    manifest: dict[str, Any] = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "source_text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "unit_count": len(manifest_units),
        "policy": "host_localize_freeze_exact_targets_before_coder",
        "units": manifest_units,
        "entrypoint": {
            "owner": "existing_project",
            "path": "",
            "symbol": "",
            "feature_symbols": [],
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return tuple(modules), manifest


def materialize_authored_execution_scaffold(
    proposal: CompleteProposal,
    project_root: Any,
) -> Any:
    """Materialize all fresh-authored architecture before the small coder is called.

    The host owns file names, package/type identity and the single Fabric entrypoint
    integration. The coder receives only already-existing exact feature files.
    """

    from pathlib import Path
    import re

    root = Path(project_root).expanduser().resolve()
    game_design = getattr(proposal, "game_design", None)
    design = game_design if isinstance(game_design, Mapping) else {}
    manifest = design.get("_authored_execution_manifest")
    if not isinstance(manifest, Mapping):
        return root
    if manifest.get("schema_version") != _AUTHORED_EXECUTION_SCHEMA:
        raise ValueError("AUTHORED_SCAFFOLD_SCHEMA_MISMATCH")
    if manifest.get("policy") != "host_exact_task_queue_no_coder_file_planning":
        raise ValueError("AUTHORED_SCAFFOLD_POLICY_MISMATCH")

    expected_manifest = dict(manifest)
    supplied_digest = str(expected_manifest.pop("manifest_sha256", "") or "")
    if supplied_digest != _sha256_json(expected_manifest):
        raise ValueError("AUTHORED_SCAFFOLD_MANIFEST_HASH_MISMATCH")

    package_name = proposal.base_proposal.spec.package_name
    package_path = package_name.replace(".", "/")
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("AUTHORED_SCAFFOLD_UNITS_MISSING")

    feature_symbols: list[str] = []
    for index, raw_unit in enumerate(units, start=1):
        if not isinstance(raw_unit, Mapping):
            raise ValueError("AUTHORED_SCAFFOLD_UNIT_INVALID")
        symbol = str(raw_unit.get("symbol") or "").strip()
        path = str(raw_unit.get("path") or "").replace("\\", "/").strip()
        expected_symbol = f"AuthoredFeature{index:03d}"
        expected_path = f"src/main/java/{package_path}/{expected_symbol}.java"
        if symbol != expected_symbol or path != expected_path:
            raise ValueError("AUTHORED_SCAFFOLD_UNIT_IDENTITY_DRIFT")
        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", symbol) is None:
            raise ValueError("AUTHORED_SCAFFOLD_SYMBOL_INVALID")
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("AUTHORED_SCAFFOLD_PATH_ESCAPE") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.is_symlink():
                raise ValueError("AUTHORED_SCAFFOLD_TARGET_NOT_REGULAR")
            source = target.read_text(encoding="utf-8")
            if (
                f"package {package_name};" not in source
                or re.search(
                    rf"\bclass\s+{re.escape(symbol)}\b",
                    source,
                )
                is None
            ):
                raise ValueError("AUTHORED_SCAFFOLD_EXISTING_IDENTITY_MISMATCH")
        else:
            target.write_text(
                (
                    f"package {package_name};\n\n"
                    f"/** Host-owned authored feature slot {index}/{len(units)}. */\n"
                    f"public final class {symbol} {{\n"
                    f"    private {symbol}() {{}}\n\n"
                    "    public static void initialize() {\n"
                    f"        // MMM_AUTHORED_FEATURE_BODY_{index:03d}\n"
                    "    }\n"
                    "}\n"
                ),
                encoding="utf-8",
                newline="\n",
            )
        feature_symbols.append(symbol)

    entry = manifest.get("entrypoint")
    if not isinstance(entry, Mapping):
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_MISSING")
    main_symbol = _main_class_name(proposal.base_proposal.spec.mod_id)
    main_path = f"src/main/java/{package_path}/{main_symbol}.java"
    if str(entry.get("symbol") or "") != main_symbol or str(
        entry.get("path") or ""
    ).replace("\\", "/") != main_path:
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_IDENTITY_DRIFT")
    if list(entry.get("feature_symbols") or ()) != feature_symbols:
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_FEATURE_DRIFT")

    main_source = (root / main_path).resolve()
    try:
        main_source.relative_to(root)
    except ValueError as exc:
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_ESCAPE") from exc
    if not main_source.is_file() or main_source.is_symlink():
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_SOURCE_MISSING")

    main_text = main_source.read_text(encoding="utf-8")
    marker = "// MMM_AUTHORED_HOST_ENTRYPOINT_BINDING"
    calls = [f"{symbol}.initialize();" for symbol in feature_symbols]
    if marker in main_text:
        if any(main_text.count(call) != 1 for call in calls):
            raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_BINDING_CORRUPT")
        return root

    if any(call in main_text for call in calls):
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_UNMARKED_BINDING")

    match = re.search(
        r"public\s+void\s+onInitialize\s*\(\s*\)\s*\{",
        main_text,
    )
    if match is None:
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_METHOD_MISSING")
    injection = (
        match.group(0)
        + "\n        "
        + marker
        + "\n"
        + "\n".join(f"        {call}" for call in calls)
    )
    main_text = main_text[: match.start()] + injection + main_text[match.end() :]
    main_source.write_text(main_text, encoding="utf-8", newline="\n")
    return root


def _bound_target(design: Mapping[str, Any]) -> dict[str, str]:
    """Return the complete host-selected target, or no target when none exists.

    The platform selector owns the target. Decode its receipt through the same
    contract used by generation, including native names and mapping receipt objects.
    Older saved designs without a selection may still carry standalone coordinates.
    """
    candidates: list[Mapping[str, Any]] = [design]
    for key in ("platform", "target", "toolchain", "build", "existing_project"):
        value = design.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)

    selection = design.get("_platform_selection")
    if isinstance(selection, Mapping) and "target" in selection:
        target = selection["target"]
        if not isinstance(target, Mapping):
            raise ValueError("Saved authored platform selection target must be an object.")
        # Never let stale design/existing-project fields override the selected target,
        # including when the selected receipt is invalid.
        coordinates = target_coordinates_from_mapping(target)
        return {key: getattr(coordinates, key) for key in _TARGET_KEYS}
    if isinstance(selection, Mapping):
        candidates.append(selection)

    first_error: TargetContractError | None = None
    for candidate in candidates:
        if not any(candidate.get(key) not in (None, "") for key in (
            *_TARGET_KEYS, "mappings_version", "yarn_mappings",
        )):
            continue
        try:
            coordinates = target_coordinates_from_mapping(candidate)
        except TargetContractError as exc:
            if first_error is None:
                first_error = exc
            continue
        return {key: getattr(coordinates, key) for key in _TARGET_KEYS}

    if first_error is not None:
        raise first_error
    return {}


def compile_authored_design(
    router: Any, plan: AuthoredPlan, *, existing_input_sha256: str = ""
) -> CompleteProposal:
    implementation_plan, source_projection = _implementation_authored_plan(plan)
    # These are host project coordinates, not inferred gameplay or placeholder content.
    mod_id = "authored_" + implementation_plan.calculate_hash()[:12]
    acceptance = (
        "Implement the behaviors in the saved authored design and exercise them in Minecraft.",
        "Build the project and verify that the mod loads and runs without errors.",
    )
    base = Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=ProposalStatus.AWAITING_APPROVAL,
        requested_prompt=plan.requested_prompt,
        spec=ModSpec(
            mod_id=mod_id,
            mod_name="Authored Minecraft Mod",
            package_name=f"ai.minecraft.generated.{mod_id}",
            version="1.0.0",
            summary=plan.requested_prompt,
            contents=(),
        ),
        assumptions=(), exclusions=(), deferred_requests=(),
        acceptance_tests=acceptance, evidence_sources=(),
    )
    design = {"authored_plan": implementation_plan.to_dict()}
    if source_projection is not None:
        design["_authored_source_projection"] = source_projection
    # Bind the actual build toolchain and existing project only. Never enter prepare(),
    # requirement extraction, design validation, or the old PlanIR compiler.
    binding = PlanningPipeline(router)
    design = binding._bind_existing_project(design)
    design, base, _, _ = binding._bind_platform(plan.requested_prompt, design, base)

    # The saved-plan route previously preserved the binding only in game_design while
    # its production module dropped the target triple. Official RAG validates the
    # production-side host contract, so make the bound target explicit at that boundary.
    target = _bound_target(design)
    design = {**design, **target}
    effective_existing = existing_input_sha256 or plan.existing_input_sha256
    if not effective_existing:
        # Fresh authored projects have a host-owned canonical Fabric package/entrypoint.
        # Lower the saved prose into an exact-path dependency queue now, before coder
        # decode, so the small model never owns file planning or entrypoint architecture.
        modules, manifest = _compile_new_authored_modules(
            implementation_plan,
            mod_id=base.spec.mod_id,
            package_name=base.spec.package_name,
            target=target,
        )
        design = {**design, "_authored_execution_manifest": manifest}
    else:
        modules, manifest = _compile_existing_authored_modules(
            implementation_plan,
            target=target,
        )
        design = {**design, "_authored_execution_manifest": manifest}

    return complete_proposal_from_parts(
        requested_prompt=plan.requested_prompt,
        base_proposal=base,
        game_design=design,
        modules=modules,
        acceptance_tests=acceptance,
        existing_input_sha256=effective_existing,
    )
