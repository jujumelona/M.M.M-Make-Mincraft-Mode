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
    "systems",
    "gameplay systems",
    "core systems",
    "mechanics",
    "gameplay mechanics",
    "architecture",
    "implementation",
    "설계",
    "설계 문서",
    "게임 설계",
    "요구사항",
    "기능",
    "기능 목록",
    "시스템",
    "게임플레이 시스템",
    "핵심 시스템",
    "메커니즘",
    "게임플레이 메커니즘",
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


def _heading_peer_is_context(
    lines: list[str],
    peers: list[tuple[int, int, str]],
    position: int,
) -> bool:
    start_line, _level, title = peers[position]
    end_line = peers[position + 1][0] if position + 1 < len(peers) else len(lines)
    return bool(
        _document_context_title(title)
        or _metadata_only_preamble(lines, start_line + 1, end_line)
    )


def _document_wrapper_split_level(
    lines: list[str],
    records: tuple[tuple[int, int, str], ...],
    shallowest: int,
) -> int:
    for depth in sorted({record[1] for record in records if record[1] > shallowest}):
        peers = [record for record in records if record[1] == depth]
        actionable = [
            record
            for position, record in enumerate(peers)
            if not _heading_peer_is_context(lines, peers, position)
        ]
        if len(actionable) >= 2:
            return depth
        if len(actionable) == 1 and not _generic_authored_container(actionable[0][2]):
            return depth
    return shallowest


def _trim_leading_authored_context(
    lines: list[str],
    split_records: list[tuple[int, int, str]],
    *,
    split_level: int,
    shallowest: int,
) -> list[int]:
    starts = [record[0] for record in split_records]
    if split_level > shallowest:
        while len(starts) >= 2:
            first_position = next(
                (
                    position
                    for position, record in enumerate(split_records)
                    if record[0] == starts[0]
                ),
                None,
            )
            if first_position is None or not _heading_peer_is_context(
                lines, split_records, first_position
            ):
                break
            starts = starts[1:]
        return starts

    if len(starts) < 2:
        return starts
    first_start, first_end = starts[:2]
    first_title = split_records[0][2]
    if (
        _document_preamble_title(first_title)
        or _metadata_only_preamble(lines, first_start + 1, first_end)
    ):
        return starts[1:]
    return starts


def _lossless_authored_partition(
    lines: list[str],
    records: tuple[tuple[int, int, str], ...],
    starts: list[int],
) -> tuple[str, ...]:
    if not starts:
        return ("".join(lines),)

    heading_indexes = {index for index, _depth, _title in records}
    blocks: list[str] = []
    pending = ""
    for position, start_line in enumerate(starts):
        end_line = starts[position + 1] if position + 1 < len(starts) else len(lines)
        segment_start = 0 if position == 0 else start_line
        segment = "".join(lines[segment_start:end_line])
        feature_has_body = any(
            line.strip() and index not in heading_indexes
            for index, line in enumerate(lines[start_line:end_line], start_line)
        )
        if feature_has_body:
            blocks.append(pending + segment)
            pending = ""
        else:
            pending += segment

    if pending:
        if blocks:
            blocks[-1] += pending
        else:
            blocks.append(pending)
    return tuple(blocks)


def _semantic_authored_blocks(text: str) -> tuple[str, ...]:
    """Split approved prose into semantic implementation units without losing bytes."""

    if not text:
        return ("",)
    lines, records = _authored_heading_records(text)
    if not records:
        return (text,)

    shallowest = min(depth for _index, depth, _title in records)
    shallow = [record for record in records if record[1] == shallowest]
    document_wrapper = bool(
        len(shallow) == 1
        and (
            _generic_authored_container(shallow[0][2])
            or _document_preamble_title(shallow[0][2])
        )
    )
    split_level = (
        _document_wrapper_split_level(lines, records, shallowest)
        if document_wrapper
        else shallowest
    )
    split_records = [record for record in records if record[1] == split_level]
    starts = _trim_leading_authored_context(
        lines,
        split_records,
        split_level=split_level,
        shallowest=shallowest,
    )
    blocks = _lossless_authored_partition(lines, records, starts)
    if "".join(blocks) != text:
        raise ValueError("Authored semantic block parsing changed approved design text.")
    return blocks


def _first_authored_feature_title(
    lines: list[str],
    records: tuple[tuple[int, int, str], ...],
    first_level: int,
) -> str:
    descendants = [record for record in records[1:] if record[1] > first_level]
    for position, (child_index, _child_level, child_title) in enumerate(descendants):
        next_index = (
            descendants[position + 1][0]
            if position + 1 < len(descendants)
            else len(lines)
        )
        if (
            _document_context_title(child_title)
            or _generic_authored_container(child_title)
            or _metadata_only_preamble(lines, child_index + 1, next_index)
        ):
            continue
        return child_title
    return descendants[-1][2] if descendants else ""


def _authored_block_section(block: str) -> str:
    lines, records = _authored_heading_records(block)
    if not records:
        return ""

    first_index, first_level, first_title = records[0]
    if _generic_authored_container(first_title) or _document_preamble_title(first_title):
        descendant_title = _first_authored_feature_title(lines, records, first_level)
        if descendant_title:
            return descendant_title

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


def _authored_block_implementation_text(block: str, section: str) -> str:
    """Return only the executable feature slice while retaining block provenance elsewhere."""

    wanted = str(section or "").strip()
    if not block or not wanted:
        return block
    lines, records = _authored_heading_records(block)
    for start_line, _level, title in records:
        if title == wanted:
            return "".join(lines[start_line:])
    return block


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
                        r"(?:.+[ \t]+)?(?:design|설계)(?:[ \t]+(?:document|문서)|서)?)",
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
            "implementation_text": "",
            "section": "",
            "text_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
            "implementation_text_sha256": "sha256:" + hashlib.sha256(b"").hexdigest(),
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
        implementation_text = _authored_block_implementation_text(chunk, section)
        units.append({
            "index": index,
            "start_byte": start,
            "end_byte": end,
            "text": chunk,
            "implementation_text": implementation_text,
            "section": section,
            "text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "implementation_text_sha256": "sha256:"
            + hashlib.sha256(implementation_text.encode("utf-8")).hexdigest(),
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
    """Reserve graph compilation, not Java files derived from Markdown sections.

    Semantic lowering runs with the production router and actual project context,
    before the first source decode. The saved document remains exact and recoverable.
    """
    main_symbol = _main_class_name(mod_id)
    main_path = f"src/main/java/{package_name.replace('.', '/')}/{main_symbol}.java"
    task_id = "authored_implementation_graph"
    request = {
        "text": plan.text, "package": package_name, "mod_id": mod_id,
        "target": dict(target), "entrypoint_path": main_path,
        "entrypoint_symbol": main_symbol,
    }
    task = _exact_authored_task(
        task_id=task_id, path=main_path, symbol=main_symbol, target=target,
        obligation="Compile the complete saved design into a responsibility/dependency IR, then execute admitted source units.",
        semantic_outcome="Implement the saved design through budgeted collaborating source units.",
        depends_on=(), consumes=(), provides=("authored_implementation_ready",),
        worksheet={"implementation_graph_request": request}, required_gates=("target_compile",),
    )
    module = ProductionModule(
        module_id=task_id, kind="custom_java",
        config={"implementation": "custom", "evidence_task": task,
                "implementation_graph_request": request, **dict(target)},
        required_gates=("target_compile",),
    )
    manifest = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "policy": "host_implementation_graph_before_source",
        "source_text_sha256": "sha256:" + hashlib.sha256(plan.text.encode()).hexdigest(),
        "source_bytes": len(plan.text.encode()), "unit_count": 0,
        "units": [], "graph_status": "pending_production_context",
        "entrypoint": {"owner": "host_scaffold", "path": main_path,
                       "symbol": main_symbol, "feature_symbols": []},
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return (module,), manifest


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

    import re
    from pathlib import Path

    root = Path(project_root).expanduser().resolve()
    game_design = getattr(proposal, "game_design", None)
    design = game_design if isinstance(game_design, Mapping) else {}
    manifest = design.get("_authored_execution_manifest")
    if not isinstance(manifest, Mapping):
        return root
    if manifest.get("schema_version") != _AUTHORED_EXECUTION_SCHEMA:
        raise ValueError("AUTHORED_SCAFFOLD_SCHEMA_MISMATCH")
    policy = str(manifest.get("policy") or "")
    if policy not in {
        "host_exact_task_queue_no_coder_file_planning",
        "host_bounded_coherent_authored_design",
        "host_implementation_graph_before_source",
    }:
        raise ValueError("AUTHORED_SCAFFOLD_POLICY_MISMATCH")

    expected_manifest = dict(manifest)
    supplied_digest = str(expected_manifest.pop("manifest_sha256", "") or "")
    if supplied_digest != _sha256_json(expected_manifest):
        raise ValueError("AUTHORED_SCAFFOLD_MANIFEST_HASH_MISMATCH")

    if policy in {"host_bounded_coherent_authored_design", "host_implementation_graph_before_source"}:
        # The canonical Fabric template is already materialized by the host. Coherent
        # authored generation owns architecture inside bounded package/resource roots,
        # so creating synthetic AuthoredFeatureNNN placeholders here would reintroduce
        # the document-section-equals-class bug.
        return root

    package_name = proposal.base_proposal.spec.package_name
    package_path = package_name.replace(".", "/")
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("AUTHORED_SCAFFOLD_UNITS_MISSING")

    feature_symbols: list[str] = []
    for index, raw_unit in enumerate(units, start=1):
        if not isinstance(raw_unit, Mapping):
            raise ValueError("AUTHORED_SCAFFOLD_UNIT_INVALID")  # noqa: TRY004 - persisted manifest error contract
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
        raise ValueError("AUTHORED_SCAFFOLD_ENTRYPOINT_MISSING")  # noqa: TRY004 - persisted manifest error contract
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


def _normalized_contract_heading(title: str) -> str:
    return re.sub(r"[\s-]+", "_", str(title or "").strip().casefold()).strip("_")


def _contract_shaped_authored_design(text: str) -> bool:
    """Recognize an engineering worksheet whose headings are facets, not features."""

    from .planning_detail_template import WORKSHEET_SECTIONS

    _lines, records = _authored_heading_records(text)
    if not records:
        return False
    canonical = set(WORKSHEET_SECTIONS)
    for depth in sorted({record[1] for record in records}):
        names = tuple(
            _normalized_contract_heading(title)
            for _index, record_depth, title in records
            if record_depth == depth and not _document_context_title(title)
        )
        contract_names = tuple(name for name in names if name in canonical)
        # Supplementary prose/overview headings do not turn engineering facets
        # into independent features. Repeated facets indicate multiple contracts.
        if len(set(contract_names)) >= 3 and len(contract_names) == len(set(contract_names)):
            return True
    return False


def _compile_coherent_authored_module(
    plan: AuthoredPlan,
    *,
    mod_id: str,
    package_name: str,
    target: Mapping[str, Any],
) -> tuple[tuple[ProductionModule, ...], dict[str, Any]]:
    """Keep one engineering contract coherent and let the coder choose bounded files."""

    raw = plan.text.encode("utf-8")
    module_id = "authored_design"
    main_symbol = _main_class_name(mod_id)
    main_path = f"src/main/java/{package_name.replace('.', '/')}/{main_symbol}.java"
    module = ProductionModule(
        module_id=module_id,
        kind="custom_java",
        config={
            "implementation": "custom",
            "authored_plan": plan.to_dict(),
            "authored_execution_mode": "bounded_coherent",
            "authored_bounded_scope": True,
            "authored_java_package": package_name,
            "authored_mod_id": mod_id,
            **dict(target),
        },
        required_gates=("project build",),
    )
    unit = {
        "module_id": module_id,
        "start_byte": 0,
        "end_byte": len(raw),
        "text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }
    manifest: dict[str, Any] = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "source_text_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "unit_count": 1,
        "policy": "host_bounded_coherent_authored_design",
        "units": [unit],
        "entrypoint": {
            "owner": "host_template",
            "path": main_path,
            "symbol": main_symbol,
            "feature_symbols": [],
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return (module,), manifest


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
        # Reserve semantic graph compilation in the actual production context.
        # No document section is assumed to be a class or executable source unit.
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

    from .root_cause_trace import emit_root_cause

    emit_root_cause(
        "authored_design_lowering_selected", stage="production",
        operation="compile_authored_production", gate="authored_execution_route", result="PASS",
        details={
            "policy": manifest["policy"], "existing_input": bool(effective_existing),
            "headings": [
                {"line": line + 1, "depth": depth, "title": title}
                for line, depth, title in _authored_heading_records(implementation_plan.text)[1]
            ],
            "module_ids": [module.module_id for module in modules],
            "source_text_sha256": manifest["source_text_sha256"],
            "manifest": manifest,
        },
    )
    return complete_proposal_from_parts(
        requested_prompt=plan.requested_prompt,
        base_proposal=base,
        game_design=design,
        modules=modules,
        acceptance_tests=acceptance,
        existing_input_sha256=effective_existing,
    )
