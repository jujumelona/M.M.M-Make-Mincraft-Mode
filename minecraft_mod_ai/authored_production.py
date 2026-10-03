"""Pass a saved design to the implementation agent without planning it again."""

from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import re
from collections.abc import Mapping
from typing import Any

from .authored_plan import AuthoredPlan
from .authored_structured_design import structured_sections_sha256
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


def _orphan_reasoning_close_projection_start(text: str) -> int:
    """Recover a final authored suffix after a standalone leaked reasoning close tag.

    Some model transports can drop the opening <think>/<analysis> token while preserving
    the closing token and the final answer. Only treat that as an envelope when the close
    tag is a standalone line outside Markdown fences and the suffix contains a substantial
    canonical authored contract. This keeps quoted/inline tags in user-authored prose intact.
    """

    from .authored_ir_parser import authored_section_id

    lines = str(text or "").splitlines(keepends=True)
    fence = ""
    offset = 0
    candidates: list[int] = []
    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not line[marker.end():].strip()
            ):
                fence = ""
            offset += len(line)
            continue
        if marker:
            fence = marker[1]
            offset += len(line)
            continue
        if re.fullmatch(r"\s*</(?:think|analysis)>\s*(?:\r?\n)?", line, re.IGNORECASE):
            candidates.append(offset + len(line))
        offset += len(line)

    for raw_start in reversed(candidates):
        start = raw_start
        while start < len(text) and text[start] in " \t\r\n":
            start += 1
        suffix = text[start:]
        if not suffix:
            continue
        sections = [
            authored_section_id(title)
            for _line, _depth, title in _authored_heading_records(suffix)[1]
        ]
        canonical = [section for section in sections if section]
        distinct = tuple(dict.fromkeys(canonical))
        if (
            len(distinct) >= 4
            and "behavior_contract" in distinct
            and "state_model" in distinct
        ):
            return start
    return 0

def _compile_new_authored_modules(
    plan: AuthoredPlan,
    *,
    mod_id: str,
    package_name: str,
    target: Mapping[str, Any],
    production_state_section: Mapping[str, Any] | None = None,
) -> tuple[tuple[ProductionModule, ...], dict[str, Any]]:
    """Lower the persisted Typed PlanIR through deterministic host backends only."""
    main_symbol = _main_class_name(mod_id)
    main_path = f"src/main/java/{package_name.replace('.', '/')}/{main_symbol}.java"

    if not getattr(plan, "typed_plan_ir", {}):
    raise ValueError(
        "TYPED_PLAN_REQUIRED: untyped authored production has been removed."
    )
    from .typed_plan_ir import (
        typed_plan_capability_ids,
        typed_plan_uses_state,
        validate_typed_plan_ir,
    )
    from .typed_plan_support import assert_typed_plan_host_support

    source_sha = "sha256:" + hashlib.sha256(plan.text.encode("utf-8")).hexdigest()
    stored_sha = str(plan.typed_plan_ir.get("source_sha256") or "")
    normalized_stored_sha = (
        stored_sha if stored_sha.startswith("sha256:") else "sha256:" + stored_sha
    )
    if normalized_stored_sha != source_sha:
        raise ValueError("TYPED_PLAN_SOURCE_HASH_MISMATCH")

    from .typed_host_capabilities import (
        typed_host_capability_contracts,
    )

    capability_contracts = typed_host_capability_contracts()
    validated_plan = validate_typed_plan_ir(
        plan.typed_plan_ir,
        capabilities=capability_contracts,
    )
    capability_ids = typed_plan_capability_ids(validated_plan)
    bound_capabilities = {
        capability_id: deepcopy(capability_contracts[capability_id])
        for capability_id in capability_ids
    }
    assert_typed_plan_host_support(
        plan.structured_sections,
        validated_plan,
    )

    raw_platform_modules = validated_plan.get("platform_modules", [])
    platform_modules: list[ProductionModule] = []
    state_store_config: dict[str, Any] | None = None
    network_sync_config: dict[str, Any] | None = None
    resource_policy_config: dict[str, Any] | None = None
    for item in raw_platform_modules:
        module_id = str(item["module_id"])
        kind = str(item["kind"])
        if module_id == "authored_typed_plan":
            raise ValueError(
                "TYPED_PLATFORM_MODULE_ID_CONFLICT: authored_typed_plan"
            )
        if kind == "state_store":
            if state_store_config is not None:
                raise ValueError(
                    "TYPED_PLATFORM_STATE_STORE_DUPLICATE"
                )
            state_store_config = deepcopy(dict(item["config"]))
            continue
        if kind == "network_sync":
            if network_sync_config is not None:
                raise ValueError("TYPED_PLATFORM_NETWORK_SYNC_DUPLICATE")
            network_sync_config = {
                **deepcopy(dict(item["config"])),
                "__covers": list(item["covers"]),
            }
            continue
        if kind == "resource_policy":
            if resource_policy_config is not None:
                raise ValueError("TYPED_PLATFORM_RESOURCE_POLICY_DUPLICATE")
            resource_policy_config = deepcopy(dict(item["config"]))
            continue
        platform_modules.append(
            ProductionModule(
                module_id=module_id,
                kind=kind,
                config=deepcopy(dict(item["config"])),
                required_gates=("target_compile",),
            )
        )

    from .authored_structured_design import active_concern_records

    state_section = (
        deepcopy(dict(production_state_section))
        if isinstance(production_state_section, Mapping)
        else {}
    )
    active_state = active_concern_records(
        plan.structured_sections,
        "state_model",
    )
    network_sync_covers = set(
        network_sync_config.get("__covers", ())
        if isinstance(network_sync_config, Mapping)
        else ()
    )
    network_sync_needs_state = bool(
        network_sync_covers
        & {
            "authority_and_network.payloads",
            "authority_and_network.synchronization",
            "authority_and_network.reconnection",
        }
    )
    state_required = (
        typed_plan_uses_state(validated_plan)
        or state_store_config is not None
        or network_sync_needs_state
        or any(bool(rows) for rows in active_state.values())
    )
    if state_required and not state_section:
        raise ValueError(
            "TYPED_PLAN_STATE_AUTHORITY_REQUIRED: active state semantics, "
            "state operations, or persistent state require canonical "
            "structured state_model authority."
        )

    program_symbol = "AuthoredProgram"
    program_path = (
        f"src/main/java/{package_name.replace('.', '/')}/{program_symbol}.java"
    )
    task_id = "authored_typed_plan"
    task = _exact_authored_task(
        task_id=task_id,
        path=program_path,
        symbol=program_symbol,
        target=target,
        obligation=(
            "Compile the persisted Typed PlanIR exactly through the host Java backend. "
            "Do not invoke a coder or reinterpret authored behavior."
        ),
        semantic_outcome="Materialize the approved Typed PlanIR deterministically.",
        depends_on=(),
        consumes=(),
        provides=("authored_typed_program_ready",),
        worksheet={
            "typed_plan_ir": deepcopy(validated_plan),
            "typed_plan_source_sha256": source_sha,
        },
        required_gates=("target_compile",),
        target_status="host_reserved",
        execution_role="host_compiler",
    )
    module = ProductionModule(
        module_id=task_id,
        kind="custom_java",
        config={
            "implementation": "custom",
            "evidence_task": task,
            "typed_plan_ir": deepcopy(validated_plan),
            "typed_plan_package": package_name,
            "typed_plan_path": program_path,
            "typed_plan_capabilities": deepcopy(bound_capabilities),
            "typed_plan_state_section": state_section,
            "typed_plan_structured_sections": deepcopy(plan.structured_sections),
            "typed_state_store": state_store_config,
            "typed_network_sync": network_sync_config,
            "typed_resource_policy": resource_policy_config,
            **dict(target),
        },
        required_gates=("target_compile",),
    )
    manifest = {
        "schema_version": _AUTHORED_EXECUTION_SCHEMA,
        "policy": "host_typed_plan_ir",
        "source_text_sha256": source_sha,
        "source_bytes": len(plan.text.encode("utf-8")),
        "unit_count": 1 + len(platform_modules),
        "units": [{
            "module_id": task_id,
            "path": program_path,
            "symbol": program_symbol,
            "source_sha256": source_sha,
        }],
        "graph_status": "not_required",
        "typed_program": {
            "path": program_path,
            "symbol": program_symbol,
            "state_required": state_required,
            "state_store": state_store_config is not None,
            "network_sync": network_sync_config is not None,
            "resource_policy": resource_policy_config is not None,
            "capabilities": list(capability_ids),
        },
        "platform_modules": [
            {
                "module_id": str(item["module_id"]),
                "kind": str(item["kind"]),
                "covers": list(item["covers"]),
            }
            for item in raw_platform_modules
        ],
        "entrypoint": {
            "owner": "host_scaffold",
            "path": main_path,
            "symbol": main_symbol,
            "feature_symbols": [program_symbol],
        },
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return (module, *platform_modules), manifest


def materialize_authored_execution_scaffold(
    proposal: CompleteProposal,
    project_root: Any,
) -> Any:
    """Materialize the deterministic Typed PlanIR scaffold only."""

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
    if policy != "host_typed_plan_ir":
        raise ValueError(
            "AUTHORED_SCAFFOLD_POLICY_MISMATCH: only Typed PlanIR is supported"
        )

    expected_manifest = dict(manifest)
    supplied_digest = str(expected_manifest.pop("manifest_sha256", "") or "")
    if supplied_digest != _sha256_json(expected_manifest):
        raise ValueError("AUTHORED_SCAFFOLD_MANIFEST_HASH_MISMATCH")

    from .production_state_compiler import render_production_state_java
    from .project_edit import (
        ensure_main_initializer_call,
        inspect_fabric_project,
        write_text_files,
    )

    info = inspect_fabric_project(root)
    package_name = proposal.base_proposal.spec.package_name
    if info.package_name != package_name:
        raise ValueError(
            "AUTHORED_TYPED_PACKAGE_MISMATCH: "
            f"{info.package_name!r} != {package_name!r}"
        )

    typed_program = manifest.get("typed_program")
    if not isinstance(typed_program, Mapping):
        raise ValueError("AUTHORED_TYPED_PROGRAM_MANIFEST_MISSING")
    program_path = str(typed_program.get("path") or "").replace("\\", "/").strip()
    expected_program_path = (
        f"src/main/java/{package_name.replace('.', '/')}/AuthoredProgram.java"
    )
    if program_path != expected_program_path:
        raise ValueError("AUTHORED_TYPED_PROGRAM_PATH_DRIFT")

    program_target = root / program_path
    if program_target.exists():
        if not program_target.is_file() or program_target.is_symlink():
            raise ValueError("AUTHORED_TYPED_PROGRAM_TARGET_INVALID")
        current_program = program_target.read_text(encoding="utf-8")
        if "// MMM:TYPED_PLAN_OWNER" not in current_program:
            raise ValueError("AUTHORED_TYPED_PROGRAM_OWNERSHIP_CONFLICT")
    else:
        placeholder = (
            f"package {package_name};\n\n"
            "// MMM:TYPED_PLAN_OWNER\n"
            "public final class AuthoredProgram {\n"
            "    private AuthoredProgram() {}\n"
            "    public static void initialize() {}\n"
            "}\n"
        )
        write_text_files(
            info,
            {program_path: placeholder},
            replace_existing=False,
        )

    if bool(typed_program.get("state_required")):
        raw_state = design.get("_production_state_section")
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise ValueError("AUTHORED_TYPED_STATE_AUTHORITY_MISSING")
        state_path = (
            f"src/main/java/{package_name.replace('.', '/')}/"
            "AuthoredStateModel.java"
        )
        state_target = root / state_path
        replace_state = False
        if state_target.exists():
            if not state_target.is_file() or state_target.is_symlink():
                raise ValueError("AUTHORED_TYPED_STATE_TARGET_INVALID")
            current_state = state_target.read_text(encoding="utf-8")
            if "// MMM:TYPED_PLAN_STATE_OWNER" not in current_state:
                raise ValueError("AUTHORED_TYPED_STATE_OWNERSHIP_CONFLICT")
            replace_state = True
        state_source = render_production_state_java(
            raw_state,
            package_name=package_name,
        )
        write_text_files(
            info,
            {state_path: state_source},
            replace_existing=replace_state,
        )

    ensure_main_initializer_call(
        info,
        import_line=f"import {package_name}.AuthoredProgram",
        call_line="AuthoredProgram.initialize()",
        marker="typed-plan",
    )
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


def compile_authored_design(
    router: Any, plan: AuthoredPlan, *, existing_input_sha256: str = ""
) -> CompleteProposal:
    from .authored_structured_design import normalize_structured_sections
    from .production_state_compiler import compile_production_state_section

    if not isinstance(plan, AuthoredPlan):
        raise TypeError("compile_authored_design requires AuthoredPlan")
    if not getattr(plan, "typed_plan_ir", {}):
        raise ValueError(
            "TYPED_PLAN_REQUIRED: legacy authored production routes have been removed."
        )

    structured = normalize_structured_sections(plan.structured_sections)
    if not structured:
        raise ValueError(
            "TYPED_STRUCTURED_AUTHORITY_REQUIRED: canonical structured design is required."
        )
    if structured != plan.structured_sections:
        raise ValueError(
            "TYPED_STRUCTURED_AUTHORITY_NONCANONICAL: planning must persist canonical records."
        )

    effective_existing = str(
        existing_input_sha256 or plan.existing_input_sha256 or ""
    ).strip()

    mod_id = "authored_" + plan.calculate_hash()[:12]
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
        assumptions=(),
        exclusions=(),
        deferred_requests=(),
        acceptance_tests=acceptance,
        evidence_sources=(),
    )

    design = {"authored_plan": plan.to_dict()}
    binding = PlanningPipeline(router)
    design = binding._bind_existing_project(design)
    design, base, _, _ = binding._bind_platform(
        plan.requested_prompt,
        design,
        base,
    )
    target = _bound_target(design)
    design = {**design, **target}

    production_state_section = compile_production_state_section(plan)
    modules, manifest = _compile_new_authored_modules(
        plan,
        mod_id=base.spec.mod_id,
        package_name=base.spec.package_name,
        target=target,
        production_state_section=production_state_section,
    )
    design = {
        **design,
        "_authored_execution_manifest": manifest,
        "_production_state_section": deepcopy(production_state_section),
    }

    from .root_cause_trace import emit_root_cause

    emit_root_cause(
        "authored_design_lowering_selected",
        stage="production",
        operation="compile_authored_production",
        gate="typed_host_execution_only",
        result="PASS",
        details={
            "policy": manifest["policy"],
            "existing_input": bool(effective_existing),
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
