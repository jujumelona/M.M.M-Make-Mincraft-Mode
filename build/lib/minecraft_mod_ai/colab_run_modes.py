from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_MODE = "Plan"
FULL_MODE = "Full"
EXISTING_MOD_MODE = "Revise"
EXISTING_PLAN_MODE = "Execute"
AUDIT_MODE = "Audit"
RUN_MODES = (
    PLAN_MODE,
    FULL_MODE,
    EXISTING_MOD_MODE,
    EXISTING_PLAN_MODE,
    AUDIT_MODE,
)

# Backward compatibility for already-open Colab notebooks and saved notebook copies.
LEGACY_RUN_MODE_ALIASES = {
    "플랜모드": PLAN_MODE,
    "풀모드": FULL_MODE,
    "이미 만들어진 모드 수정보안모드": EXISTING_MOD_MODE,
    "이미 있는 플랜을 만드는모드": EXISTING_PLAN_MODE,
    # The old run-mode label "Debug" meant repository audit. It is now "Audit"
    # because DEBUG_MODE in the Colab UI means planner-bypass implementation debug.
    "Debug": AUDIT_MODE,
}

AUDIT_RELATIVE_PATH = "tools/full_project_audit.py"
# Compatibility aliases for callers that imported the old audit name.
DEBUG_AUDIT_RELATIVE_PATH = AUDIT_RELATIVE_PATH
DEBUG_MODE = "Debug"


@dataclass(frozen=True)
class PlanDialogResult:
    reply: Any
    plan_path: Path
    approved: bool


def validate_run_mode(run_mode: str) -> str:
    value = run_mode.strip()
    canonical = LEGACY_RUN_MODE_ALIASES.get(value, value)
    if canonical not in RUN_MODES:
        raise ValueError(f"지원하지 않는 실행 모드: {value!r}")
    return canonical


def needs_prompt(run_mode: str) -> bool:
    return validate_run_mode(run_mode) not in {EXISTING_PLAN_MODE, AUDIT_MODE}


def needs_existing_mod(run_mode: str) -> bool:
    return validate_run_mode(run_mode) == EXISTING_MOD_MODE


def should_build(run_mode: str) -> bool:
    return validate_run_mode(run_mode) not in {PLAN_MODE, AUDIT_MODE}


def audit_path(repo_dir: str | Path) -> Path:
    return Path(repo_dir) / AUDIT_RELATIVE_PATH


def debug_audit_path(repo_dir: str | Path) -> Path:
    """Backward-compatible alias for the repository Audit entrypoint."""
    return audit_path(repo_dir)


def _uploaded_file(*, suffix: str, destination: Path, purpose: str) -> Path:
    try:
        from google.colab import files as colab_files
    except ImportError as exc:
        raise RuntimeError(f"{purpose} 업로드는 Google Colab에서 실행해야 합니다.") from exc

    print(f"{purpose}: 파일 선택", flush=True)
    uploaded = colab_files.upload()
    if len(uploaded) != 1:
        raise ValueError(f"{purpose}에는 파일을 정확히 하나 선택해야 합니다.")
    uploaded_name, uploaded_bytes = next(iter(uploaded.items()))
    safe_name = Path(uploaded_name).name
    if safe_name != uploaded_name or "/" in uploaded_name or "\\" in uploaded_name:
        raise ValueError("업로드 파일명에는 경로가 포함될 수 없습니다.")
    if Path(safe_name).suffix.lower() != suffix.lower():
        raise ValueError(f"{purpose}에는 {suffix} 파일이 필요합니다.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / safe_name
    target.write_bytes(uploaded_bytes)
    print(f"{purpose}: 준비 완료 {target}", flush=True)
    return target


def prepare_existing_mod_input(run_mode: str) -> Path | None:
    if not needs_existing_mod(run_mode):
        print("Revise input: 사용 안 함", flush=True)
        return None

    source = _uploaded_file(
        suffix=".zip",
        destination=Path("/content/mmm-existing-input"),
        purpose="Revise",
    )
    from .importer import inspect_existing_project_archive

    report = inspect_existing_project_archive(source)
    if not report.has_sources or not report.has_gradle_project:
        raise ValueError(
            "Revise에는 소스와 Gradle 프로젝트가 포함된 source/release ZIP이 필요합니다."
        )
    print(
        "Revise target:",
        report.mod_name or report.mod_id or source.name,
        flush=True,
    )
    return source


def resolve_plan_path(
    *,
    run_mode: str,
    output_root: str | Path,
    configured_path: str = "",
) -> Path:
    mode = validate_run_mode(run_mode)
    configured = configured_path.strip()
    if configured:
        path = Path(configured).expanduser()
    else:
        path = Path(output_root) / "proposal.json"

    if mode != EXISTING_PLAN_MODE:
        return path
    if path.is_file():
        return path

    if configured:
        raise FileNotFoundError(f"플랜 파일을 찾을 수 없습니다: {path}")

    print(f"기본 플랜 파일 없음: {path}", flush=True)
    return _uploaded_file(
        suffix=".json",
        destination=Path("/content/mmm-existing-plan"),
        purpose="Execute plan",
    )


def show_full_plan(reply: Any, *, print_fn: Callable[..., None] = print) -> None:
    proposal = reply.complete_proposal
    print_fn("")
    print_fn("=" * 80)
    print_fn("현재 플랜")
    print_fn("=" * 80)
    print_fn(reply.message)
    print_fn("")
    print_fn("플랜 전체 데이터")
    print_fn(json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2))
    print_fn("=" * 80)


def _debug_target(*, minecraft_version: str, loader: str):
    from .platform_catalog import adapter_for_target, executable_loaders, newest_adapter

    requested_version = str(minecraft_version or "").strip()
    requested_loader = str(loader or "").strip().casefold()
    if requested_version.casefold() == "auto":
        requested_version = ""
    if requested_loader == "auto":
        requested_loader = ""

    if requested_loader:
        selected_loader = requested_loader
    else:
        available = executable_loaders()
        if not available:
            raise RuntimeError("Debug Mode에 사용할 실행 가능한 loader가 없습니다.")
        selected_loader = available[0]

    if requested_version:
        return adapter_for_target(requested_version, selected_loader)
    return newest_adapter(loader=selected_loader)


def _debug_task_contract(platform: Any) -> dict[str, Any]:
    """Build the same task-local authority shape consumed by the normal custom coder path."""

    from .json_stream import canonical_json_sha256

    task_id = "debug_token"
    locator = "src/main/java/dev/mmm/debugfixture/DebugToken.java#DebugToken"
    anchor = {
        "kind": "symbol",
        "locator": locator,
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": ":",
        "source_set": "main",
    }
    task: dict[str, Any] = {
        "task_id": task_id,
        "task_sha256": "",
        "sequence": 0,
        "semantic_outcome": (
            "Implement one deterministic debug_token Fabric item fixture in the exact "
            "host-owned source target so the normal coder and verification pipeline runs."
        ),
        "execution_role": "production",
        "requirement_refs": ["debug_fixture_requirement"],
        "gap_refs": [],
        "owned_anchors": [anchor],
        "reuse_refs": [],
        "consumes": [],
        "provides": ["requirement_done:debug_fixture_requirement"],
        "depends_on": [],
        "implementation_obligations": [
            "Implement the debug_token item fixture using the immutable Fabric target APIs; keep all production source changes inside the owned DebugToken.java target.",
        ],
        "engineering_worksheet": {
            "schema_version": "mmm/debug-engineering-worksheet-v1",
            "objective": "Exercise the real task-local custom coding path without running the planner.",
            "implementation": [
                "Create the exact owned DebugToken Java source.",
                "Use the selected Fabric/Minecraft target coordinates without changing the target.",
                "Keep the fixture deterministic and self-contained for repeatable pipeline debugging.",
            ],
            "boundaries": [
                "Do not edit files outside the declared owned target.",
                "Do not replace host verification with model self-report.",
            ],
            "verification": ["target_compile"],
        },
        "target_cell": {
            "minecraft_version": platform.minecraft_version,
            "loader": platform.loader,
            "mappings": platform.yarn_mappings,
            "java_version": platform.java_version,
        },
        "production_bindings": [
            {
                "task_ref": task_id,
                "reuse_action": "fresh",
                "owned_anchors": [anchor],
            }
        ],
        "required_gates": ["target_compile"],
        "acceptance": [
            "The exact owned DebugToken.java source implements the debug_token fixture.",
            "The selected target compile gate passes for the generated project.",
        ],
        "public_acceptance": [],
        "runtime_acceptance": [],
        "impact_probes": ["changed_symbols"],
    }
    task["task_sha256"] = canonical_json_sha256(task)
    return task


def write_debug_example_plan(
    target: str | Path,
    *,
    minecraft_version: str = "Auto",
    loader: str = "Auto",
) -> Path:
    """Write one host-owned, schema-valid implementation fixture without calling an LLM planner."""

    from .capabilities import capability_manifest_hash
    from .complete_spec import (
        CompleteProposal,
        CompleteProposalStatus,
        ProductionModule,
    )
    from .implementation_template_contract import build_implementation_template
    from .knowledge import evidence_catalog_for_version, evidence_snapshot_hash
    from .platform_resolver import lock_from_adapter
    from .spec import ModSpec, Proposal, ProposalStatus

    adapter = _debug_target(minecraft_version=minecraft_version, loader=loader)
    platform = lock_from_adapter(adapter)
    evidence = evidence_catalog_for_version(platform.minecraft_version)
    prompt = (
        "M.M.M Debug Mode fixture: add one deterministic debug token item and "
        "run the normal implementation/verification pipeline."
    )
    acceptance = (
        "The generated project contains the debug_token item.",
        "The generated Fabric project passes the normal build and validation pipeline.",
    )
    base = Proposal(
        schema_version="minecraft-mod-ai/proposal-v1",
        proposal_version=1,
        status=ProposalStatus.AWAITING_APPROVAL,
        requested_prompt=prompt,
        spec=ModSpec(
            mod_id="mmm_debug_fixture",
            mod_name="MMM Debug Fixture",
            package_name="dev.mmm.debugfixture",
            version="1.0.0",
            summary="Deterministic implementation fixture for Colab Debug Mode.",
            contents=(),
            platform=platform,
        ),
        assumptions=(),
        exclusions=(),
        deferred_requests=(),
        acceptance_tests=acceptance,
        evidence_sources=evidence,
        evidence_snapshot_hash=evidence_snapshot_hash(evidence),
        capability_manifest_hash=capability_manifest_hash(),
        imported_source_snapshot_hash="",
        risk_approvals=(),
        approval_hash="",
    ).with_hash()
    evidence_task = _debug_task_contract(platform)
    coder_contract = build_implementation_template(evidence_task)
    proposal = CompleteProposal(
        schema_version="mmm/complete-proposal-v1",
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=prompt,
        base_proposal=base,
        game_design={
            "mode": "debug_fixture",
            "goal": "Exercise implementation and verification without planner/model planning.",
            "fixture": {
                "module_id": "debug_token",
                "kind": "custom_java",
                "semantic_kind": "item",
                "deterministic": True,
            },
        },
        modules=(
            ProductionModule(
                module_id="debug_token",
                kind="custom_java",
                config={
                    "summary": "Deterministic debug_token implementation fixture.",
                    "evidence_task": evidence_task,
                    "coder_execution_contract": coder_contract,
                },
                required_gates=tuple(evidence_task["required_gates"]),
            ),
        ),
        assets=(),
        acceptance_tests=acceptance,
        external_runtime_required=False,
        existing_input_sha256="",
        approval_hash="",
    ).with_hash()
    proposal.validate()

    path = Path(target).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_plan_dialog(
    *,
    session: Any,
    run_mode: str,
    prompt: str,
    plan_path: str | Path,
    debug_mode: bool = False,
    minecraft_version: str = "Auto",
    loader: str = "Auto",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> PlanDialogResult:
    """Create/load a plan and continue without a manual approval prompt.

    When ``debug_mode`` is enabled, the LLM planner is not called. A deterministic,
    host-owned example CompleteProposal is written, validated by ``session.load_plan``,
    and then returned to the normal build path.
    """

    del input_fn
    mode = validate_run_mode(run_mode)
    target = Path(plan_path)

    if debug_mode and mode != FULL_MODE:
        raise ValueError("Debug Mode는 RUN_MODE=Full에서만 사용할 수 있습니다.")

    if mode == AUDIT_MODE:
        raise RuntimeError("Audit 모드는 플랜/제작 대신 프로젝트 전체 진단만 실행합니다.")

    if debug_mode:
        write_debug_example_plan(
            target,
            minecraft_version=minecraft_version,
            loader=loader,
        )
        reply = session.load_plan(target)
        show_full_plan(reply, print_fn=print_fn)
        print_fn("Debug Mode: planner 호출 없이 예제 플랜을 주입해 바로 제작 단계로 진행합니다.")
        return PlanDialogResult(reply=reply, plan_path=target, approved=True)

    if mode == EXISTING_PLAN_MODE:
        reply = session.load_plan(target)
        show_full_plan(reply, print_fn=print_fn)
        print_fn("플랜 로드 완료: 사용자 승인 대기 없이 제작 단계로 진행합니다.")
        return PlanDialogResult(reply=reply, plan_path=target, approved=True)

    if not prompt.strip():
        raise ValueError(f"{mode}에서는 PROMPT를 입력해야 합니다.")

    reply = session.plan(prompt)
    target = session.save_plan(target)
    show_full_plan(reply, print_fn=print_fn)
    if mode == PLAN_MODE:
        print_fn("플랜 저장 완료: Plan 모드는 제작하지 않고 종료합니다.")
    else:
        print_fn("플랜 확정 완료: 사용자 승인 대기 없이 제작 단계로 진행합니다.")
    return PlanDialogResult(reply=reply, plan_path=target, approved=True)


__all__ = [
    "AUDIT_MODE",
    "AUDIT_RELATIVE_PATH",
    "DEBUG_AUDIT_RELATIVE_PATH",
    "DEBUG_MODE",
    "EXISTING_MOD_MODE",
    "EXISTING_PLAN_MODE",
    "FULL_MODE",
    "LEGACY_RUN_MODE_ALIASES",
    "PLAN_MODE",
    "RUN_MODES",
    "PlanDialogResult",
    "audit_path",
    "debug_audit_path",
    "needs_existing_mod",
    "needs_prompt",
    "prepare_existing_mod_input",
    "resolve_plan_path",
    "run_plan_dialog",
    "should_build",
    "show_full_plan",
    "validate_run_mode",
    "write_debug_example_plan",
]
