from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .debug_prebuilt_plan import DEBUG_PREBUILT_PROMPT, write_prebuilt_debug_plan

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

AUDIT_RELATIVE_PATH = "tools/full_project_audit.py"
DEBUG_MODE = "Debug"
DEBUG_STRATEGIES = ("prebuilt", "model_path", "model_replay", "host_smoke")
DEBUG_DEFAULT_PROMPT = (
    "Fabric Minecraft 모드: 플레이어가 수정 조각을 획득하고, "
    "조각 4개로 수정 블록을 제작할 수 있게 구현해줘. "
    "아이템 등록, 레시피, 리소스, 실제 게임 내 검증까지 포함해줘."
)


@dataclass(frozen=True)
class PlanDialogResult:
    reply: Any
    plan_path: Path
    approved: bool


def validate_run_mode(run_mode: str) -> str:
    value = run_mode.strip()
    if value not in RUN_MODES:
        raise ValueError(f"지원하지 않는 실행 모드: {value!r}")
    return value


def needs_prompt(run_mode: str) -> bool:
    return validate_run_mode(run_mode) not in {EXISTING_PLAN_MODE, AUDIT_MODE}


def needs_existing_mod(run_mode: str) -> bool:
    return validate_run_mode(run_mode) == EXISTING_MOD_MODE


def should_build(run_mode: str) -> bool:
    return validate_run_mode(run_mode) not in {PLAN_MODE, AUDIT_MODE}


def _user_download_zip(build_result: Any) -> Path | None:
    """Package only end-user mod artifacts from the verified downloadable bundle."""

    distribution = getattr(build_result, "distribution_receipt", None)
    if not isinstance(distribution, Mapping):
        return None
    bundle = distribution.get("downloadable_bundle")
    if not isinstance(bundle, Mapping) or bundle.get("status") != "PASS":
        return None

    result_proposal_hash = str(
        getattr(build_result, "complete_proposal_hash", "") or ""
    ).strip()
    bundle_proposal_hash = str(bundle.get("proposal_hash") or "").strip()
    if (
        not result_proposal_hash
        or not bundle_proposal_hash
        or bundle_proposal_hash != result_proposal_hash
    ):
        return None

    raw_root = bundle.get("path")
    artifact_name = str(bundle.get("artifact") or "").strip()
    if not isinstance(raw_root, str) or not raw_root.strip():
        return None
    if not artifact_name or Path(artifact_name).name != artifact_name:
        return None

    # Reject the original symlink before resolving: Path.resolve() hides it.
    raw_directory = Path(raw_root).expanduser()
    if raw_directory.is_symlink():
        return None
    root = raw_directory.resolve()
    if not root.is_dir():
        return None

    names = [artifact_name]
    additional = bundle.get("additional_artifacts")
    if (
        isinstance(additional, Mapping)
        and "generated-resource-pack.zip" in additional
    ):
        names.append("generated-resource-pack.zip")

    receipt_members = bundle.get("members")
    if not isinstance(receipt_members, list):
        return None
    expected_hashes = {
        str(item.get("path") or ""): str(item.get("sha256") or "")
        for item in receipt_members
        if isinstance(item, Mapping)
    }
    artifact_sha256 = str(bundle.get("artifact_sha256") or "").strip()
    if expected_hashes.get(artifact_name) != artifact_sha256:
        return None

    members: list[Path] = []
    for name in names:
        # Resolve only after checking the lexical member; resolved paths are
        # never symlinks, even if the bundle member itself was a symlink.
        raw_candidate = root / name
        if raw_candidate.is_symlink():
            return None
        candidate = raw_candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        expected_sha256 = expected_hashes.get(name)
        if not expected_sha256:
            return None
        digest = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != expected_sha256:
            return None
        if (
            name != artifact_name
            and (
                not isinstance(additional, Mapping)
                or str(additional.get(name) or "") != expected_sha256
            )
        ):
            return None
        members.append(candidate)

    target = root.with_name(root.name + "-user.zip")
    temp = target.with_name("." + target.name + ".tmp")
    temp.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                archive.write(member, arcname=member.name)
        temp.replace(target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return target


def build_result_download_target(build_result: Any) -> tuple[Path | None, str]:
    """Return the end-user artifact produced by one build.

    Verified Full builds expose a user ZIP containing only the runnable mod JAR and,
    when required, its separately installable generated resource pack. Internal
    source/audit release ZIPs and receipt-heavy build bundles are fallback artifacts,
    not the primary user download.
    """

    if build_result is None:
        return None, "none"

    user_zip = _user_download_zip(build_result)
    if user_zip is not None:
        return user_zip, "user_mod_zip"

    for attribute, kind in (
        ("release_zip", "release_zip"),
        ("build_bundle_zip", "build_bundle_zip"),
        ("jar_path", "build_jar"),
    ):
        raw = getattr(build_result, attribute, None)
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            return path, kind
    return None, "none"


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
    debug_strategy: str | None = None,
) -> Path:
    mode = validate_run_mode(run_mode)
    if debug_strategy in {"prebuilt", "host_smoke", "model_path"}:
        if mode != FULL_MODE:
            raise ValueError("Debug Mode는 RUN_MODE=Full에서만 사용할 수 있습니다.")
        # Never overwrite the user's ordinary proposal.json or replay input.
        # Debug fixture files own a distinct, deterministic output namespace.
        return Path(output_root) / "debug" / f"{debug_strategy}-proposal.json"
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
    )
    from .debug_fixture_host import debug_fixture_source_contract
    from .knowledge import evidence_catalog_for_version, evidence_snapshot_hash
    from .platform_resolver import lock_from_adapter
    from .spec import ModSpec, Proposal, ProposalStatus

    adapter = _debug_target(minecraft_version=minecraft_version, loader=loader)
    platform = lock_from_adapter(adapter)
    evidence = evidence_catalog_for_version(platform.minecraft_version)
    source_contract = debug_fixture_source_contract(
        package_name="dev.mmm.debugfixture",
        minecraft_version=platform.minecraft_version,
    )
    prompt = (
        "M.M.M Debug Mode fixture: add one deterministic host-generated debug "
        "token item and run the normal implementation/verification pipeline."
    )
    acceptance = (
        "The generated project contains a deterministic host-generated debug_token item registration.",
        "The generated project contains the deterministic debug_token item resource definition.",
        "The generated project passes the normal build and validation pipeline.",
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
    proposal = CompleteProposal(
        schema_version="mmm/complete-proposal-v1",
        proposal_version=1,
        status=CompleteProposalStatus.AWAITING_APPROVAL,
        requested_prompt=prompt,
        base_proposal=base,
        game_design={
            "mode": "debug_fixture",
            "goal": (
                "Exercise deterministic host implementation and verification "
                "without planner or coder generation."
            ),
            "fixture": {
                "artifact_id": "debug_token",
                "kind": "item",
                "semantic_kind": "item",
                "deterministic": True,
                "source_contract": source_contract,
            },
        },
        modules=(),
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
    debug_strategy: str = "model_path",
    minecraft_version: str = "Auto",
    loader: str = "Auto",
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> PlanDialogResult:
    """Use the real planner for Debug by default, then use normal production.

    prebuilt exercises the saved AuthoredPlan and canonical content graph with no
    planning LLM calls. Source-reuse donor proofs still belong to production.
    model_path is an opt-in real model-authored plan and reference-retrieval path,
    not a hand-built CompleteProposal that bypasses the planning contracts.
    model_replay repeats a recorded real-model AuthoredPlan without another model call.\n    host_smoke remains available for narrow deterministic registry testing.
    """

    del input_fn
    mode = validate_run_mode(run_mode)
    target = Path(plan_path)

    if debug_mode and mode != FULL_MODE:
        raise ValueError("Debug Mode는 RUN_MODE=Full에서만 사용할 수 있습니다.")

    if mode == AUDIT_MODE:
        raise RuntimeError("Audit 모드는 플랜/제작 대신 프로젝트 전체 진단만 실행합니다.")

    if debug_mode:
        if debug_strategy not in DEBUG_STRATEGIES:
            raise ValueError(
                f"지원하지 않는 Debug 전략: {debug_strategy!r}; "
                f"allowed={DEBUG_STRATEGIES}"
            )
        if debug_strategy == "prebuilt":
            from .debug_prebuilt_plan import write_prebuilt_debug_plan

            write_prebuilt_debug_plan(target)
            print_fn(
                "DEBUG_FIXED_FIXTURE_WARNING: Prebuilt는 수정 아이템/블록/조합법 고정 사례만 생성합니다. "
                "사용자 PROMPT와 모델 계획·선택 경로를 검증하지 않습니다. "
                "실제 요청의 제작 검증에는 Model path를 선택하세요."
            )
        if debug_strategy == "host_smoke":
            write_debug_example_plan(
                target,
                minecraft_version=minecraft_version,
                loader=loader,
            )
            reply = session.load_plan(target)
            show_full_plan(reply, print_fn=print_fn)
            print_fn("Debug host_smoke: 호스트 아이템 등록만 검사합니다. 실제 플랜/참고 모드 경로 검증이 아닙니다.")
            return PlanDialogResult(reply=reply, plan_path=target, approved=True)

        from .authored_plan import AuthoredPlan
        from .authored_structured_design import normalize_structured_sections
        from .typed_plan_ir import validate_typed_plan_ir
        from .typed_host_capabilities import typed_host_capability_contracts

        if debug_strategy == "model_replay":
            if not target.is_file():
                raise FileNotFoundError(
                    f"DEBUG_MODEL_REPLAY_PLAN_MISSING: 실제 모델이 저장한 AuthoredPlan이 필요합니다: {target}"
                )
            reply = session.load_plan(target)
        elif debug_strategy == "prebuilt":
            reply = session.load_plan(target)
        else:
            model_prompt = prompt.strip() or DEBUG_DEFAULT_PROMPT
            # The ordinary Full-mode planner owns structured sections, Typed
            # PlanIR, content discovery and candidate source retrieval.
            print_fn(f"DEBUG_MODEL_PATH_PROMPT: {model_prompt}")
            reply = session.plan(model_prompt)
        authored = getattr(reply, "complete_proposal", None)
        if not isinstance(authored, AuthoredPlan):
            raise RuntimeError(
                "DEBUG_MODEL_PLAN_TYPE_MISMATCH: planner must return AuthoredPlan"
            )
        if debug_strategy == "model_path" and authored.requested_prompt.strip() != model_prompt:
            raise RuntimeError(
                "DEBUG_MODEL_PROMPT_BINDING_MISMATCH: the generated plan does not "
                "correspond to the requested prompt; refusing to build an unrelated mod"
            )
        if not normalize_structured_sections(authored.structured_sections):
            raise RuntimeError("DEBUG_MODEL_PLAN_STRUCTURED_SECTIONS_MISSING")
        if not authored.typed_plan_ir:
            raise RuntimeError("DEBUG_MODEL_PLAN_TYPED_IR_MISSING")
        validate_typed_plan_ir(
            authored.typed_plan_ir, capabilities=typed_host_capability_contracts()
        )
        content = authored.content_design
        if content.get("modules") and not content.get("_implementation_facts"):
            raise RuntimeError(
                "DEBUG_MODEL_PLAN_CONTENT_FACTS_MISSING: content modules cannot "
                "reach canonical artifact production without implementation facts"
            )
        has_content = bool(content.get("modules"))
        has_typed_work = bool(
            authored.typed_plan_ir.get("functions")
            or authored.typed_plan_ir.get("initialize")
            or authored.typed_plan_ir.get("platform_modules")
        )
        if not (has_content or has_typed_work):
            raise RuntimeError(
                "DEBUG_MODEL_PLAN_EMPTY_IMPLEMENTATION: no executable content "
                "or typed host operation was authored"
            )
        expected_hash = authored.calculate_hash()
        if debug_strategy in {"model_replay", "prebuilt"}:
            saved = target
            reloaded = reply
        else:
            saved = Path(session.save_plan(target))
            # Exercise the real serialization boundary before production.
            # Reject model/saved-plan ABI drift rather than hiding it.
            reloaded = session.load_plan(saved)
        restored = getattr(reloaded, "complete_proposal", None)
        if not isinstance(restored, AuthoredPlan) or restored.calculate_hash() != expected_hash:
            raise RuntimeError("DEBUG_MODEL_PLAN_ROUNDTRIP_MISMATCH")
        show_full_plan(reloaded, print_fn=print_fn)
        source_receipt = restored.content_design.get("_host_source_reuse")
        if isinstance(source_receipt, Mapping):
            print_fn(
                "Debug source-reuse plan:",
                json.dumps(
                    {
                        "bound_target": source_receipt.get("bound_target"),
                        "origin": source_receipt.get("origin"),
                        "capabilities": [
                            {
                                "capability": row.get("capability"),
                                "mode": row.get("mode"),
                                "source_id": row.get("source_id"),
                            }
                            for row in source_receipt.get("capabilities", ())
                            if isinstance(row, Mapping)
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            print_fn(
                "Debug source-reuse: 플랜 단계에 검증된 후보가 없으면 "
                "일반 제작의 타깃 바인딩 후 검색·선택·증명을 수행합니다."
            )
        print_fn(
            f"Debug {debug_strategy}: 저장된 AuthoredPlan → 일반 제작. "
            "실제 참고 모드 재사용은 타깃 바인딩 후 호스트 검색·검증·증명 경로를 사용하며 "
            "사전 계획은 확인되지 않은 donor를 검증된 코드로 표시하지 않습니다."
        )
        return PlanDialogResult(reply=reloaded, plan_path=saved, approved=True)

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
    "DEBUG_MODE",
    "DEBUG_STRATEGIES",
    "DEBUG_DEFAULT_PROMPT",
    "EXISTING_MOD_MODE",
    "EXISTING_PLAN_MODE",
    "FULL_MODE",
    "PLAN_MODE",
    "RUN_MODES",
    "PlanDialogResult",
    "audit_path",
    "build_result_download_target",
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
    "DEBUG_PREBUILT_PROMPT",
    "write_prebuilt_debug_plan",
]
