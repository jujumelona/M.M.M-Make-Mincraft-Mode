from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PLAN_MODE = "Plan"
FULL_MODE = "Full"
EXISTING_MOD_MODE = "Revise"
EXISTING_PLAN_MODE = "Execute"
DEBUG_MODE = "Debug"
DEBUG_AUDIT_RELATIVE_PATH = "tools/full_project_audit.py"
RUN_MODES = (
    PLAN_MODE,
    FULL_MODE,
    EXISTING_MOD_MODE,
    EXISTING_PLAN_MODE,
    DEBUG_MODE,
)

# Backward compatibility for already-open Colab notebooks and saved notebook copies
# created before the UI labels were renamed to concise English names. These values
# are accepted as input only; all runtime branching uses the canonical English mode.
LEGACY_RUN_MODE_ALIASES = {
    "플랜모드": PLAN_MODE,
    "풀모드": FULL_MODE,
    "이미 만들어진 모드 수정보안모드": EXISTING_MOD_MODE,
    "이미 있는 플랜을 만드는모드": EXISTING_PLAN_MODE,
}


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
    return validate_run_mode(run_mode) not in {EXISTING_PLAN_MODE, DEBUG_MODE}


def needs_existing_mod(run_mode: str) -> bool:
    return validate_run_mode(run_mode) == EXISTING_MOD_MODE


def should_build(run_mode: str) -> bool:
    return validate_run_mode(run_mode) not in {PLAN_MODE, DEBUG_MODE}


def debug_audit_path(repo_dir: str | Path) -> Path:
    """Return the canonical Debug audit entrypoint inside a repository checkout."""

    return Path(repo_dir) / DEBUG_AUDIT_RELATIVE_PATH


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


def run_plan_dialog(
    *,
    session: Any,
    run_mode: str,
    prompt: str,
    plan_path: str | Path,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> PlanDialogResult:
    """Create/load the plan and continue without a manual approval prompt.

    ``input_fn`` is retained only for API compatibility with older notebooks and
    tests; it is deliberately never called. Integrity/authorization remains
    content-bound through the proposal hash that CompleteModAISession.build() passes
    to the orchestrator automatically.
    """

    del input_fn
    mode = validate_run_mode(run_mode)
    target = Path(plan_path)

    if mode == DEBUG_MODE:
        raise RuntimeError("Debug 모드는 플랜을 만들지 않고 프로젝트 진단만 실행합니다.")

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
    "DEBUG_AUDIT_RELATIVE_PATH",
    "DEBUG_MODE",
    "EXISTING_MOD_MODE",
    "EXISTING_PLAN_MODE",
    "FULL_MODE",
    "LEGACY_RUN_MODE_ALIASES",
    "PLAN_MODE",
    "RUN_MODES",
    "PlanDialogResult",
    "debug_audit_path",
    "needs_existing_mod",
    "needs_prompt",
    "prepare_existing_mod_input",
    "resolve_plan_path",
    "run_plan_dialog",
    "should_build",
    "show_full_plan",
    "validate_run_mode",
]
