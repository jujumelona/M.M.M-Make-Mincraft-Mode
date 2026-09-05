from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import CompleteProposal
from .importer import inspect_existing_project_archive
from .model_router import ModelRouter
from .plan_render import render_complete_plan
from .planner import HeuristicPlanner
from .routed_planner import RoutedPlanner
from .scalable_pipeline import ScalableMinecraftModPipeline
from .spec import Proposal

_HASH_CHUNK_SIZE = 1024 * 1024
_SUCCESS_STATUSES = frozenset({"VERIFIED", "SOURCE_READY"})
_COMPLETE_SCHEMA_VERSIONS = frozenset(
    {"mmm/complete-proposal-v1", "mmm/complete-proposal-v2"}
)


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _sha256_file(path: Path) -> str:
    """Hash a file without materializing the whole payload in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmm",
        description=(
            "대화식 게임 기획을 Minecraft Java 모드로 "
            "제작하고 확인합니다."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="전체 모드·월드·GUI·시스템·자산 계획을 생성합니다.",
    )
    plan.add_argument("prompt")
    plan.add_argument("--profile", default="Qwen3.5-9B_6GB")
    plan.add_argument("--media", type=Path, action="append", default=[])
    plan.add_argument("--existing-zip", type=Path)
    plan.add_argument("--save", type=Path)
    plan.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    execute = subparsers.add_parser(
        "execute",
        help="저장한 계획을 이어서 제작합니다.",
    )
    execute.add_argument("proposal", type=Path)
    execute.add_argument("--approve", help=argparse.SUPPRESS)
    execute.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    execute.add_argument("--output", type=Path, default=Path("mmm-output"))
    execute.add_argument("--profile", default="Qwen3.5-9B_6GB")
    execute.add_argument("--run-name", default="complete-run")
    execute.add_argument("--existing-zip", type=Path)
    execute.add_argument("--source-only", action="store_true")
    execute.add_argument("--skip-jdt", action="store_true")
    execute.add_argument("--skip-gametest", action="store_true")
    execute.add_argument("--no-repair", action="store_true")
    execute.add_argument("--repair-attempts", type=int)
    execute.add_argument("--skip-blockbench", action="store_true")
    execute.add_argument("--skip-runtime", action="store_true")
    execute.add_argument("--skip-client", action="store_true")
    execute.add_argument("--skip-mineflayer", action="store_true")
    execute.add_argument("--skip-visual", action="store_true")
    execute.add_argument("--server-launcher")
    execute.add_argument("--accept-eula", action="store_true")
    execute.add_argument("--screenshot", action="append", default=[])
    execute.add_argument("--playtest-actions", type=Path)
    execute.add_argument(
        "--publish-provider",
        choices=("modrinth", "curseforge"),
    )
    execute.add_argument("--publish-project-id")
    execute.add_argument(
        "--changelog",
        default="Generated and verified by M.M.M",
    )

    slice_plan = subparsers.add_parser(
        "plan-slice",
        help="호환용 아이템·블록 슬라이스를 scalable 명세로 계획합니다.",
    )
    slice_plan.add_argument("prompt")
    slice_plan.add_argument(
        "--backend",
        choices=("local", "heuristic-dev"),
        default="local",
    )
    slice_plan.add_argument("--profile", default="Qwen3.5-9B_6GB")
    slice_plan.add_argument("--existing-zip", type=Path)
    slice_plan.add_argument("--save", type=Path)

    slice_execute = subparsers.add_parser(
        "execute-slice",
        help="호환용 슬라이스를 shard generator와 정책 검증기로 실행합니다.",
    )
    slice_execute.add_argument("proposal", type=Path)
    slice_execute.add_argument("--approve", required=True)
    slice_execute.add_argument(
        "--output",
        type=Path,
        default=Path("mmm-output"),
    )
    slice_execute.add_argument("--source-only", action="store_true")
    slice_execute.add_argument("--skip-gametest", action="store_true")
    slice_execute.add_argument("--existing-zip", type=Path)

    inspect_existing = subparsers.add_parser("inspect-existing")
    inspect_existing.add_argument("archive", type=Path)
    validate = subparsers.add_parser("validate-proposal")
    validate.add_argument("proposal", type=Path)
    ui = subparsers.add_parser("ui")
    ui.add_argument("--output", type=Path, default=Path("mmm-output"))
    ui.add_argument("--profile", default="Qwen3.5-9B_6GB")
    ui.add_argument("--share", action="store_true")
    ui.add_argument("--server-name", default="127.0.0.1")
    return parser


def _read_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON document must be an object.")
    return raw


def _read_playtest_actions(path: Path | None) -> list[dict]:
    if path is None:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("playtest-actions must be a JSON list.")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError("Every playtest action must be a JSON object.")
    return value


def _proposal_validation_result(raw: dict) -> dict[str, object]:
    if raw.get("schema_version") in _COMPLETE_SCHEMA_VERSIONS:
        proposal = CompleteProposal.from_dict(raw)
        kind = "complete"
    else:
        proposal = Proposal.from_dict(raw)
        kind = "slice"
    approval_hash = proposal.calculate_hash()
    return {
        "status": "PASS",
        "kind": kind,
        "approval_hash": approval_hash,
        "stored_hash_matches": proposal.approval_hash == approval_hash,
    }


def _render_complete_result(result: object) -> str:
    status = str(getattr(result, "status", "UNKNOWN"))
    project_root = str(getattr(result, "project_root", ""))
    lines = [f"제작 상태: {status}"]
    if project_root:
        lines.append(f"프로젝트: {project_root}")
    release_zip = getattr(result, "release_zip", None)
    jar_path = getattr(result, "jar_path", None)
    if release_zip:
        lines.append(f"다운로드 ZIP: {release_zip}")
    if jar_path:
        lines.append(f"모드 JAR: {jar_path}")
    unresolved = tuple(getattr(result, "unresolved_gates", ()) or ())
    if unresolved:
        lines.append("추가 확인이 필요한 항목: " + ", ".join(unresolved))
    if bool(getattr(result, "run_resumed", False)):
        lines.append("이전 실행에서 끝난 작업을 이어서 사용했습니다.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            existing_hash = _sha256_file(args.existing_zip) if args.existing_zip else ""
            proposal = CompleteGameDesignPlanner(
                ModelRouter(profile=args.profile)
            ).plan(
                args.prompt,
                media_paths=args.media,
                existing_input_sha256=existing_hash,
            )
            serialized = _json_dump(proposal.to_dict()) + "\n"
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(serialized, encoding="utf-8")
            if args.json:
                sys.stdout.write(serialized)
            else:
                rendered = render_complete_plan(
                    requested_prompt=proposal.requested_prompt,
                    game_design=proposal.game_design,
                    modules=proposal.modules,
                    acceptance_tests=proposal.acceptance_tests,
                )
                if args.save:
                    rendered += f"\n\n제작용 계획 저장: {args.save}"
                sys.stdout.write(rendered + "\n")
            return 0

        if args.command == "execute":
            proposal = CompleteProposal.from_dict(_read_json(args.proposal))
            actions = _read_playtest_actions(args.playtest_actions)
            options = CompleteExecutionOptions(
                source_only=args.source_only,
                run_jdt=not args.skip_jdt,
                run_gametest=not args.skip_gametest,
                auto_repair=not args.no_repair,
                max_repair_attempts=args.repair_attempts,
                run_blockbench=not args.skip_blockbench,
                run_runtime=not args.skip_runtime,
                run_client=not args.skip_client,
                run_mineflayer=not args.skip_mineflayer,
                run_visual_review=not args.skip_visual,
                eula_accepted=args.accept_eula,
                server_launcher=args.server_launcher,
                screenshot_paths=tuple(args.screenshot),
                playtest_actions=tuple(actions),
                publish_provider=args.publish_provider,
                publish_project_id=args.publish_project_id,
                changelog=args.changelog,
            )
            result = CompleteProductionOrchestrator(
                workspace_root=args.output,
                profile=args.profile,
            ).execute(
                proposal,
                approval_hash=args.approve or proposal.calculate_hash(),
                run_name=args.run_name,
                options=options,
                existing_input=args.existing_zip,
            )
            if args.json:
                sys.stdout.write(_json_dump(result.to_dict()) + "\n")
            else:
                sys.stdout.write(_render_complete_result(result) + "\n")
            return 0 if result.status in _SUCCESS_STATUSES else 1

        if args.command == "plan-slice":
            planner = (
                RoutedPlanner(profile=args.profile)
                if args.backend == "local"
                else HeuristicPlanner()
            )
            proposal = ScalableMinecraftModPipeline(planner=planner).plan(
                args.prompt,
                existing_input=args.existing_zip,
            )
            rendered = _json_dump(proposal.to_dict()) + "\n"
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(rendered, encoding="utf-8")
            sys.stdout.write(rendered)
            return 0

        if args.command == "execute-slice":
            proposal = Proposal.from_dict(_read_json(args.proposal))
            result = ScalableMinecraftModPipeline().execute(
                proposal,
                approval_hash=args.approve,
                output_root=args.output,
                build=not args.source_only,
                run_gametest=not args.skip_gametest,
                existing_input=args.existing_zip,
            )
            sys.stdout.write(_json_dump(result.to_dict()) + "\n")
            return 0 if result.status in _SUCCESS_STATUSES else 1

        if args.command == "inspect-existing":
            sys.stdout.write(
                _json_dump(
                    inspect_existing_project_archive(args.archive).to_dict()
                )
                + "\n"
            )
            return 0

        if args.command == "validate-proposal":
            sys.stdout.write(
                _json_dump(_proposal_validation_result(_read_json(args.proposal)))
                + "\n"
            )
            return 0

        if args.command == "ui":
            from .ai_webui import launch

            launch(
                output_root=args.output,
                profile=args.profile,
                share=args.share,
                server_name=args.server_name,
            )
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
