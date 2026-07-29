from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .complete_orchestrator import CompleteExecutionOptions, CompleteProductionOrchestrator
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import CompleteProposal
from .importer import inspect_existing_project_archive
from .model_router import ModelRouter
from .pipeline import MinecraftModPipeline
from .planner import HeuristicPlanner
from .routed_planner import RoutedPlanner
from .spec import Proposal


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmm",
        description="승인된 전체 Minecraft Fabric 1.20.1 제작 그래프를 생성·수리·실행·검증합니다.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="전체 모드·월드·GUI·시스템·자산 계획을 생성합니다.")
    plan.add_argument("prompt")
    plan.add_argument("--profile", default="t4_local")
    plan.add_argument("--media", type=Path, action="append", default=[])
    plan.add_argument("--existing-zip", type=Path)
    plan.add_argument("--save", type=Path)

    execute = subparsers.add_parser("execute", help="전체 제안서를 승인 해시로 끝까지 실행합니다.")
    execute.add_argument("proposal", type=Path)
    execute.add_argument("--approve", required=True)
    execute.add_argument("--output", type=Path, default=Path("mmm-output"))
    execute.add_argument("--profile", default="t4_local")
    execute.add_argument("--run-name", default="complete-run")
    execute.add_argument("--existing-zip", type=Path)
    execute.add_argument("--source-only", action="store_true")
    execute.add_argument("--skip-jdt", action="store_true")
    execute.add_argument("--skip-gametest", action="store_true")
    execute.add_argument("--no-repair", action="store_true")
    execute.add_argument("--repair-attempts", type=int, default=3)
    execute.add_argument("--skip-blockbench", action="store_true")
    execute.add_argument("--skip-runtime", action="store_true")
    execute.add_argument("--skip-client", action="store_true")
    execute.add_argument("--skip-mineflayer", action="store_true")
    execute.add_argument("--skip-visual", action="store_true")
    execute.add_argument("--server-launcher")
    execute.add_argument("--accept-eula", action="store_true")
    execute.add_argument("--screenshot", action="append", default=[])
    execute.add_argument("--playtest-actions", type=Path)
    execute.add_argument("--publish-provider", choices=("modrinth", "curseforge"))
    execute.add_argument("--publish-project-id")
    execute.add_argument("--changelog", default="Generated and verified by M.M.M")

    legacy_plan = subparsers.add_parser("plan-slice", help="호환용 아이템·블록 슬라이스를 계획합니다.")
    legacy_plan.add_argument("prompt")
    legacy_plan.add_argument("--backend", choices=("local", "heuristic-dev"), default="local")
    legacy_plan.add_argument("--profile", default="t4_local")
    legacy_plan.add_argument("--existing-zip", type=Path)
    legacy_plan.add_argument("--save", type=Path)

    legacy_execute = subparsers.add_parser("execute-slice", help="호환용 슬라이스 제안서를 실행합니다.")
    legacy_execute.add_argument("proposal", type=Path)
    legacy_execute.add_argument("--approve", required=True)
    legacy_execute.add_argument("--output", type=Path, default=Path("mmm-output"))
    legacy_execute.add_argument("--source-only", action="store_true")
    legacy_execute.add_argument("--skip-gametest", action="store_true")
    legacy_execute.add_argument("--existing-zip", type=Path)

    inspect_existing = subparsers.add_parser("inspect-existing")
    inspect_existing.add_argument("archive", type=Path)
    validate = subparsers.add_parser("validate-proposal")
    validate.add_argument("proposal", type=Path)
    ui = subparsers.add_parser("ui")
    ui.add_argument("--output", type=Path, default=Path("mmm-output"))
    ui.add_argument("--profile", default="t4_local")
    ui.add_argument("--share", action="store_true")
    ui.add_argument("--server-name", default="127.0.0.1")
    return parser


def _read_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON document must be an object.")
    return raw


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            existing_hash = ""
            if args.existing_zip:
                existing_hash = "sha256:" + hashlib.sha256(args.existing_zip.read_bytes()).hexdigest()
            proposal = CompleteGameDesignPlanner(ModelRouter(profile=args.profile)).plan(
                args.prompt,
                media_paths=args.media,
                existing_input_sha256=existing_hash,
            )
            rendered = _json_dump(proposal.to_dict()) + "\n"
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(rendered, encoding="utf-8")
            sys.stdout.write(rendered)
            return 0

        if args.command == "execute":
            proposal = CompleteProposal.from_dict(_read_json(args.proposal))
            actions = []
            if args.playtest_actions:
                value = json.loads(args.playtest_actions.read_text(encoding="utf-8"))
                if not isinstance(value, list):
                    raise ValueError("playtest-actions must be a JSON list.")
                actions = value
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
                approval_hash=args.approve,
                run_name=args.run_name,
                options=options,
                existing_input=args.existing_zip,
            )
            sys.stdout.write(_json_dump(result.to_dict()) + "\n")
            return 0 if result.status in {"VERIFIED", "SOURCE_READY"} else 1

        if args.command == "plan-slice":
            planner = RoutedPlanner(profile=args.profile) if args.backend == "local" else HeuristicPlanner()
            proposal = MinecraftModPipeline(planner=planner).plan(args.prompt, existing_input=args.existing_zip)
            rendered = _json_dump(proposal.to_dict()) + "\n"
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(rendered, encoding="utf-8")
            sys.stdout.write(rendered)
            return 0

        if args.command == "execute-slice":
            proposal = Proposal.from_dict(_read_json(args.proposal))
            result = MinecraftModPipeline().execute(
                proposal,
                approval_hash=args.approve,
                output_root=args.output,
                build=not args.source_only,
                run_gametest=not args.skip_gametest,
                existing_input=args.existing_zip,
            )
            sys.stdout.write(_json_dump(result.to_dict()) + "\n")
            return 0 if result.status in {"VERIFIED", "SOURCE_READY"} else 1

        if args.command == "inspect-existing":
            sys.stdout.write(_json_dump(inspect_existing_project_archive(args.archive).to_dict()) + "\n")
            return 0

        if args.command == "validate-proposal":
            raw = _read_json(args.proposal)
            if raw.get("schema_version") == "mmm/complete-proposal-v1":
                proposal = CompleteProposal.from_dict(raw)
                result = {
                    "status": "PASS",
                    "kind": "complete",
                    "approval_hash": proposal.calculate_hash(),
                    "stored_hash_matches": proposal.approval_hash == proposal.calculate_hash(),
                }
            else:
                proposal = Proposal.from_dict(raw)
                result = {
                    "status": "PASS",
                    "kind": "slice",
                    "approval_hash": proposal.calculate_hash(),
                    "stored_hash_matches": proposal.approval_hash == proposal.calculate_hash(),
                }
            sys.stdout.write(_json_dump(result) + "\n")
            return 0

        if args.command == "ui":
            from .ai_webui import launch

            launch(output_root=args.output, profile=args.profile, share=args.share, server_name=args.server_name)
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
