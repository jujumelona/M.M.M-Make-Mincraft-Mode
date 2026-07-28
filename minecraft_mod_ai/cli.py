from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .importer import inspect_existing_project_archive
from .pipeline import MinecraftModPipeline
from .planner import HeuristicPlanner
from .routed_planner import RoutedPlanner
from .spec import Proposal


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmm",
        description="역할별 AI와 승인 해시 뒤에만 Fabric 1.20.1 프로젝트를 생성·검증합니다.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="쓰기 없이 제안서 JSON을 생성합니다.")
    plan.add_argument("prompt", help="만들고 싶은 모드 설명")
    plan.add_argument(
        "--backend",
        choices=("local", "heuristic-dev"),
        default="local",
        help="heuristic-dev is an explicit non-AI diagnostic backend.",
    )
    plan.add_argument("--profile", default="t4_local")
    plan.add_argument("--existing-zip", type=Path)
    plan.add_argument("--save", type=Path)

    execute = subparsers.add_parser(
        "execute", help="저장한 제안서와 정확한 승인 해시로 생성·빌드·검증합니다."
    )
    execute.add_argument("proposal", type=Path)
    execute.add_argument("--approve", required=True)
    execute.add_argument("--output", type=Path, default=Path("mmm-output"))
    execute.add_argument("--source-only", action="store_true")
    execute.add_argument("--skip-gametest", action="store_true")
    execute.add_argument("--gradle-cache", type=Path)
    execute.add_argument("--existing-zip", type=Path)

    inspect_existing = subparsers.add_parser(
        "inspect-existing", help="기존 모드 ZIP을 실행 없이 inventory합니다."
    )
    inspect_existing.add_argument("archive", type=Path)

    validate = subparsers.add_parser(
        "validate-proposal", help="제안서 스키마와 해시를 읽기 전용으로 검사합니다."
    )
    validate.add_argument("proposal", type=Path)

    ui = subparsers.add_parser("ui", help="역할 라우터 기반 Gradio 승인 UI를 실행합니다.")
    ui.add_argument("--output", type=Path, default=Path("mmm-output"))
    ui.add_argument("--profile", default="t4_local")
    ui.add_argument("--share", action="store_true")
    ui.add_argument("--server-name", default="127.0.0.1")
    return parser


def _read_proposal(path: Path) -> Proposal:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Proposal JSON must be an object.")
    return Proposal.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            planner = (
                RoutedPlanner(profile=args.profile)
                if args.backend == "local"
                else HeuristicPlanner()
            )
            proposal = MinecraftModPipeline(planner=planner).plan(
                args.prompt, existing_input=args.existing_zip
            )
            rendered = _json_dump(proposal.to_dict()) + "\n"
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                args.save.write_text(rendered, encoding="utf-8")
            sys.stdout.write(rendered)
            return 0

        if args.command == "inspect-existing":
            report = inspect_existing_project_archive(args.archive)
            sys.stdout.write(_json_dump(report.to_dict()) + "\n")
            return 0

        if args.command == "validate-proposal":
            proposal = _read_proposal(args.proposal)
            sys.stdout.write(
                _json_dump(
                    {
                        "status": "PASS",
                        "mod_id": proposal.spec.mod_id,
                        "approval_hash": proposal.calculate_hash(),
                        "stored_hash_matches": proposal.approval_hash == proposal.calculate_hash(),
                    }
                )
                + "\n"
            )
            return 0

        if args.command == "execute":
            proposal = _read_proposal(args.proposal)
            result = MinecraftModPipeline().execute(
                proposal,
                approval_hash=args.approve,
                output_root=args.output,
                build=not args.source_only,
                run_gametest=not args.skip_gametest,
                gradle_cache=args.gradle_cache,
                existing_input=args.existing_zip,
            )
            sys.stdout.write(_json_dump(result.to_dict()) + "\n")
            return 0 if result.status in {"VERIFIED", "SOURCE_READY"} else 1

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
