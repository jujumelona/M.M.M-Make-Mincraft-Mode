from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minecraft_mod_ai.final_artifact import (
    FinalArtifactError,
    append_github_outputs,
    bundle_from_pipeline_result,
    verify_final_mod_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify exactly one generated production mod JAR and emit receipts."
    )
    parser.add_argument("project_root", nargs="?", type=Path)
    parser.add_argument("--result", type=Path, help="JSON output from `mmm execute --json`.")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--mod-id", default="")
    parser.add_argument("--loader", default="")
    parser.add_argument("--minecraft-version", default="")
    parser.add_argument("--java", default="")
    parser.add_argument("--gradle", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.result is not None:
            if args.project_root is not None:
                raise FinalArtifactError(
                    "Supply either project_root or --result, not both."
                )
            if args.bundle_dir is None:
                raise FinalArtifactError("--result requires --bundle-dir.")
            result = json.loads(args.result.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise FinalArtifactError("Complete result JSON must be an object.")
            receipt = bundle_from_pipeline_result(
                result,
                args.bundle_dir,
                require_runtime=args.require_runtime,
            )
            if args.github_output is not None:
                append_github_outputs(args.github_output, receipt)
            sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return 0

        if args.project_root is None:
            raise FinalArtifactError("project_root or --result is required.")
        receipt_path = args.receipt or (
            args.project_root / ".minecraft_ai/final-artifact-receipt.json"
        )
        receipt = verify_final_mod_artifact(
            args.project_root,
            expected_mod_id=args.mod_id,
            expected_loader=args.loader,
            expected_minecraft_version=args.minecraft_version,
            expected_java=args.java,
            expected_gradle=args.gradle,
            receipt_path=receipt_path,
        ).to_dict()
        sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    except (FinalArtifactError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"final artifact verification failed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
