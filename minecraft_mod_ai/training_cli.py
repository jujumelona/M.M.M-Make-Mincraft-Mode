from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import TrainingTraceStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmm-training")
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("trace_json", type=Path)
    record.add_argument("--store", type=Path, default=Path("mmm-output/training/traces"))
    export = sub.add_parser("export")
    export.add_argument("--store", type=Path, default=Path("mmm-output/training/traces"))
    export.add_argument(
        "--output",
        type=Path,
        default=Path("mmm-output/training/mmm-fabric-coder-1201.jsonl"),
    )
    args = parser.parse_args()
    store = TrainingTraceStore(args.store)
    if args.command == "record":
        raw = json.loads(args.trace_json.read_text(encoding="utf-8"))
        print(json.dumps(store.record(raw), ensure_ascii=False, indent=2))
    elif args.command == "export":
        print(json.dumps(store.export_sft(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
