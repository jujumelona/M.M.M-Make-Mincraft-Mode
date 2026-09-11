"""Explicit registration, JAR inspection and real build evidence maintenance CLI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registration = sub.add_parser("registration")
    registration.add_argument("--minecraft", required=True)
    registration.add_argument("--output", type=Path, required=True)
    inspection = sub.add_parser("inspect")
    inspection.add_argument("--minecraft", required=True)
    inspection.add_argument("--loader", choices=["fabric", "neoforge", "forge"], required=True)
    inspection.add_argument("--namespace", required=True)
    inspection.add_argument("--java", type=int, required=True)
    inspection.add_argument("--jar", type=Path, action="append", required=True)
    inspection.add_argument("--mappings", type=Path)
    inspection.add_argument("--source-namespace")
    inspection.add_argument("--output", type=Path, required=True)
    evidence = sub.add_parser("evidence")
    evidence.add_argument("--project", type=Path, required=True)
    evidence.add_argument("--store", type=Path, required=True)
    evidence.add_argument("--expected", type=Path, required=True)
    evidence.add_argument("--gametest-task", required=True)
    evidence.add_argument("--report-glob", required=True)
    evidence.add_argument("--classpath", type=Path, action="append", default=[])
    candidate = sub.add_parser("candidate", help="Generate a deterministic mold candidate and run real gates before admission")
    candidate.add_argument("--project", type=Path, required=True)
    candidate.add_argument("--leaf", required=True)
    candidate.add_argument("--inputs", type=Path, required=True)
    candidate.add_argument("--target", type=Path, required=True)
    candidate.add_argument("--store", type=Path, required=True)
    candidate.add_argument("--classpath", type=Path, action="append", required=True)
    candidate.add_argument("--gametest-task", required=True)
    candidate.add_argument("--report-glob", required=True)
    candidate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "registration":
        from minecraft_mod_ai.integrity_bootstrap import bootstrap_integrity
        from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
        from minecraft_mod_ai.integrity_catalog import audit_reachability
        authority = bootstrap_integrity()
        facts = build_version_facts(args.minecraft, base_facts={})
        report = audit_reachability(facts, authority=authority)
        args.output.write_text(json.dumps({"audit": report, "facts": facts}, indent=2), encoding="utf-8")
        if report["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "inspect":
        from minecraft_mod_ai.api_epoch_catalog import inspect_api_epoch
        from minecraft_mod_ai.tiny_mappings import TinyMappings
        mappings = TinyMappings.parse(args.mappings.read_text(encoding="utf-8")) if args.mappings else None
        result = inspect_api_epoch(args.minecraft, loader=args.loader, namespace=args.namespace,
            java_version=args.java, jars=args.jar, mappings=mappings, source_namespace=args.source_namespace)
        args.output.write_text(json.dumps({"epoch_id": result.epoch_id, **asdict(result)}, indent=2), encoding="utf-8")
    elif args.command == "candidate":
        from minecraft_mod_ai.integrity_candidate import build_candidate_evidence
        from minecraft_mod_ai.evidence_store import EvidenceStore
        result = build_candidate_evidence(project_root=args.project, leaf_id=args.leaf,
            inputs=json.loads(args.inputs.read_text(encoding="utf-8")),
            target=json.loads(args.target.read_text(encoding="utf-8")), router=None,
            store=EvidenceStore(args.store), classpath=args.classpath,
            gametest_task=args.gametest_task, report_glob=args.report_glob)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        from minecraft_mod_ai.evidence_store import EvidenceStore
        from minecraft_mod_ai.integrity_evidence import run_gradle_evidence, verify_execution_evidence
        store = EvidenceStore(args.store)
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        identifier = run_gradle_evidence(args.project, store=store, expected=expected,
            gametest_task=args.gametest_task, report_glob=args.report_glob, classpath=args.classpath)
        print(identifier)
        verify_execution_evidence(store, identifier, expected=expected)


if __name__ == "__main__":
    main()
