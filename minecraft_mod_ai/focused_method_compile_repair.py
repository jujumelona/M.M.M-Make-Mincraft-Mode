"""Model-backed compile repair of one existing, compiler-located Java method.

Select a method by *real javac line diagnostics*. The model gets the exact
unmodified method header and may only replace its body. Host applies the
replacement and checks it with real compileJava. No model API selection loop.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .construction_context import java_source_units
from .feature_compile_repair import compile_and_repair_feature


def _failing_method(source: str, target_path: str, diagnostics: tuple[dict[str, Any], ...]):
    units = java_source_units(source, path=target_path)
    eligible = [row for row in units if row.get("kind") == "method"
                and row.get("has_body")]
    for diagnostic in diagnostics:
        try:
            line = int(diagnostic.get("line") or 0)
        except (ValueError, TypeError):
            continue
        matched = [row for row in eligible
                   if int(row.get("start_line") or 0) <= line
                   <= int(row.get("end_line") or 0)]
        if len(matched) == 1:
            return matched[0]
    return None


def repair_staged_java_with_model(
    router: Any, project_root: str | Path, *, target_path: str,
    max_repairs: int = 2,
) -> dict[str, Any]:
    """Run a bounded diagnostic-guided compile of a staged Java feature.

    Requires an explicitly selected task-owned Java path. It cannot choose
    arbitrary files from a compiler log and does not touch Gradle metadata.
    """
    def rewrite(source: str, diagnostics: tuple[dict[str, Any], ...], attempt: int) -> str:
        method = _failing_method(source, target_path, diagnostics)
        if method is None:
            return source  # no method-local fix possible; fail without guessing
        old = str(method["code"])
        if source.count(old) != 1:
            return source
        signature = str(method["signature"]).strip()
        frame = {
            "kind": "method",
            "prefix": signature + " {\n",
            "suffix": "\n}",
        }
        errors = [{
            "line": d.get("line"),
            "code": str(d.get("code") or "")[:120],
            "message": str(d.get("message") or "")[:500],
        } for d in diagnostics]
        prompt = [
            {"role": "system", "content": (
                "You are repairing ONE Java method against exact compiler "
                "diagnostics. Return JSON with one implementation string containing "
                "only the body; the host supplies its fixed signature. "
                "Use target Java syntax, existing dependencies and "
                "real APIs; preserve the behavior. Do not change package, imports, "
                "class, method parameters, other methods or state ownership. "
                "Never silence an error with a TODO or empty stub."
            )},
            {"role": "user", "content": json.dumps({
                "target_path": target_path,
                "method_signature": signature,
                "method_source": old[:6500],
                "compiler_diagnostics": errors,
                "repair_attempt": attempt,
                "allowed_changes": "one method body",
            }, ensure_ascii=False)},
        ]
        replacement = router.generate_java_source(
            "coder", prompt,
            data_name="feature_compile_repair",
            source_frame=frame,
        )
        if (not isinstance(replacement, str)
                or not replacement.startswith(frame["prefix"])
                or not replacement.endswith(frame["suffix"])):
            return source
        body = replacement[len(frame["prefix"]):-len(frame["suffix"])].strip()
        if (not body or body in {";", "return;"}
                or re.search(r"\b(?:TODO|FIXME|UnsupportedOperationException)\b", body)):
            # Compiler-pass alone must not promote a trivially erased feature.
            return source
        return source.replace(old, replacement, 1)

    return compile_and_repair_feature(
        project_root, target_path=target_path, edit=rewrite,
        max_repairs=max_repairs,
    )
