from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


validation_path = Path("minecraft_mod_ai/validation_checkpoint_policy.py")
validation = validation_path.read_text(encoding="utf-8")
validation = replace_once(
    validation,
    '_VALIDATION_CHECKPOINTS = frozenset({"validate-source", "validate-jdt"})\n',
    '''_VALIDATION_CHECKPOINT_FAMILIES = {
    "validate-source": "validate-source",
    "validate-source-final": "validate-source",
    "validate-jdt": "validate-jdt",
    "validate-jdt-final": "validate-jdt",
}


def _canonical_validation_checkpoint(checkpoint_id: str) -> str:
    family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)
    if family is None:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")
    return family
''',
    "validation checkpoint families",
)
validation = replace_once(
    validation,
    '''    if checkpoint_id not in _VALIDATION_CHECKPOINTS:
        raise ValueError(f"Unsupported validation checkpoint: {checkpoint_id}")

    digest = hashlib.sha256()
    for module in _validation_modules(checkpoint_id):
''',
    '''    checkpoint_family = _canonical_validation_checkpoint(checkpoint_id)

    digest = hashlib.sha256()
    for module in _validation_modules(checkpoint_family):
''',
    "validation implementation canonicalization",
)
validation = replace_once(
    validation,
    '''def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if checkpoint_id == "validate-source":
        return _complete_source_receipt(value)
    if checkpoint_id == "validate-jdt":
        return _complete_jdt_receipt(value)
    return False
''',
    '''def cached_validation_is_reusable(checkpoint_id: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    checkpoint_family = _VALIDATION_CHECKPOINT_FAMILIES.get(checkpoint_id)
    if checkpoint_family == "validate-source":
        return _complete_source_receipt(value)
    if checkpoint_family == "validate-jdt":
        return _complete_jdt_receipt(value)
    return False
''',
    "cached validation canonicalization",
)
validation_path.write_text(validation, encoding="utf-8")

repair_path = Path("minecraft_mod_ai/repair_engine.py")
repair = repair_path.read_text(encoding="utf-8")
repair = replace_once(
    repair,
    "import json\nimport os\nimport re\n",
    "import hashlib\nimport json\nimport os\nimport re\n",
    "repair hashlib import",
)
repair = replace_once(
    repair,
    "_HARD_REPAIR_ATTEMPTS = 2\n",
    "_HARD_REPAIR_ATTEMPTS = 2\n_REPAIR_LOG_SNIPPET_CHARS = 3000\n_REPAIR_LOG_READ_BYTES = 16384\n_REPAIR_BUILD_LOG_LIMIT = 4\n_REPAIR_QUERY_PART_LIMIT = 12\n",
    "repair log bounds",
)
class_marker = "\n\nclass RepairEngine:\n"
helper = '''


def _read_bounded_build_log(log_path: Any) -> str:
    raw_path = str(log_path or "").strip()
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    try:
        if not path.is_file() or path.is_symlink():
            return ""
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _REPAIR_LOG_READ_BYTES))
            payload = handle.read(_REPAIR_LOG_READ_BYTES)
    except (OSError, ValueError):
        return ""
    text = payload.decode("utf-8", errors="replace")
    normalized = text.replace("\\r\\n", "\\n").replace("\\r", "\\n").strip()
    return normalized[-_REPAIR_LOG_SNIPPET_CHARS:]


def _failed_build_log_diagnostics(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    build = evidence.get("build")
    if not isinstance(build, dict):
        return []
    commands = [item for item in build.get("commands", []) if isinstance(item, dict)]
    failed = [
        item
        for item in commands
        if bool(item.get("timed_out"))
        or (
            isinstance(item.get("exit_code"), int)
            and not isinstance(item.get("exit_code"), bool)
            and item.get("exit_code") != 0
        )
    ]
    if not failed and build.get("status") == "FAIL" and commands:
        failed = [commands[-1]]

    diagnostics: list[dict[str, Any]] = []
    for command in failed[-_REPAIR_BUILD_LOG_LIMIT:]:
        output = _read_bounded_build_log(command.get("log_path"))
        item: dict[str, Any] = {
            "name": command.get("name"),
            "exit_code": command.get("exit_code"),
            "timed_out": bool(command.get("timed_out")),
        }
        if output:
            item["output"] = output
        diagnostics.append(item)
    return diagnostics
'''
repair = replace_once(repair, class_marker, helper + class_marker, "repair build log helpers")
repair = replace_once(
    repair,
    '''    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                {
                    "path": item.get("path") or item.get("uri"),
                    "message": item.get("message"),
                    "code": item.get("code"),
                    "severity": item.get("severity"),
                }
            )
        build = evidence.get("build", {})
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
''',
    '''    @staticmethod
    def _signature(evidence: dict[str, Any]) -> str:
        diagnostics = []
        for item in _diagnostic_items(evidence.get("diagnostics")):
            if not isinstance(item, dict):
                continue
            diagnostics.append(
                {
                    "path": item.get("path") or item.get("uri"),
                    "message": item.get("message"),
                    "code": item.get("code"),
                    "severity": item.get("severity"),
                }
            )
        build = evidence.get("build", {})
        build_logs = []
        for item in _failed_build_log_diagnostics(evidence):
            output = str(item.get("output") or "")
            build_logs.append(
                {
                    "name": item.get("name"),
                    "exit_code": item.get("exit_code"),
                    "timed_out": item.get("timed_out"),
                    "output_sha256": (
                        hashlib.sha256(output.encode("utf-8")).hexdigest()
                        if output
                        else ""
                    ),
                }
            )
        return json.dumps(
            {
                "diagnostics": diagnostics,
                "build_status": build.get("status"),
                "build_error": build.get("error"),
                "build_logs": build_logs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
''',
    "repair signature",
)
repair = replace_once(
    repair,
    '''        for command in evidence.get("build", {}).get("commands", []):
            if not isinstance(command, dict):
                continue
            output = command.get("output")
            if isinstance(output, str):
                query_parts.append(output)
        from .production_tools import ProjectRAGIndex
''',
    '''        build_logs = _failed_build_log_diagnostics(evidence)
        for command in build_logs:
            output = command.get("output")
            if isinstance(output, str) and output:
                query_parts.append(output)
        from .production_tools import ProjectRAGIndex
''',
    "repair context log loading",
)
repair = replace_once(
    repair,
    '            query = " ".join(query_parts) if query_parts else "Minecraft Fabric mod build repair"\n',
    '            bounded_query_parts = query_parts[-_REPAIR_QUERY_PART_LIMIT:]\n            query = " ".join(bounded_query_parts) if bounded_query_parts else "Minecraft Fabric mod build repair"\n',
    "repair RAG query bound",
)
repair = replace_once(
    repair,
    '''        return {
            "diagnostics_files": tuple(sorted(set(diagnostic_paths))),
            "rag": {"hits": rag_hits},
        }
''',
    '''        return {
            "diagnostics_files": tuple(sorted(set(diagnostic_paths))),
            "build_logs": build_logs,
            "rag": {"hits": rag_hits},
        }
''',
    "repair context build logs",
)
repair_path.write_text(repair, encoding="utf-8")

adaptive_path = Path("minecraft_mod_ai/adaptive_retrieval_contract.py")
adaptive = adaptive_path.read_text(encoding="utf-8")
adaptive = replace_once(
    adaptive,
    '''        for command in build.get("commands", []):
            if not isinstance(command, dict) or not isinstance(command.get("log_path"), str):
                continue
            log = Path(command["log_path"])
            if log.is_file() and not log.is_symlink():
                query_parts.append(log.read_text(encoding="utf-8", errors="replace")[-32_000:])
''',
    '''        build_logs = repair_engine._failed_build_log_diagnostics(evidence)
        for command in build_logs:
            output = command.get("output")
            if isinstance(output, str) and output:
                query_parts.append(output)
''',
    "adaptive repair log extraction",
)
adaptive = replace_once(
    adaptive,
    '''        index = repair_engine.active_repair_project_index(root, self.policy)
        return build_repair_repository_context(
            self.router,
            index,
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=_runtime_grounding_budget(
                self.router,
                self.policy.model_context_bytes,
                role="coder_safe",
            ),
        )
''',
    '''        index = repair_engine.active_repair_project_index(root, self.policy)
        context = build_repair_repository_context(
            self.router,
            index,
            query=query,
            diagnostic_paths=diagnostic_paths,
            byte_budget=_runtime_grounding_budget(
                self.router,
                self.policy.model_context_bytes,
                role="coder_safe",
            ),
        )
        context["build_logs"] = build_logs
        return context
''',
    "adaptive repair context evidence",
)
adaptive_path.write_text(adaptive, encoding="utf-8")
