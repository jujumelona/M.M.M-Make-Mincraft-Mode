from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


path = Path("minecraft_mod_ai/adaptive_retrieval_contract.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
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
text = replace_once(
    text,
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
path.write_text(text, encoding="utf-8")


test_path = Path("tests/test_validation_repair_regressions.py")
test = test_path.read_text(encoding="utf-8")
test = replace_once(
    test,
    'import minecraft_mod_ai.production_tools as production_tools\n',
    '',
    "remove obsolete production_tools import",
)
test = replace_once(
    test,
    'from minecraft_mod_ai.repair_engine import RepairEngine\n',
    'from minecraft_mod_ai.repair_engine import RepairEngine\nfrom minecraft_mod_ai.scale_policy import ScalePolicy\n',
    "scale policy import",
)
test = replace_once(
    test,
    '        "diagnostics": None,\n',
    '''        "diagnostics": {
            "schema_version": "mmm/java-diagnostics-v2",
            "status": "PASS",
            "diagnostics": {},
            "diagnostics_by_uri": {},
            "pages": [],
            "files_opened": 0,
            "page_count": 0,
            "error_count": 0,
            "warning_count": 0,
        },
''',
    "realistic diagnostics receipt",
)
old = '''def test_repair_context_reads_bounded_failed_gradle_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "gradle-build.log"
    marker = "error: cannot find symbol ExampleRegistry"
    log_path.write_text("x" * 5000 + "\\n" + marker + "\\n", encoding="utf-8")

    class BrokenRag:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("RAG intentionally unavailable in unit test")

    monkeypatch.setattr(production_tools, "ProjectRAGIndex", BrokenRag)
    engine = object.__new__(RepairEngine)
    context = engine._context(tmp_path, _failed_evidence(log_path))
    assert context["build_logs"]
    output = context["build_logs"][0]["output"]
    assert marker in output
    assert len(output) <= repair_module._REPAIR_LOG_SNIPPET_CHARS
'''
new = '''def test_repair_context_reads_bounded_failed_gradle_log(tmp_path: Path) -> None:
    log_path = tmp_path / "gradle-build.log"
    marker = "error: cannot find symbol ExampleRegistry"
    log_path.write_text("x" * 5000 + "\\n" + marker + "\\n", encoding="utf-8")

    engine = object.__new__(RepairEngine)
    engine.policy = ScalePolicy.from_environment()
    engine.router = None
    context = engine._context(tmp_path, _failed_evidence(log_path))
    assert context["build_logs"]
    output = context["build_logs"][0]["output"]
    assert marker in output
    assert len(output) <= repair_module._REPAIR_LOG_SNIPPET_CHARS
'''
test = replace_once(test, old, new, "runtime repair context test")
test_path.write_text(test, encoding="utf-8")
