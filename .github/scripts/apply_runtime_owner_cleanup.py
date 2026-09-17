from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_java_lsp() -> None:
    path = ROOT / "minecraft_mod_ai/java_lsp.py"
    replace_once(
        path,
        '            "import": {"gradle": {"enabled": True}},\n',
        '            "import": {\n                "gradle": {"enabled": True, "java": {"home": runtime["path"]}}\n            },\n',
        "native Gradle project JDK",
    )


def patch_retrieval() -> None:
    path = ROOT / "minecraft_mod_ai/retrieval.py"
    replace_once(path, "import re\n", "import re\nimport threading\n", "retrieval threading import")
    replace_once(
        path,
        '_TOKEN_PATTERN = re.compile(r"[a-z0-9_.:+-]+|[가-힣]{2,}", re.IGNORECASE)\n',
        '_TOKEN_PATTERN = re.compile(r"[a-z0-9_.:+-]+|[가-힣]{2,}", re.IGNORECASE)\n_CORPUS_THREAD_STATE = threading.local()\n',
        "thread local corpus state",
    )
    old = '''def retrieve_official_evidence(\n    query: str,\n    *,\n    minecraft_version: str | None = None,\n    loader: str | None = None,\n    mappings: str | None = None,\n    limit: int = 6,\n) -> RetrievalReceipt:\n    with OfficialCorpusIndex() as index:\n        return index.retrieve(\n            query,\n            minecraft_version=minecraft_version,\n            loader=loader,\n            mappings=mappings,\n            limit=limit,\n        )\n'''
    new = '''def retrieve_official_evidence(\n    query: str,\n    *,\n    minecraft_version: str | None = None,\n    loader: str | None = None,\n    mappings: str | None = None,\n    limit: int = 6,\n) -> RetrievalReceipt:\n    index = getattr(_CORPUS_THREAD_STATE, "official_index", None)\n    if index is None:\n        index = OfficialCorpusIndex()\n        _CORPUS_THREAD_STATE.official_index = index\n    return index.retrieve(\n        query,\n        minecraft_version=minecraft_version,\n        loader=loader,\n        mappings=mappings,\n        limit=limit,\n    )\n'''
    replace_once(path, old, new, "native corpus reuse")


def patch_production_tools() -> None:
    path = ROOT / "minecraft_mod_ai/production_tools.py"
    replace_once(
        path,
        "from .model_router import ModelRouter\n",
        "from .model_router import ModelRouter\nfrom .project_java_diagnostics_installation import run_project_java_diagnostics\n",
        "project diagnostics import",
    )
    replace_once(
        path,
        '''    def java_diagnostics(self, project_root: str, relative_files: list[str] | None=None, timeout_seconds: int=60) -> dict[str, Any]:\n        root = self._existing_dir(project_root)\n        return self.java.diagnostics(root, relative_files=relative_files, timeout_seconds=timeout_seconds)\n''',
        '''    def java_diagnostics(self, project_root: str, relative_files: list[str] | None=None, timeout_seconds: int=60) -> dict[str, Any]:\n        return run_project_java_diagnostics(\n            self,\n            project_root,\n            relative_files=relative_files,\n            timeout_seconds=timeout_seconds,\n        )\n''',
        "native project diagnostics",
    )


def patch_shared_shapes() -> None:
    mutation = ROOT / "minecraft_mod_ai/mutation_authority_final_guard.py"
    replace_once(
        mutation,
        "from .mutation_failure_classification import is_recoverable_mutation_failure\n",
        "from .mutation_failure_classification import is_recoverable_mutation_failure\nfrom .value_shapes import structured_payload as _structured_payload\n",
        "mutation shape import",
    )
    replace_once(
        mutation,
        '''def _structured_payload(content: Any) -> Any | None:\n    if isinstance(content, (Mapping, list, tuple)):\n        return content\n    if not isinstance(content, str):\n        return None\n    raw = content.strip()\n    if not raw.startswith(("{", "[")):\n        return None\n    try:\n        return json.loads(raw)\n    except (json.JSONDecodeError, TypeError, ValueError):\n        return None\n\n\n''',
        "",
        "mutation duplicate structured payload",
    )

    progress = ROOT / "minecraft_mod_ai/progress_aware_tool_loop.py"
    replace_once(
        progress,
        "from .source_mutation_contract import mutation_history_applied, mutation_payload_applied\n",
        "from .source_mutation_contract import mutation_history_applied, mutation_payload_applied\nfrom .value_shapes import as_sequence as _sequence, structured_payload as _structured_payload\n",
        "progress shape imports",
    )
    replace_once(
        progress,
        '''def _structured_payload(content: Any) -> Any | None:\n    if isinstance(content, (Mapping, list, tuple)):\n        return content\n    if not isinstance(content, str):\n        return None\n    raw = content.strip()\n    if not raw.startswith(("{", "[")):\n        return None\n    try:\n        return json.loads(raw)\n    except (json.JSONDecodeError, TypeError, ValueError):\n        return None\n\n\ndef _sequence(value: Any) -> tuple[Any, ...]:\n    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):\n        return tuple(value)\n    return ()\n\n\n''',
        "",
        "progress duplicate value shapes",
    )

    translation = ROOT / "minecraft_mod_ai/translation_runtime.py"
    replace_once(
        translation,
        "from .task_template_catalog import load_template\n",
        "from .task_template_catalog import load_template\nfrom .value_shapes import as_sequence as _sequence\n",
        "translation shape import",
    )
    replace_once(
        translation,
        '''def _sequence(value: Any) -> tuple[Any, ...]:\n    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):\n        return tuple(value)\n    return ()\n\n\n''',
        "",
        "translation duplicate sequence",
    )


def patch_diagnostic_owner() -> None:
    validation = ROOT / "minecraft_mod_ai/validation_diagnostic_contract.py"
    text = validation.read_text(encoding="utf-8")
    if text.count("def _is_error(item: Mapping[str, Any]) -> bool:") != 1:
        raise RuntimeError("validation diagnostic owner anchor missing")
    text = text.replace(
        "def _is_error(item: Mapping[str, Any]) -> bool:",
        "def is_error_diagnostic(item: Mapping[str, Any]) -> bool:",
        1,
    ).replace("_is_error(", "is_error_diagnostic(")
    validation.write_text(text, encoding="utf-8")

    verifier = ROOT / "minecraft_mod_ai/generation_verifier_resilience.py"
    replace_once(
        verifier,
        "from .root_cause_trace import emit_root_cause\n",
        "from .root_cause_trace import emit_root_cause\nfrom .validation_diagnostic_contract import is_error_diagnostic\n",
        "verifier diagnostic owner import",
    )
    replace_once(
        verifier,
        '''def _is_error_diagnostic(item: Mapping[str, Any]) -> bool:\n    try:\n        return int(item.get("severity", 1)) == 1\n    except (TypeError, ValueError, OverflowError):\n        return True\n\n\n''',
        "",
        "duplicate error diagnostic",
    )
    text = verifier.read_text(encoding="utf-8").replace(
        "_is_error_diagnostic(", "is_error_diagnostic("
    )
    verifier.write_text(text, encoding="utf-8")


def patch_java_separation_test() -> None:
    path = ROOT / "tests/test_java_toolchain_separation_installation.py"
    path.write_text(
        '''from __future__ import annotations\n\nfrom pathlib import Path\n\nfrom minecraft_mod_ai import java_lsp\n\n\ndef test_gradle_daemon_uses_resolved_project_jdk(monkeypatch) -> None:\n    project_home = "/opt/mmm/project-jdk-25"\n    monkeypatch.setattr(\n        java_lsp,\n        "_project_java_runtime",\n        lambda _home=None: {\n            "name": "JavaSE-25",\n            "path": project_home,\n            "default": True,\n        },\n    )\n\n    result = java_lsp._jdt_configuration()\n\n    assert result["java"]["configuration"]["runtimes"][0]["path"] == project_home\n    assert result["java"]["import"]["gradle"]["java"]["home"] == project_home\n\n\ndef test_explicit_project_jdk_is_forwarded_to_runtime_resolution(monkeypatch) -> None:\n    explicit = Path("/opt/mmm/selected-jdk-25")\n    observed = []\n\n    def runtime(home=None):\n        observed.append(home)\n        return {"name": "JavaSE-25", "path": str(explicit), "default": True}\n\n    monkeypatch.setattr(java_lsp, "_project_java_runtime", runtime)\n    result = java_lsp._jdt_configuration(explicit)\n\n    assert observed == [explicit]\n    assert result["java"]["import"]["gradle"]["java"]["home"] == str(explicit)\n\n\ndef test_jdt_launcher_java_home_is_process_local_and_separate(monkeypatch, tmp_path) -> None:\n    launcher = tmp_path / "jdt-launcher"\n    binary = launcher / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")\n    binary.parent.mkdir(parents=True)\n    binary.write_text("", encoding="utf-8")\n\n    monkeypatch.setenv("MMM_JDTLS_JAVA_HOME", str(launcher))\n    monkeypatch.setenv("JAVA_HOME", "/host/default-java")\n    env = java_lsp._jdtls_environment()\n\n    assert env["JAVA_HOME"] == str(launcher.resolve())\n    assert str(launcher.resolve() / "bin") in env["PATH"].split(__import__("os").pathsep)[0]\n''',
        encoding="utf-8",
    )


def main() -> None:
    patch_java_lsp()
    patch_retrieval()
    patch_production_tools()
    patch_shared_shapes()
    patch_diagnostic_owner()
    patch_java_separation_test()


if __name__ == "__main__":
    main()
