from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_autotune() -> None:
    path = Path("minecraft_mod_ai/llama_server_autotune.py")
    text = path.read_text(encoding="utf-8")
    old = '''        "--load-mode",\n        "none",\n        "--no-ui",\n        "--log-disable",\n'''
    new = '''        "--load-mode",\n        "none",\n        # Tool-capable OpenAI chat requests require the Jinja chat engine.\n        # This belongs to the server launch contract itself because autotune,\n        # planner/coder priming and adapters can all be the first launch owner.\n        "--jinja",\n        "--no-ui",\n        "--log-disable",\n'''
    text = replace_once(text, old, new, label="autotune --jinja owner")
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def patch_adapter() -> None:
    path = Path("minecraft_mod_ai/model_adapters/llama_cpp_adapter.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''            from .. import llama_server_autotune\n\n            _enable_jinja_tool_templates(llama_server_autotune)\n            selected = llama_server_autotune.ensure_tuned_server(self.config, request)\n''',
        '''            from .. import llama_server_autotune\n\n            selected = llama_server_autotune.ensure_tuned_server(self.config, request)\n''',
        label="remove adapter jinja mutation call",
    )
    helper_start = text.find("\ndef _enable_jinja_tool_templates(autotune_module: Any) -> None:\n")
    if helper_start < 0:
        raise RuntimeError("adapter jinja helper not found")
    text = text[:helper_start].rstrip() + "\n"
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def patch_colab_setup() -> None:
    path = Path("tools/colab_runtime_setup.py")
    text = path.read_text(encoding="utf-8")
    anchor = '''def _installed_version(distribution: str) -> str | None:\n    try:\n        return package_version(distribution)\n    except PackageNotFoundError:\n        return None\n\n\ndef _validate_checkout(\n'''
    replacement = '''def _installed_version(distribution: str) -> str | None:\n    try:\n        return package_version(distribution)\n    except PackageNotFoundError:\n        return None\n\n\ndef _shutdown_loaded_managed_llama_server() -> bool:\n    \"\"\"Stop the old managed native server before purging a hot Colab engine.\n\n    A source update can change server launch flags. Keeping the previous process alive\n    while replacing its Python owner leaves LLAMA_SERVER_URL pointing at a server with\n    stale capabilities and loses the process handle needed for a clean restart.\n    \"\"\"\n\n    module = sys.modules.get("minecraft_mod_ai.llama_server_autotune")\n    if module is None:\n        return False\n    shutdown = getattr(module, "_shutdown_managed_server", None)\n    if not callable(shutdown):\n        return False\n    managed_url = str(getattr(module, "_MANAGED_URL", "") or "").strip()\n    try:\n        shutdown()\n    except Exception as exc:\n        raise RuntimeError(\n            "Failed to stop the managed llama-server before engine reload. "\n            "Restart the Colab runtime and rerun setup cell 2."\n        ) from exc\n    if managed_url and os.environ.get("LLAMA_SERVER_URL", "").strip() == managed_url:\n        os.environ.pop("LLAMA_SERVER_URL", None)\n    return True\n\n\ndef _validate_checkout(\n'''
    text = replace_once(text, anchor, replacement, label="colab managed server cleanup helper")
    old_reload = '''        print(\n            f"engine reload: {previous_commit[:7] if previous_commit else 'old'} -> "\n            f"{used_commit[:7]}",\n            flush=True,\n        )\n        for name in list(sys.modules):\n'''
    new_reload = '''        print(\n            f"engine reload: {previous_commit[:7] if previous_commit else 'old'} -> "\n            f"{used_commit[:7]}",\n            flush=True,\n        )\n        _shutdown_loaded_managed_llama_server()\n        for name in list(sys.modules):\n'''
    text = replace_once(text, old_reload, new_reload, label="colab reload cleanup call")
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def patch_coverage_diagnostics() -> None:
    path = Path("minecraft_mod_ai/minecraft_knowledge_contract.py")
    text = path.read_text(encoding="utf-8")
    old = '''            if coverage["status"] != "PASS":\n                error = getattr(agentic_module, "SpecValidationError", RuntimeError)\n                raise error(\n                    "Minecraft knowledge route coverage is incomplete: "\n                    + ", ".join(coverage["blocking_requirement_refs"][:16])\n                )\n'''
    new = '''            if coverage["status"] != "PASS":\n                error = getattr(agentic_module, "SpecValidationError", RuntimeError)\n                blocked_domains = [\n                    item\n                    for item in coverage.get("domains", [])\n                    if isinstance(item, Mapping)\n                    and str(item.get("status", "")) not in {\n                        "ROUTES_EXECUTED",\n                        "ROUTES_EXECUTED_WITH_GAPS",\n                    }\n                ]\n                domain_detail = "; ".join(\n                    f"{item.get('domain_id', 'unknown')}={item.get('status', 'unknown')}"\n                    for item in blocked_domains[:12]\n                )\n                notes = {\n                    str(item.get("domain_id", "")): item\n                    for item in result.get("domain_notes", [])\n                    if isinstance(item, Mapping)\n                }\n                failures = []\n                for item in blocked_domains[:4]:\n                    note = notes.get(str(item.get("domain_id", "")))\n                    if not isinstance(note, Mapping) or not note.get("worker_error"):\n                        continue\n                    failure = str(note.get("retry_error") or note.get("parallel_error") or "").strip()\n                    if failure:\n                        failures.append(\n                            f"{item.get('domain_id', 'unknown')}:{failure[:400]}"\n                        )\n                message = (\n                    "Minecraft knowledge route coverage is incomplete: "\n                    + ", ".join(coverage["blocking_requirement_refs"][:16])\n                )\n                if domain_detail:\n                    message += "; domains: " + domain_detail\n                if failures:\n                    message += "; research_errors: " + " | ".join(failures)\n                raise error(message)\n'''
    text = replace_once(text, old, new, label="coverage failure diagnostics")
    compile(text, str(path), "exec")
    path.write_text(text, encoding="utf-8")


def add_tests() -> None:
    path = Path("tests/test_llama_tool_startup_contract.py")
    if path.exists():
        raise RuntimeError(f"test already exists: {path}")
    path.write_text(
        '''from __future__ import annotations\n\nimport os\nimport sys\nfrom types import SimpleNamespace\n\nfrom minecraft_mod_ai import llama_server_autotune as autotune\nfrom minecraft_mod_ai.agentic_search_efficiency_contract import _prime_native_slots\nfrom minecraft_mod_ai.model_adapters.base import GenerationRequest\n\n\ndef test_every_native_server_start_path_includes_jinja() -> None:\n    config = SimpleNamespace(max_context=32768)\n    args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)\n    assert "--jinja" in args\n    assert args.count("--jinja") == 1\n\n\ndef test_planner_priming_uses_server_contract_that_already_owns_jinja(monkeypatch) -> None:\n    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)\n    monkeypatch.delenv("MMM_LLAMA_ACTIVE_PARALLEL", raising=False)\n    seen: dict[str, object] = {}\n\n    config = SimpleNamespace(\n        provider="local",\n        adapter="llama_cpp",\n        max_context=32768,\n    )\n    router = SimpleNamespace(\n        registry=SimpleNamespace(role=lambda profile, role: config),\n        profile="t4_local",\n    )\n\n    def ensure_tuned_server(received_config, request):\n        seen["config"] = received_config\n        seen["request"] = request\n        args = autotune._base_args("llama-server", "/tmp/model.gguf", config, 8910)\n        assert "--jinja" in args\n        return "http://127.0.0.1:8910/v1"\n\n    monkeypatch.setattr(autotune, "ensure_tuned_server", ensure_tuned_server)\n    returned = _prime_native_slots(\n        router,\n        system_prompt="system",\n        request={"plan": "test"},\n        media_paths=(),\n    )\n    assert returned is config\n    assert seen["config"] is config\n    assert isinstance(seen["request"], GenerationRequest)\n\n\ndef test_hot_colab_reload_stops_old_managed_server_before_module_purge(\n    monkeypatch, tmp_path\n) -> None:\n    import importlib.util\n    from pathlib import Path\n\n    setup_path = Path("tools/colab_runtime_setup.py").resolve()\n    spec = importlib.util.spec_from_file_location("_mmm_test_colab_runtime_setup", setup_path)\n    assert spec is not None and spec.loader is not None\n    setup = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(setup)\n\n    calls: list[str] = []\n    managed_url = "http://127.0.0.1:8910/v1"\n    fake_autotune = SimpleNamespace(\n        _MANAGED_URL=managed_url,\n        _shutdown_managed_server=lambda: calls.append("shutdown"),\n    )\n    monkeypatch.setitem(\n        sys.modules, "minecraft_mod_ai.llama_server_autotune", fake_autotune\n    )\n    monkeypatch.setenv("LLAMA_SERVER_URL", managed_url)\n\n    assert setup._shutdown_loaded_managed_llama_server() is True\n    assert calls == ["shutdown"]\n    assert "LLAMA_SERVER_URL" not in os.environ\n\n\ndef test_hot_colab_reload_preserves_unrelated_external_server_url(monkeypatch) -> None:\n    import importlib.util\n    from pathlib import Path\n\n    setup_path = Path("tools/colab_runtime_setup.py").resolve()\n    spec = importlib.util.spec_from_file_location("_mmm_test_colab_runtime_setup_2", setup_path)\n    assert spec is not None and spec.loader is not None\n    setup = importlib.util.module_from_spec(spec)\n    spec.loader.exec_module(setup)\n\n    fake_autotune = SimpleNamespace(\n        _MANAGED_URL="http://127.0.0.1:8910/v1",\n        _shutdown_managed_server=lambda: None,\n    )\n    monkeypatch.setitem(\n        sys.modules, "minecraft_mod_ai.llama_server_autotune", fake_autotune\n    )\n    monkeypatch.setenv("LLAMA_SERVER_URL", "https://example.invalid/v1")\n\n    assert setup._shutdown_loaded_managed_llama_server() is True\n    assert os.environ["LLAMA_SERVER_URL"] == "https://example.invalid/v1"\n''',
        encoding="utf-8",
    )
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def main() -> None:
    patch_autotune()
    patch_adapter()
    patch_colab_setup()
    patch_coverage_diagnostics()
    add_tests()


if __name__ == "__main__":
    main()
