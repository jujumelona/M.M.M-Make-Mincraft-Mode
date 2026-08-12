from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) < count:
        raise RuntimeError(f"{path}: patch target changed: {old[:120]!r}")
    target.write_text(text.replace(old, new, count), encoding="utf-8")


def replace_section(path: str, start: str, end: str, replacement: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    left = text.find(start)
    if left < 0:
        raise RuntimeError(f"{path}: section start changed: {start!r}")
    right = text.find(end, left)
    if right < 0:
        raise RuntimeError(f"{path}: section end changed: {end!r}")
    target.write_text(text[:left] + replacement + text[right:], encoding="utf-8")


def remove_legacy_mtp() -> None:
    for name in (
        "minecraft_mod_ai/colab_mtp_server.py",
        "tests/test_colab_mtp_server.py",
    ):
        path = Path(name)
        if path.exists():
            path.unlink()

    replace(
        "tools/validate_colab_notebook.py",
        '    "mtp-server",\n',
        "",
    )
    replace(
        "tests/test_notebook_registry_policy.py",
        '        "mtp-server",\n',
        "",
    )


def fix_source_build_test() -> None:
    path = Path("tests/test_colab_native_llama_prebuilt_integration.py")
    text = path.read_text(encoding="utf-8")
    start = text.index("def test_prebuilt_failure_falls_back_to_exact_pinned_source_build(")
    end = text.index("\ndef test_bundle_loader_is_source_local_and_exposes_verified_installer", start)
    block = text[start:end]
    marker = '    monkeypatch.delenv("MMM_LLAMA_SERVER_DISTRIBUTION", raising=False)\n'
    if 'MMM_LLAMA_ALLOW_SOURCE_BUILD' not in block:
        if marker not in block:
            raise RuntimeError("source-build test environment marker changed")
        block = block.replace(
            marker,
            marker + '    monkeypatch.setenv("MMM_LLAMA_ALLOW_SOURCE_BUILD", "1")\n',
            1,
        )
    path.write_text(text[:start] + block + text[end:], encoding="utf-8")


def fix_native_colab_contract() -> None:
    replacement = '''def test_local_colab_profiles_use_verified_native_llama_cuda_runtime() -> None:\n    cells = _cells()\n    setup_source = Path("tools/colab_runtime_setup.py").read_text(encoding="utf-8")\n\n    assert 'MODEL_PROFILE = "Qwen3.5-9B_6GB"' in cells["configuration"]\n    assert '["Qwen3.5-9B_6GB", "Gemma4-12B_7GB", "Gemma4-26B_14GB", "Qwen3.6-35B_23GB", "Qwen3.6-27B_18GB", "Qwen3.6-27B_14GB", "mini_mod", "fast_test"]' in cells["configuration"]\n    assert '"tools" / "colab_runtime_setup.py"' in cells["setup"]\n    assert "spec_from_file_location" in cells["setup"]\n    assert "USED_COMMIT[:12]" in cells["setup"]\n    assert '"+main:refs/remotes/origin/main"' in cells["setup"]\n    assert '"merge",' in cells["setup"]\n    assert '"pull",' not in cells["setup"]\n    assert "refs/remotes/origin/main" in cells["setup"]\n    assert '"--untracked-files=no"' in cells["setup"]\n    assert "setup_colab_runtime(" in cells["setup"]\n    assert "engine_module_file=engine_module_file" in cells["setup"]\n    assert 'SETUP_STATE["receipt"]' in cells["setup"]\n    assert cells["setup"].index('print("GitHub commit:"') < cells["setup"].index("setup_colab_runtime(")\n    assert "flash-linear-attention" not in cells["setup"]\n\n    assert "LOCAL_PROFILES = frozenset(" in setup_source\n    assert '"Qwen3.5-9B_6GB"' in setup_source\n    assert 'REMOTE_PROJECT_INSTALL_TARGET = ".[ui,rag,' in setup_source\n    assert 'LOCAL_PROJECT_INSTALL_TARGET = ".[ui,local-model,rag,' in setup_source\n    assert "_install_project(local_profile=profile in LOCAL_PROFILES)" in setup_source\n    assert "LLAMA_SERVER_SOURCE_REF" in setup_source\n    assert "LLAMA_NATIVE_BUNDLE_VERSION" in setup_source\n    assert '"-DGGML_CUDA=ON"' in setup_source\n    assert '"-DGGML_CUDA_GRAPHS=ON"' in setup_source\n    assert "MMM_LLAMA_ALLOW_SOURCE_BUILD" in setup_source\n    assert "flash-linear-attention[cuda,conv1d]" not in setup_source\n\n\n'''
    replace_section(
        "tests/test_notebook_registry_policy.py",
        "def test_local_colab_profiles_require_verified_qwen_fast_kernels() -> None:\n",
        "def test_notebook_checks_setup_fingerprint_and_prints_resolved_planner() -> None:\n",
        replacement,
    )


def fix_incremental_journal_tests() -> None:
    replace(
        "tests/test_planner_incremental_repair_contract.py",
        '''    assert checkpoint["status"] == "complete"\n    assert checkpoint["pending_batches"] == []\n    assert checkpoint["pending_patch"] is None\n    assert checkpoint["saved_batches"] == [good, repaired]\n''',
        '''    assert checkpoint["status"] == "complete"\n    assert checkpoint["journal_version"] == 1\n    assert checkpoint["accepted_count"] == 2\n    assert checkpoint["pending_remaining"] == 0\n    assert checkpoint["pending_patch"] is None\n    assert "saved_batches" not in checkpoint\n    assert "pending_batches" not in checkpoint\n''',
    )
    replace(
        "tests/test_planner_incremental_repair_contract.py",
        '''    assert interrupted["saved_batches"] == [good]\n    assert interrupted["pending_batches"] == [bad]\n    assert interrupted["pending_patch"]["current_value"] == bad\n''',
        '''    assert interrupted["journal_version"] == 1\n    assert interrupted["accepted_count"] == 1\n    assert interrupted["pending_remaining"] == 1\n    assert "saved_batches" not in interrupted\n    assert "pending_batches" not in interrupted\n    assert interrupted["pending_patch"]["current_value"] == bad\n''',
    )


def fix_planner_json_tests() -> None:
    path = Path("tests/test_planner_json_runtime_contract.py")
    text = path.read_text(encoding="utf-8")
    if "import pytest\n" not in text:
        text = text.replace("import json\n", "import json\n\nimport pytest\n", 1)
    text = text.replace(
        '        \'{"modules":[{"module_id":"cut_off"\',\n',
        '        \'{"modules":[\',\n',
        1,
    )
    old_start = "def test_production_page_can_repair_more_than_once() -> None:\n"
    old_end = "def test_production_page_repairs_zero_progress_with_exact_host_diagnostic() -> None:\n"
    left = text.find(old_start)
    right = text.find(old_end, left)
    if left < 0 or right < 0:
        raise RuntimeError("production full-page repair test section changed")
    replacement = '''def test_production_page_limits_full_page_repair_to_once() -> None:\n    router = _Router("not json", "still not json", json.dumps({"modules": [_module()]}))\n\n    with pytest.raises(\n        complete_planner.SpecValidationError,\n        match="one page-local repair",\n    ):\n        complete_planner._generate_json_page_with_repair(\n            router,\n            system_prompt="Return the production page.",\n            request=_request(),\n            media_paths=(),\n            expected_contracts=(frozenset(complete_planner._PRODUCTION_PAGE_CONTRACT),),\n            stage="unit production page",\n        )\n\n    assert len(router.calls) == 2\n    assert "REPAIR THIS PAGE" in router.calls[1]["messages"][0]["content"]\n\n\n'''
    path.write_text(text[:left] + replacement + text[right:], encoding="utf-8")


def fix_execution_atomicity() -> None:
    path = "minecraft_mod_ai/execution_efficiency_contract.py"
    replace(
        path,
        '''            parts.modules.extend(page_modules)\n            parts.assets.extend(page_assets)\n            parts.audio.extend(page_audio)\n            parts.acceptance_tests.extend(tests)\n            test_catalog.update(tests)\n\n            completed_raw = page.get("completed_deliverables", [])\n            completed = {\n                str(value).strip()\n                for value in completed_raw\n                if isinstance(value, str) and str(value).strip() in set(remaining)\n            }\n            if not completed:\n                raise module.SpecValidationError(\n                    f"Production batch {batch.batch_id!r} page made no verified progress."\n                )\n\n            remaining = [value for value in remaining if value not in completed]\n''',
        '''            completed_raw = page.get("completed_deliverables", [])\n            completed = {\n                str(value).strip()\n                for value in completed_raw\n                if isinstance(value, str) and str(value).strip() in set(remaining)\n            }\n            if not completed:\n                raise module.SpecValidationError(\n                    f"Production batch {batch.batch_id!r} page made no verified progress."\n                )\n\n            # Commit page artifacts only after the host has verified monotonic progress.\n            # A rejected page must not leak partial modules/assets/tests into shared state.\n            parts.modules.extend(page_modules)\n            parts.assets.extend(page_assets)\n            parts.audio.extend(page_audio)\n            parts.acceptance_tests.extend(tests)\n            test_catalog.update(tests)\n\n            remaining = [value for value in remaining if value not in completed]\n''',
    )


def main() -> None:
    remove_legacy_mtp()
    fix_source_build_test()
    fix_native_colab_contract()
    fix_incremental_journal_tests()
    fix_planner_json_tests()
    fix_execution_atomicity()


if __name__ == "__main__":
    main()
