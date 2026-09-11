from __future__ import annotations

import inspect


def test_custom_staged_commit_uses_exact_path_lock() -> None:
    import minecraft_mod_ai.performance_final_contract as contract

    source = inspect.getsource(contract._install_staged_custom_generator)
    tail = source[source.index("captured = _select_custom_patch_capture"):]
    assert "with project_path_write_locks(live_root, commit_paths):" in tail
    assert "with project_write_lock(live_root):" not in tail
