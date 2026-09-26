from __future__ import annotations

import inspect

from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator


def test_direct_coder_uses_source_owned_live_context_without_checkpoint_staging() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)

    assert "_prepare_generation_checkpoint(" not in source
    assert "TransactionalSourcePatcher" not in source
    assert "ProjectIndex(" not in source
    assert "_project_context(" in source
    assert "_dependency_source_context(" in source
    assert "require_fresh_evidence=False" in source


def test_direct_coder_applies_whole_file_write_under_project_lock() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    lock = source.index("with project_write_lock(root):")
    write = source.index("_atomic_write(target, candidate)")

    assert lock < write
    assert "patch or diff" in source
    assert "complete corrected Java file" in source
