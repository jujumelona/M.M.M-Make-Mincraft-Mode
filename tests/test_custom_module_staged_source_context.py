from __future__ import annotations

import inspect

from minecraft_mod_ai.custom_module_generator import CustomModuleGenerator


def test_initial_coder_source_context_is_checkpoint_staged() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    prepare = source.index("_prepare_generation_checkpoint(")
    staged_index = source.index("ProjectIndex(staged_root, policy=self.policy)")
    request = source.index("request = {")

    assert prepare < staged_index < request
    assert "ProjectIndex(root, policy=self.policy)" not in source[:prepare]
    assert source.count("ProjectIndex(staged_root, policy=self.policy)") >= 2

    request_source = source[staged_index:]
    assert '"project_manifest": index.manifest_receipt()' in request_source
    assert '"source_observation_receipt": observation_ledger["receipt"]' in request_source
    assert '"initial_exact_source_context": observation_pages[0]' in request_source


def test_live_project_index_cache_is_only_refreshed_after_patch_apply() -> None:
    source = inspect.getsource(CustomModuleGenerator.generate)
    patch_apply = source.index("TransactionalSourcePatcher(root).apply(operations)")
    live_index = source.index("ProjectIndex(root, policy=self.policy)")

    assert live_index > patch_apply
