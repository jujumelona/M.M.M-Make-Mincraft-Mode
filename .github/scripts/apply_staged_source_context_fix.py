from __future__ import annotations

from pathlib import Path


SOURCE = Path("minecraft_mod_ai/custom_module_generator.py")
TEST = Path("tests/test_custom_module_staged_source_context.py")


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    early_index = '''        if self._cached_root == root and self._cached_index is not None:
            index = self._cached_index
        else:
            index = ProjectIndex(root, policy=self.policy)
            self._cached_root = root
            self._cached_index = index

'''
    require_once(text, early_index, "pre-checkpoint ProjectIndex cache block")
    text = text.replace(early_index, "", 1)

    budget_marker = '''        project_context_budget = _coder_project_context_budget(
            self.router,
            self.policy,
            fast_mode=self.fast_mode,
        )

'''
    budget_at = text.index(budget_marker) + len(budget_marker)
    observation_marker = "        observation_ledger: dict[str, Any] | None = None\n"
    observation_at = text.index(observation_marker, budget_at)
    research_marker = "        research_context = select_module_research_context(\n"
    research_at = text.index(research_marker, observation_at)
    text = text[:observation_at] + text[research_at:]

    research_at = text.index(research_marker, budget_at)
    host_marker = "        host_grounding = build_coder_grounding(\n"
    host_at = text.index(host_marker, research_at)
    before_marker = "        before = _project_snapshot(root)\n"
    before_at = text.index(before_marker, host_at)
    text = text[:host_at] + text[before_at:]

    insertion_marker = '''        approved_reuse_context = _materialize_owned_reuse_context(
            staged_root,
            module,
        )
'''
    require_once(text, insertion_marker, "approved-reuse insertion point")

    staged_context = '''        # Initial coder grounding must describe the same checkpoint workspace used
        # by source mutation and verification. A resumed checkpoint may differ from
        # the live project root, so build all exact-source context from staged_root.
        index = ProjectIndex(staged_root, policy=self.policy)
        observation_ledger: dict[str, Any] | None = None
        last_snapshot_error: ValueError | None = None
        for snapshot_attempt in range(3):
            try:
                observation_ledger = _collect_initial_observations(
                    index,
                    query=query,
                    byte_budget=project_context_budget,
                )
                break
            except ValueError as exc:
                if not _is_stale_project_index_error(exc):
                    raise
                last_snapshot_error = exc
                index = ProjectIndex(staged_root, policy=self.policy)
                print(
                    "custom module: refreshed changing staged ProjectIndex snapshot",
                    f"attempt={snapshot_attempt + 1}/3",
                    flush=True,
                )
        if observation_ledger is None:
            raise CustomModuleGenerationError(
                "Staged project source kept changing while custom-module context was captured; "
                f"last error: {last_snapshot_error}"
            )

        observation_pages = _observation_context_pages(
            observation_ledger,
            query=query,
            byte_budget=project_context_budget,
        )
        host_grounding = build_coder_grounding(
            module_kind=module.kind,
            source_observation_receipt=observation_ledger["receipt"],
            research_context=research_context,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
        )

'''
    text = text.replace(insertion_marker, staged_context + insertion_marker, 1)

    generate_start = text.index("    def generate(\n")
    generate_end = text.index(
        "    def _register_generation_checkpoint_cleanup(\n", generate_start
    )
    generate_source = text[generate_start:generate_end]
    prepare_at = generate_source.index("_prepare_generation_checkpoint(")
    staged_index_at = generate_source.index(
        "ProjectIndex(staged_root, policy=self.policy)"
    )
    request_at = generate_source.index("        request = {")
    if not (prepare_at < staged_index_at < request_at):
        raise SystemExit(
            "staged ProjectIndex must be built after checkpoint preparation and before request creation"
        )
    if "ProjectIndex(root, policy=self.policy)" in generate_source[:prepare_at]:
        raise SystemExit("live-root ProjectIndex still seeds pre-checkpoint coder context")
    if generate_source.count("ProjectIndex(staged_root, policy=self.policy)") < 2:
        raise SystemExit("staged ProjectIndex retry path is missing")
    if (
        '"initial_exact_source_context": observation_pages[0]'
        not in generate_source[staged_index_at:]
    ):
        raise SystemExit(
            "initial exact source context is not sourced from staged observation pages"
        )

    SOURCE.write_text(text, encoding="utf-8")

    TEST.write_text(
        '''from __future__ import annotations

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
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
