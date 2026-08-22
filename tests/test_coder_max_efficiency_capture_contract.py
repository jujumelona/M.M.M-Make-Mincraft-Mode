from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import custom_generation_search_contract as custom_search
from minecraft_mod_ai.coder_max_efficiency_contract import _parallel_generate
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.source_patch import TransactionalSourcePatcher


class _Router:
    pass


def test_width_two_candidate_wrapper_captures_and_commits_winner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the exact caller/callee API that previously lacked a symbol."""

    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    monkeypatch.setenv("MMM_CUSTOM_CANDIDATE_JDT", "off")

    root = tmp_path / "project"
    root.mkdir()
    acknowledgements: list[dict] = []
    releases: list[dict] = []
    discards: list[dict] = []

    def acknowledge(result):
        # The checkpoint contract may only be acknowledged after the winning patch
        # is visible in the live project.
        assert (root / "src/main/java/example/ParallelCapture.java").is_file()
        acknowledgements.append(result)
        return True

    def release(result):
        releases.append(result)
        return True

    def discard(result):
        discards.append(result)
        return True

    module = SimpleNamespace(
        kind="custom_java",
        config={"feature": "parallel capture"},
        depends_on=(),
        required_gates=(),
    )
    owner = SimpleNamespace(
        router=_Router(),
        policy=ScalePolicy(model_context_bytes=4096),
        _cached_index=None,
        _cached_root=None,
        acknowledge_generation_checkpoint=acknowledge,
        release_generation_checkpoint=release,
        discard_generation_checkpoint=discard,
    )
    platform = adapter_for_target("1.20.1", "fabric")

    def single_generate(self, project_root, *args, **kwargs):
        assert self.router is not owner.router
        relative = "src/main/java/example/ParallelCapture.java"
        receipt = TransactionalSourcePatcher(project_root).apply(
            [
                {
                    "operation": "create",
                    "path": relative,
                    "content": "package example;\nfinal class ParallelCapture {}\n",
                }
            ]
        )
        return {
            "schema_version": "test/custom-candidate-v1",
            "patch_receipt": receipt,
            "touched_paths": [relative],
            "operation_count": 1,
            "runtime_tests": [],
        }

    result = _parallel_generate(
        owner,
        single_generate,
        root,
        args=(),
        kwargs={
            "module": module,
            "minecraft_version": platform.minecraft_version,
            "loader": platform.loader,
            "mappings": platform.yarn_mappings,
        },
        search_module=custom_search,
    )

    assert result["agentic_generation_search"]["candidate_count"] == 2
    assert result["agentic_generation_search"]["candidate_workers"] == 2
    assert result["patch_receipt"]["status"] == "APPLIED"
    assert result["generation_checkpoint_acknowledged"] is True
    assert acknowledgements == [result]
    assert releases == []
    assert len(discards) == 1
    assert (root / "src/main/java/example/ParallelCapture.java").read_text(
        encoding="utf-8"
    ) == "package example;\nfinal class ParallelCapture {}\n"
    assert owner.router.__class__ is _Router


def test_legacy_width_two_search_acknowledges_only_after_live_commit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MMM_AGENTIC_SEARCH", "on")
    monkeypatch.setenv("MMM_CUSTOM_SEARCH_WIDTH", "2")
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    monkeypatch.setenv("MMM_CUSTOM_CANDIDATE_JDT", "off")

    root = tmp_path / "project"
    root.mkdir()
    relative = "src/main/java/example/LegacySearch.java"
    acknowledgement_states: list[bool] = []
    releases: list[dict] = []
    discards: list[dict] = []

    class _Generator:
        def __init__(self) -> None:
            self.router = _Router()
            self.policy = ScalePolicy(model_context_bytes=4096)
            self._cached_index = None
            self._cached_root = None

        def generate(
            self,
            project_root,
            *,
            module,
            research_modules=(),
            minecraft_version=None,
            loader=None,
            mappings=None,
        ):
            receipt = TransactionalSourcePatcher(project_root).apply(
                [
                    {
                        "operation": "create",
                        "path": relative,
                        "content": "package example;\nfinal class LegacySearch {}\n",
                    }
                ]
            )
            return {
                "patch_receipt": receipt,
                "touched_paths": [relative],
                "operation_count": 1,
                "generation_checkpoint": {
                    "schema_version": "mmm/custom-module-checkpoint-v2",
                    "status": "AWAITING_LIVE_COMMIT",
                    "cleanup_token": "a" * 64,
                },
            }

        def acknowledge_generation_checkpoint(self, result) -> bool:
            acknowledgement_states.append((root / relative).is_file())
            return True

        def release_generation_checkpoint(self, result) -> bool:
            releases.append(result)
            return True

        def discard_generation_checkpoint(self, result) -> bool:
            discards.append(result)
            return True

    module_api = SimpleNamespace(CustomModuleGenerator=_Generator)
    custom_search.install(module_api)
    platform = adapter_for_target("1.20.1", "fabric")
    result = _Generator().generate(
        root,
        module=SimpleNamespace(
            kind="custom_java",
            config={"feature": "legacy search commit"},
            depends_on=(),
            required_gates=(),
        ),
        minecraft_version=platform.minecraft_version,
        loader=platform.loader,
        mappings=platform.yarn_mappings,
    )

    assert result["generation_checkpoint_acknowledged"] is True
    assert acknowledgement_states == [True]
    assert releases == []
    assert len(discards) == 1
    assert (root / relative).read_text(encoding="utf-8") == (
        "package example;\nfinal class LegacySearch {}\n"
    )
