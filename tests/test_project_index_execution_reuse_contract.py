from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import project_index_execution_reuse_contract as contract


class _FakeIndex:
    constructions = 0

    def __init__(self, project_root, *, policy=None) -> None:
        type(self).constructions += 1
        self.project_root = str(project_root)
        self.policy = policy
        self.updated: list[tuple[str, ...]] = []
        self.manifests = 0

    def update_files(self, paths) -> None:
        self.updated.append(tuple(str(path) for path in paths))

    def write_manifest(self) -> None:
        self.manifests += 1


class _FakeOrchestrator:
    def __init__(self, module, *, unknown_receipt: bool = False) -> None:
        self.module = module
        self.policy = object()
        self.heap_policy = object()
        self.unknown_receipt = unknown_receipt

    def _execute_generation_work(self):
        return {"status": "SUCCEEDED"}

    def execute(self, project_root):
        self._execute_generation_work()
        first = self.module.ProjectIndex(project_root, policy=self.policy)
        self.module.tune_gradle_resources(
            project_root,
            policy=self.heap_policy,
            unknown_receipt=self.unknown_receipt,
        )
        second = self.module.ProjectIndex(project_root, policy=self.policy)
        second.write_manifest()
        return first, second


def _module():
    module = SimpleNamespace()
    module.ProjectIndex = _FakeIndex

    def tune_gradle_resources(project_root, *, policy=None, unknown_receipt=False):
        if unknown_receipt:
            return {"status": "TUNED"}
        return {
            "status": "TUNED",
            "receipts": [
                {
                    "operations": [
                        {
                            "operation": "replace",
                            "path": "gradle.properties",
                        },
                        {
                            "operation": "replace",
                            "path": "src/main/resources/fabric.mod.json",
                        },
                    ]
                }
            ],
        }

    module.tune_gradle_resources = tune_gradle_resources

    class BoundOrchestrator(_FakeOrchestrator):
        def __init__(self, *, unknown_receipt: bool = False) -> None:
            super().__init__(module, unknown_receipt=unknown_receipt)

    module.CompleteProductionOrchestrator = BoundOrchestrator
    return module


def test_tuning_updates_cached_index_even_with_derived_policy(tmp_path) -> None:
    _FakeIndex.constructions = 0
    module = _module()
    contract.install(module)

    first, second = module.CompleteProductionOrchestrator().execute(tmp_path)

    assert first is second
    assert _FakeIndex.constructions == 1
    assert first.updated == [
        (
            "gradle.properties",
            "src/main/resources/fabric.mod.json",
        )
    ]
    assert second.manifests == 1


def test_unknown_mutating_receipt_falls_back_to_fresh_index(tmp_path) -> None:
    _FakeIndex.constructions = 0
    module = _module()
    contract.install(module)

    first, second = module.CompleteProductionOrchestrator(
        unknown_receipt=True
    ).execute(tmp_path)

    assert first is not second
    assert _FakeIndex.constructions == 2
    assert first.updated == []
    assert second.manifests == 1


def test_execution_cache_never_leaks_between_runs(tmp_path) -> None:
    _FakeIndex.constructions = 0
    module = _module()
    contract.install(module)
    orchestrator = module.CompleteProductionOrchestrator()

    first_run = orchestrator.execute(tmp_path)
    second_run = orchestrator.execute(tmp_path)

    assert first_run[0] is first_run[1]
    assert second_run[0] is second_run[1]
    assert first_run[0] is not second_run[0]
    assert _FakeIndex.constructions == 2


def test_pre_generation_index_construction_is_not_cached(tmp_path) -> None:
    _FakeIndex.constructions = 0
    module = _module()
    contract.install(module)

    before_one = module.ProjectIndex(tmp_path, policy=object())
    before_two = module.ProjectIndex(tmp_path, policy=object())

    assert before_one is not before_two
    assert _FakeIndex.constructions == 2
