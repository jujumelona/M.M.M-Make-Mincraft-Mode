from __future__ import annotations

from minecraft_mod_ai import project_index_execution_reuse_contract as contract


class _FakeIndex:
    constructions = 0

    def __init__(self, project_root, *, policy=None) -> None:
        type(self).constructions += 1
        self.project_root = str(project_root)
        self.policy = policy
        self.updated: list[tuple[str, ...]] = []

    def update_files(self, paths) -> None:
        self.updated.append(tuple(str(path) for path in paths))


def _known_receipt() -> dict[str, object]:
    return {
        "status": "TUNED",
        "receipts": [
            {
                "operations": [
                    {"operation": "replace", "path": "gradle.properties"},
                    {
                        "operation": "replace",
                        "path": "src/main/resources/fabric.mod.json",
                    },
                ]
            }
        ],
    }


def test_receipt_updates_cached_post_generation_index(tmp_path) -> None:
    _FakeIndex.constructions = 0
    policy = object()

    @contract.execution_scoped
    def run():
        contract.mark_post_generation()
        first = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        contract._update_from_receipt(tmp_path, _known_receipt())
        second = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        return first, second

    first, second = run()

    assert first is second
    assert _FakeIndex.constructions == 1
    assert first.updated == [
        (
            "gradle.properties",
            "src/main/resources/fabric.mod.json",
        )
    ]


def test_unknown_mutating_receipt_evicts_cached_index(tmp_path) -> None:
    _FakeIndex.constructions = 0
    policy = object()

    @contract.execution_scoped
    def run():
        contract.mark_post_generation()
        first = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        contract._update_from_receipt(tmp_path, {"status": "TUNED"})
        second = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        return first, second

    first, second = run()

    assert first is not second
    assert _FakeIndex.constructions == 2
    assert first.updated == []


def test_execution_cache_never_leaks_between_runs(tmp_path) -> None:
    _FakeIndex.constructions = 0
    policy = object()

    @contract.execution_scoped
    def run():
        contract.mark_post_generation()
        first = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        second = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        return first, second

    first_run = run()
    second_run = run()

    assert first_run[0] is first_run[1]
    assert second_run[0] is second_run[1]
    assert first_run[0] is not second_run[0]
    assert _FakeIndex.constructions == 2


def test_pre_generation_index_construction_is_not_cached(tmp_path) -> None:
    _FakeIndex.constructions = 0
    policy = object()

    @contract.execution_scoped
    def run():
        before_one = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        before_two = contract.project_index(_FakeIndex, tmp_path, policy=policy)
        return before_one, before_two

    before_one, before_two = run()

    assert before_one is not before_two
    assert _FakeIndex.constructions == 2
