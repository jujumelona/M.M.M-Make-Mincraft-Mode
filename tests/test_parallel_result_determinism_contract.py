from types import SimpleNamespace

from minecraft_mod_ai.parallel_result_determinism_contract import (
    _canonicalize_generation_result,
    _normalize_synthesized_batches,
    install,
)


def test_audio_batches_are_flattened_in_sound_id_order() -> None:
    batches = [
        {"synthesized": [{"sound_id": "zeta", "size_bytes": 3}]},
        {
            "synthesized": [
                {"sound_id": "beta", "size_bytes": 2},
                {"sound_id": "alpha", "size_bytes": 1},
            ]
        },
    ]

    normalized = _normalize_synthesized_batches(batches)

    assert [
        item["sound_id"] for item in normalized[0]["synthesized"]
    ] == ["alpha", "beta", "zeta"]


def test_generation_receipt_lists_are_canonicalized() -> None:
    result = {
        "module_receipts": [
            {"module_id": "z"},
            {"module_id": "a"},
        ],
        "blockbench_receipts": [
            {"entity_id": "wolf"},
            {"entity_id": "bee"},
        ],
        "asset_receipt": {
            "shards": [
                {"assets": [{"asset_id": "z"}]},
                {"assets": [{"asset_id": "a"}]},
            ]
        },
        "audio_receipt": {
            "shards": [
                {"sounds": [{"sound_id": "z"}]},
                {"sounds": [{"sound_id": "a"}]},
            ]
        },
        "unresolved": ["z", "a"],
    }

    canonical = _canonicalize_generation_result(result)

    assert [item["module_id"] for item in canonical["module_receipts"]] == ["a", "z"]
    assert [item["entity_id"] for item in canonical["blockbench_receipts"]] == ["bee", "wolf"]
    assert canonical["unresolved"] == ["a", "z"]
    assert canonical["asset_receipt"]["shards"][0]["assets"][0]["asset_id"] == "a"
    assert canonical["audio_receipt"]["shards"][0]["sounds"][0]["sound_id"] == "a"


def test_install_patches_orchestrator_bound_finalizer_and_sorts_before_finalize() -> None:
    observed = {}

    def original_finalizer(*, synthesized_batches, **_kwargs):
        observed["ids"] = [
            item["sound_id"]
            for batch in synthesized_batches
            for item in batch.get("synthesized", [])
        ]
        return {
            "sounds": [
                {"sound_id": sound_id}
                for sound_id in observed["ids"]
            ]
        }

    class FakeOrchestrator:
        def _execute_generation_work(self):
            return {
                "module_receipts": [
                    {"module_id": "z"},
                    {"module_id": "a"},
                ],
                "unresolved": ["z", "a"],
            }

    audio = SimpleNamespace(finalize_audio_registry=original_finalizer)
    orchestrator = SimpleNamespace(
        finalize_audio_registry=original_finalizer,
        CompleteProductionOrchestrator=FakeOrchestrator,
    )

    install(
        audio_generator_module=audio,
        orchestrator_module=orchestrator,
    )

    receipt = orchestrator.finalize_audio_registry(
        synthesized_batches=[
            {"synthesized": [{"sound_id": "z"}]},
            {"synthesized": [{"sound_id": "a"}]},
        ]
    )
    assert observed["ids"] == ["a", "z"]
    assert [item["sound_id"] for item in receipt["sounds"]] == ["a", "z"]

    result = FakeOrchestrator()._execute_generation_work()
    assert [item["module_id"] for item in result["module_receipts"]] == ["a", "z"]
    assert result["unresolved"] == ["a", "z"]
