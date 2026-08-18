from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.llama_tuning_pipeline import NativeLlamaTuningPipeline


def test_multimodal_upgrade_is_last_tuning_stage() -> None:
    pipeline = NativeLlamaTuningPipeline(
        autotune=SimpleNamespace(),
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )

    names = [stage.name for stage in pipeline.stages()]

    assert names[-1] == "multimodal"
    assert names.index("multimodal") > names.index("decode-speed")
    assert names.index("multimodal") > names.index("kernel-autotune")
