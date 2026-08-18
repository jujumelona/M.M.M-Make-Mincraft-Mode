from __future__ import annotations

from pathlib import Path


_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "model_registry.yaml"


def _anchor_block(anchor: str) -> str:
    text = _REGISTRY.read_text(encoding="utf-8")
    marker = f"{anchor}: &{anchor.removeprefix('_')}\n"
    assert marker in text, f"missing model registry anchor: {anchor}"
    return text.split(marker, 1)[1].split("\n\n", 1)[0]


def test_three_production_qwen_models_keep_native_context_and_projector() -> None:
    expected = {
        "_qwen35_9b": (
            "unsloth/Qwen3.5-9B-MTP-GGUF",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        ),
        "_qwen36_27b_q4": (
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        ),
        "_qwen36_27b_q3": (
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "Qwen3.6-27B-Q3_K_M.gguf",
        ),
        "_qwen36_35b": (
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        ),
    }

    for anchor, (model_id, gguf_filename) in expected.items():
        block = _anchor_block(anchor)
        assert f"model_id: {model_id}" in block
        assert f"gguf_filename: {gguf_filename}" in block
        assert "mmproj_filename: mmproj-F16.gguf" in block
        assert "max_context: 262144" in block
