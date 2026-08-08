from pathlib import Path

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_registry import ModelRegistry


def test_t4_registry_has_role_specific_real_model_ids() -> None:
    registry = ModelRegistry()
    profile = registry.load_profile("t4_local")
    assert profile.roles["planner"].model_id == "Qwen/Qwen3.5-4B"
    assert profile.roles["planner"].adapter == "transformers_multimodal"
    assert profile.roles["coder"].model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert profile.roles["coder_safe"].model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert profile.roles["coder"].adapter == "transformers_text"
    assert profile.roles["embedding"].model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert profile.roles["reranker"].model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert profile.roles["image_generator"].exclusive_gpu is True
    assert not any(
        config.model_id.endswith("Qwen3.5-4B-Instruct")
        or config.model_id.endswith("Qwen3.5-9B-Instruct")
        for config in profile.roles.values()
    )


def test_invalid_qwen35_instruct_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    path.write_text(
        """
schema_version: mmm/model-registry-v1
profiles:
  broken:
    roles:
      planner: {model_id: Qwen/Qwen3.5-4B-Instruct, adapter: transformers_multimodal}
      researcher: {model_id: a/b, adapter: transformers_text}
      coder: {model_id: Qwen/Qwen2.5-Coder-3B-Instruct, adapter: transformers_text}
      visual_critic: {model_id: a/b, adapter: transformers_multimodal}
      image_generator: {model_id: a/b, adapter: image_diffusion}
      speech_recognition: {model_id: a/b, adapter: speech}
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ModelConfigurationError, match="Invalid Qwen3.5"):
        ModelRegistry(path).load_profile("broken")


def test_local_coder_must_be_code_specialized(tmp_path: Path) -> None:
    path = tmp_path / "registry.yaml"
    text = Path("config/model_registry.yaml").read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen3-4B-Instruct-2507",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelConfigurationError, match="code-specialized"):
        ModelRegistry(path).load_profile("t4_local")
