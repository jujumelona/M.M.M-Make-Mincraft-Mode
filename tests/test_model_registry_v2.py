from pathlib import Path

import pytest

from minecraft_mod_ai.model_adapters import ModelConfigurationError
from minecraft_mod_ai.model_registry import ModelRegistry


T4_QUANTIZED_QWEN_ROLES = {
    "planner",
    "researcher",
    "coder",
    "coder_safe",
    "visual_critic",
}


def test_t4_registry_has_role_specific_real_model_ids() -> None:
    registry = ModelRegistry()
    profile = registry.load_profile("t4_local")
    assert profile.roles["planner"].model_id == "Qwen/Qwen3.5-4B"
    assert profile.roles["planner"].adapter == "transformers_multimodal"
    assert profile.roles["planner"].max_context == 262144
    assert profile.roles["planner"].max_new_tokens == 4096
    assert profile.roles["coder"].model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert profile.roles["coder_safe"].model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert profile.roles["coder"].adapter == "transformers_text"
    assert profile.roles["embedding"].model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert profile.roles["reranker"].model_id == "Qwen/Qwen3-Reranker-0.6B"
    assert profile.roles["visual_critic"].max_context == 262144
    assert profile.roles["image_generator"].exclusive_gpu is True
    assert not any(
        config.model_id.endswith("Qwen3.5-4B-Instruct")
        or config.model_id.endswith("Qwen3.5-9B-Instruct")
        for config in profile.roles.values()
    )


def test_t4_quality_uses_quantized_qwen35_9b_with_bounded_context() -> None:
    profile = ModelRegistry().load_profile("t4_quality")
    planner = profile.roles["planner"]

    assert planner.model_id == "Qwen/Qwen3.5-9B"
    assert planner.adapter == "transformers_multimodal"
    assert planner.quantization == "bnb_4bit"
    assert planner.torch_dtype == "float16"
    assert planner.max_context == 262144
    assert planner.max_new_tokens == 4096
    assert planner.min_free_vram_mb == 10500
    assert set(profile.roles) == {
        "planner",
        "researcher",
        "coder",
        "coder_safe",
        "visual_critic",
        "embedding",
        "reranker",
        "image_generator",
        "speech_recognition",
    }


def test_t4_quality_non_planner_roles_match_t4_local_except_visual_critic() -> None:
    registry = ModelRegistry()
    local = registry.load_profile("t4_local")
    quality = registry.load_profile("t4_quality")

    for role in set(local.roles) - {"planner", "visual_critic"}:
        assert quality.roles[role] == local.roles[role]
    assert quality.roles["visual_critic"].model_id == "Qwen/Qwen3.5-9B"
    assert quality.roles["visual_critic"].quantization == "bnb_4bit"
    assert quality.roles["visual_critic"].torch_dtype == "float16"
    assert quality.roles["visual_critic"].max_context == 262144


@pytest.mark.parametrize(
    "registry_path",
    ["config/model_registry.yaml", "minecraft_mod_ai/config/model_registry.yaml"],
)
@pytest.mark.parametrize("profile_name", ["t4_local", "t4_quality"])
def test_t4_quantized_qwen_roles_force_fp16(
    registry_path: str,
    profile_name: str,
) -> None:
    profile = ModelRegistry(registry_path).load_profile(profile_name)
    quantized_qwen = {
        role: config
        for role, config in profile.roles.items()
        if config.model_id.startswith("Qwen/")
        and config.quantization == "bnb_4bit"
    }

    assert set(quantized_qwen) == T4_QUANTIZED_QWEN_ROLES
    assert {config.torch_dtype for config in quantized_qwen.values()} == {"float16"}


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


def test_repository_and_packaged_planner_budgets_stay_in_sync() -> None:
    repository = ModelRegistry("config/model_registry.yaml").role(
        "t4_local", "planner"
    )
    packaged = ModelRegistry(
        "minecraft_mod_ai/config/model_registry.yaml"
    ).role("t4_local", "planner")

    assert repository.max_context == packaged.max_context == 262144
    assert repository.max_new_tokens == packaged.max_new_tokens == 4096

    repository_quality = ModelRegistry("config/model_registry.yaml").role(
        "t4_quality", "planner"
    )
    packaged_quality = ModelRegistry(
        "minecraft_mod_ai/config/model_registry.yaml"
    ).role("t4_quality", "planner")
    assert repository_quality == packaged_quality
