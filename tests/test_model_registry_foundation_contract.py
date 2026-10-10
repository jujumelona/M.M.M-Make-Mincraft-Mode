from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import AdapterConfig, ModelConfigurationError
from minecraft_mod_ai.model_registry import (
    _completion_budget,
    _validate_single_foundation_model,
)


def _config(role: str, model_id: str, *, base_url: str = "") -> AdapterConfig:
    return AdapterConfig(
        role=role,
        adapter="llama_cpp" if not base_url else "openai_compatible",
        provider="local" if not base_url else "openai_compatible",
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=4096,
    )


@pytest.mark.parametrize("model_id", ["bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF", "unsloth/Qwen3.8-27B-GGUF"])
def test_selected_foundation_models_share_one_identity_across_roles(model_id: str):
    roles = {
        role: _config(role, model_id)
        for role in ("planner", "researcher", "coder", "coder_safe")
    }
    _validate_single_foundation_model("selected", roles)


def test_role_split_foundation_models_are_rejected():
    roles = {
        "planner": _config("planner", "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"),
        "researcher": _config("researcher", "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"),
        "coder": _config("coder", "unsloth/Qwen3.8-27B-GGUF"),
        "coder_safe": _config("coder_safe", "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"),
    }
    with pytest.raises(ModelConfigurationError, match="one selected foundation model"):
        _validate_single_foundation_model("split", roles)


def test_role_split_remote_endpoints_are_rejected_even_with_same_model_name():
    roles = {
        "planner": _config("planner", "Qwen", base_url="http://one/v1"),
        "researcher": _config("researcher", "Qwen", base_url="http://one/v1"),
        "coder": _config("coder", "Qwen", base_url="http://two/v1"),
        "coder_safe": _config("coder_safe", "Qwen", base_url="http://one/v1"),
    }
    with pytest.raises(ModelConfigurationError, match="one selected foundation model"):
        _validate_single_foundation_model("split-endpoint", roles)


def test_dedicated_embedding_or_image_models_do_not_create_foundation_split():
    roles = {
        role: _config(role, "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF")
        for role in ("planner", "researcher", "coder", "coder_safe")
    }
    roles["embedding"] = AdapterConfig(role="embedding", adapter="embedding", model_id="embed")
    roles["image_generator"] = AdapterConfig(role="image_generator", adapter="image_diffusion", model_id="image")
    _validate_single_foundation_model("with-tools", roles)


def test_unlimited_completion_budget_is_forbidden():
    with pytest.raises(ModelConfigurationError, match="unlimited generation is forbidden"):
        _completion_budget(-1, "coder.max_new_tokens")
    with pytest.raises(ModelConfigurationError):
        _completion_budget(0, "coder.max_new_tokens")
    assert _completion_budget(8192, "coder.max_new_tokens") == 8192
