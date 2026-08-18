from __future__ import annotations

from minecraft_mod_ai.qwen_model_profiles import (
    QWEN35_GENERAL_THINKING,
    QWEN35_NON_THINKING,
    QWEN35_PRECISE_CODING,
    QWEN36_27B_GENERAL_THINKING,
    QWEN36_35B_A3B_GENERAL_THINKING,
    QWEN36_NON_THINKING,
    QWEN36_PRECISE_CODING,
    qwen_family,
    qwen_registry_model,
    qwen_sampling_profile,
)


def test_registry_models_classify_all_three_production_qwen_models() -> None:
    assert (
        qwen_registry_model("unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
        == "qwen3.6-35b-a3b"
    )
    assert qwen_registry_model("unsloth/Qwen3.6-27B-MTP-GGUF") == "qwen3.6-27b"
    assert qwen_registry_model("unsloth/Qwen3.5-9B-MTP-GGUF") == "qwen3.5-9b"


def test_qwen36_27b_quantizations_share_one_logical_model_identity() -> None:
    assert (
        qwen_registry_model("", "Qwen3.6-27B-UD-Q4_K_XL.gguf")
        == "qwen3.6-27b"
    )
    assert qwen_registry_model("", "Qwen3.6-27B-Q3_K_M.gguf") == "qwen3.6-27b"


def test_family_detection_remains_forward_compatible_with_unknown_variants() -> None:
    assert qwen_family("Qwen3.6-future-variant") == "qwen3.6"
    assert qwen_family("Qwen3.5-future-variant") == "qwen3.5"
    assert qwen_family("generic/model") is None


def test_conflicting_model_and_filename_do_not_cross_qwen_families() -> None:
    assert (
        qwen_registry_model(
            "unsloth/Qwen3.6-future-variant",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        )
        is None
    )
    assert (
        qwen_family(
            "unsloth/Qwen3.6-future-variant",
            "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        )
        == "qwen3.6"
    )


def test_shared_profiles_preserve_established_vendor_sampling_contracts() -> None:
    assert QWEN35_GENERAL_THINKING == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    }
    assert QWEN35_PRECISE_CODING == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
    assert QWEN35_NON_THINKING == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "reasoning_effort": "none",
    }
    assert QWEN36_PRECISE_CODING == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }
    assert QWEN36_NON_THINKING == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "reasoning_effort": "none",
    }


def test_qwen36_general_thinking_is_model_specific() -> None:
    assert QWEN36_27B_GENERAL_THINKING["presence_penalty"] == 0.0
    assert QWEN36_35B_A3B_GENERAL_THINKING["presence_penalty"] == 1.5
    assert (
        qwen_sampling_profile(
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            mode="general_thinking",
        )
        == QWEN36_27B_GENERAL_THINKING
    )
    assert (
        qwen_sampling_profile(
            "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
            mode="general_thinking",
        )
        == QWEN36_35B_A3B_GENERAL_THINKING
    )
    assert (
        qwen_sampling_profile(
            "unsloth/Qwen3.5-9B-MTP-GGUF",
            mode="general_thinking",
        )
        == QWEN35_GENERAL_THINKING
    )


def test_precise_coding_is_shared_only_where_vendor_contract_matches() -> None:
    expected = QWEN35_PRECISE_CODING
    assert QWEN36_PRECISE_CODING == expected
    for model_id in (
        "unsloth/Qwen3.5-9B-MTP-GGUF",
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    ):
        assert qwen_sampling_profile(model_id, mode="precise_coding") == expected


def test_sampling_selector_does_not_guess_future_model_profiles() -> None:
    assert qwen_sampling_profile("Qwen3.6-future-variant", mode="general_thinking") is None


def test_sampling_selector_returns_fresh_payload() -> None:
    first = qwen_sampling_profile("Qwen3.6-27B", mode="general_thinking")
    second = qwen_sampling_profile("Qwen3.6-27B", mode="general_thinking")
    assert first == second == QWEN36_27B_GENERAL_THINKING
    assert first is not second
    assert first is not None
    first["presence_penalty"] = 99.0
    assert second is not None
    assert second["presence_penalty"] == 0.0
