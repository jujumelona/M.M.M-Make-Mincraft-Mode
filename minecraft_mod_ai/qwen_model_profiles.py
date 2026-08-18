from __future__ import annotations

"""Shared Qwen model identity and request sampling profiles.

Keep model classification and vendor-recommended sampling values in one dependency-
free module. Runtime wrappers may add mode-specific switches, but they must not each
re-encode model names or duplicate the underlying sampling profiles.
"""

from functools import lru_cache
from typing import Final, Literal

QwenFamily = Literal["qwen3.5", "qwen3.6"]
QwenRegistryModel = Literal["qwen3.5-9b", "qwen3.6-27b", "qwen3.6-35b-a3b"]
QwenSamplingMode = Literal["general_thinking", "precise_coding", "non_thinking"]

QWEN35_GENERAL_THINKING: Final = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}
QWEN35_PRECISE_CODING: Final = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
QWEN35_NON_THINKING: Final = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "reasoning_effort": "none",
}

# Qwen3.6 precise coding is intentionally shared by 27B dense and 35B-A3B MoE.
QWEN36_PRECISE_CODING: Final = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
# General thinking is not shared: the production 27B and 35B-A3B recommendations
# differ in presence penalty.
QWEN36_27B_GENERAL_THINKING: Final = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
QWEN36_35B_A3B_GENERAL_THINKING: Final = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
}
QWEN36_NON_THINKING: Final = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "reasoning_effort": "none",
}


def _normalize(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


@lru_cache(maxsize=128)
def _classify(normalized: str) -> tuple[QwenFamily | None, QwenRegistryModel | None]:
    if "qwen36" in normalized:
        if "35ba3b" in normalized:
            return "qwen3.6", "qwen3.6-35b-a3b"
        if "27b" in normalized:
            return "qwen3.6", "qwen3.6-27b"
        return "qwen3.6", None
    if "qwen35" in normalized:
        if "9b" in normalized:
            return "qwen3.5", "qwen3.5-9b"
        return "qwen3.5", None
    return None, None


def _identity(
    model_id: object,
    gguf_filename: object = "",
) -> tuple[QwenFamily | None, QwenRegistryModel | None]:
    """Classify the configured model from either registry id or GGUF filename."""

    model = _classify(_normalize(model_id))
    if model[1] is not None:
        return model
    filename = _classify(_normalize(gguf_filename))
    if filename[1] is not None:
        # Never cross families when a generic model id and filename disagree. A
        # mismatched registry should fail classification rather than silently route
        # requests through another family's policy.
        if model[0] is not None and filename[0] != model[0]:
            return model[0], None
        return filename
    return model if model[0] is not None else filename


def qwen_family(model_id: object, gguf_filename: object = "") -> QwenFamily | None:
    return _identity(model_id, gguf_filename)[0]


def qwen_registry_model(
    model_id: object,
    gguf_filename: object = "",
) -> QwenRegistryModel | None:
    """Return one of the three production Qwen model identities in model_registry."""

    return _identity(model_id, gguf_filename)[1]


def qwen_sampling_profile(
    model_id: object,
    gguf_filename: object = "",
    *,
    mode: QwenSamplingMode,
) -> dict[str, float | str] | None:
    """Return a fresh sampling payload for an exact production model and mode.

    Returning ``None`` for an unknown production identity is deliberate: family-only
    detection remains useful for compatible transport behavior, but vendor sampling
    must not be guessed for a future model variant.
    """

    model = qwen_registry_model(model_id, gguf_filename)
    if model is None:
        return None

    if mode == "precise_coding":
        profile = QWEN35_PRECISE_CODING if model == "qwen3.5-9b" else QWEN36_PRECISE_CODING
    elif mode == "non_thinking":
        profile = QWEN35_NON_THINKING if model == "qwen3.5-9b" else QWEN36_NON_THINKING
    elif mode == "general_thinking":
        if model == "qwen3.5-9b":
            profile = QWEN35_GENERAL_THINKING
        elif model == "qwen3.6-27b":
            profile = QWEN36_27B_GENERAL_THINKING
        else:
            profile = QWEN36_35B_A3B_GENERAL_THINKING
    else:
        raise ValueError(f"unsupported Qwen sampling mode: {mode!r}")
    return dict(profile)


def is_qwen35_9b(model_id: object, gguf_filename: object = "") -> bool:
    return qwen_registry_model(model_id, gguf_filename) == "qwen3.5-9b"


__all__ = [
    "QWEN35_GENERAL_THINKING",
    "QWEN35_NON_THINKING",
    "QWEN35_PRECISE_CODING",
    "QWEN36_27B_GENERAL_THINKING",
    "QWEN36_35B_A3B_GENERAL_THINKING",
    "QWEN36_NON_THINKING",
    "QWEN36_PRECISE_CODING",
    "QwenFamily",
    "QwenRegistryModel",
    "QwenSamplingMode",
    "is_qwen35_9b",
    "qwen_family",
    "qwen_registry_model",
    "qwen_sampling_profile",
]
