from __future__ import annotations

"""Registry-declared Qwen chat-template and tool transport capabilities.

Qwen3.5, Qwen3.6, and Qwen3.8 share tagged ``qwen3_coder`` tool markup, but
their reasoning-history controls are not interchangeable.  Runtime code must use
these declared capabilities instead of inferring a family from a model id or GGUF
filename.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from .model_adapters.base import ModelConfigurationError

_TOOL_MARKUP = "qwen3_coder_xml"
_ACTION_CONTROL = "enable_thinking_false"


@dataclass(frozen=True)
class QwenFamilyCapabilities:
    family: str
    tool_markup: str
    action_thinking_control: str
    preserve_thinking: bool
    reasoning_effort: bool
    assistant_prefill: bool

    def action_template_kwargs(self) -> dict[str, bool]:
        result = {"enable_thinking": False}
        if self.preserve_thinking:
            result["preserve_thinking"] = False
        return result


_OFFICIAL_CAPABILITIES: dict[str, QwenFamilyCapabilities] = {
    "qwen3.5": QwenFamilyCapabilities(
        family="qwen3.5",
        tool_markup=_TOOL_MARKUP,
        action_thinking_control=_ACTION_CONTROL,
        preserve_thinking=False,
        reasoning_effort=False,
        assistant_prefill=True,
    ),
    "qwen3.6": QwenFamilyCapabilities(
        family="qwen3.6",
        tool_markup=_TOOL_MARKUP,
        action_thinking_control=_ACTION_CONTROL,
        preserve_thinking=True,
        reasoning_effort=False,
        assistant_prefill=True,
    ),
    "qwen3.8": QwenFamilyCapabilities(
        family="qwen3.8",
        tool_markup=_TOOL_MARKUP,
        action_thinking_control=_ACTION_CONTROL,
        preserve_thinking=True,
        reasoning_effort=True,
        assistant_prefill=True,
    ),
}


def _extra(config: Any) -> Mapping[str, Any]:
    raw = getattr(config, "extra", {})
    return raw if isinstance(raw, Mapping) else {}


def qwen_family_capabilities(
    config: Any,
    *,
    required: bool = False,
) -> QwenFamilyCapabilities | None:
    """Return and validate the exact registry-declared Qwen family contract."""

    extra = _extra(config)
    if str(extra.get("runtime_contract", "")).strip().casefold() != "qwen":
        return None

    family = str(extra.get("qwen_family", "")).strip().casefold()
    expected = _OFFICIAL_CAPABILITIES.get(family)
    if expected is None:
        raise ModelConfigurationError(
            "registry-declared Qwen runtime requires qwen_family to be one of "
            "qwen3.5, qwen3.6, or qwen3.8"
        )

    declared = QwenFamilyCapabilities(
        family=family,
        tool_markup=str(extra.get("qwen_tool_markup", "")).strip().casefold(),
        action_thinking_control=str(
            extra.get("qwen_action_thinking_control", "")
        ).strip().casefold(),
        preserve_thinking=extra.get("qwen_preserve_thinking") is True,
        reasoning_effort=extra.get("qwen_reasoning_effort") is True,
        assistant_prefill=extra.get("qwen_assistant_prefill") is True,
    )
    if declared != expected:
        raise ModelConfigurationError(
            f"registry Qwen family contract does not match supported {family} "
            "chat-template/tool capabilities"
        )
    return declared


def qwen_family_name(config: Any) -> str:
    capabilities = qwen_family_capabilities(config, required=False)
    return capabilities.family if capabilities is not None else ""


__all__ = [
    "QwenFamilyCapabilities",
    "qwen_family_capabilities",
    "qwen_family_name",
]
