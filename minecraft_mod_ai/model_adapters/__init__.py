from .base import (
    AdapterConfig,
    GenerationRequest,
    GenerationResponse,
    HardwarePreflightError,
    ModelBackendError,
    ModelConfigurationError,
    ToolCall,
)
from .embedding import EmbeddingAdapter
from .image_diffusion import ImageDiffusionAdapter
from . import llama_cpp_adapter as _llama_cpp_adapter
from .qwen_tool_parser import parse_qwen_tool_markup
from .openai_compatible import OpenAICompatibleAdapter
from .reranker import RerankerAdapter
from .transformers_multimodal import TransformersMultimodalAdapter
from .transformers_text import TransformersTextAdapter

# Install the strict schema-guided parser before any caller can obtain the adapter.
# Direct imports of the submodule also execute this package initializer first.
_llama_cpp_adapter._parse_qwen_tool_markup = parse_qwen_tool_markup
LlamaCppAdapter = _llama_cpp_adapter.LlamaCppAdapter

__all__ = [
    "AdapterConfig",
    "EmbeddingAdapter",
    "GenerationRequest",
    "GenerationResponse",
    "HardwarePreflightError",
    "ImageDiffusionAdapter",
    "LlamaCppAdapter",
    "ModelBackendError",
    "ModelConfigurationError",
    "OpenAICompatibleAdapter",
    "RerankerAdapter",
    "ToolCall",
    "TransformersMultimodalAdapter",
    "TransformersTextAdapter",
]
