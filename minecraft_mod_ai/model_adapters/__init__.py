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
from .llama_cpp_adapter import LlamaCppAdapter
from .llama_turn_retry import install_llama_turn_retry
from .openai_compatible import OpenAICompatibleAdapter
from .reranker import RerankerAdapter
from .transformers_multimodal import TransformersMultimodalAdapter
from .transformers_text import TransformersTextAdapter

# llama_cpp_adapter already keys this cache by the managed server generation identity
# plus the exact template-calibration payload. Keep the bounded cache on the adapter
# class so request-local adapter instances share safe calibration results instead of
# issuing the same /apply-template probe again. Tests or specialized callers can still
# shadow it with an instance-local dictionary when isolation is required.
LlamaCppAdapter._prefill_template_prefix_cache = {}

install_llama_turn_retry(LlamaCppAdapter)

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