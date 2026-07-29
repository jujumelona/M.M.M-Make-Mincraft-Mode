from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_adapters import (
    EmbeddingAdapter,
    GenerationRequest,
    ImageDiffusionAdapter,
    ModelConfigurationError,
    OpenAICompatibleAdapter,
    RerankerAdapter,
    SpeechAdapter,
    TransformersMultimodalAdapter,
    TransformersTextAdapter,
)
from .model_registry import ModelRegistry


_GPU_EXCLUSIVE_LOCK = threading.RLock()


class ModelRouter:
    """Role router with strict profile selection and no silent backend fallback."""

    def __init__(
        self,
        *,
        profile: str = "t4_local",
        registry: ModelRegistry | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.profile = profile
        self.registry.load_profile(profile)

    def generate_text(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        media_paths: Sequence[str | Path] = (),
        response_format: str = "text",
    ) -> str:
        config = self.registry.role(self.profile, role)
        if config.adapter == "transformers_text":
            adapter = TransformersTextAdapter(config)
        elif config.adapter == "transformers_multimodal":
            adapter = TransformersMultimodalAdapter(config)
        elif config.adapter == "openai_compatible":
            adapter = OpenAICompatibleAdapter(config)
        else:
            raise ModelConfigurationError(
                f"Role {role!r} cannot generate text with adapter {config.adapter!r}."
            )
        request = GenerationRequest(
            messages=messages,
            media_paths=tuple(Path(path) for path in media_paths),
            response_format=response_format,
        )
        with self._gpu_scope(config.exclusive_gpu):
            return adapter.generate(request)

    def embed(self, texts: Sequence[str], role: str = "embedding") -> list[list[float]]:
        config = self.registry.role(self.profile, role)
        if config.adapter != "embedding":
            raise ModelConfigurationError(
                f"Role {role!r} does not expose an embedding adapter."
            )
        return EmbeddingAdapter(config).embed(texts)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        role: str = "reranker",
        instruction: str = (
            "Retrieve Minecraft Fabric 1.20.1 and Yarn 1.20.1 evidence that directly "
            "answers the query."
        ),
    ) -> list[float]:
        config = self.registry.role(self.profile, role)
        if config.adapter != "reranker":
            raise ModelConfigurationError(
                f"Role {role!r} does not expose a reranker adapter."
            )
        return RerankerAdapter(config).score(
            query,
            documents,
            instruction=instruction,
        )

    def generate_image(
        self,
        role: str,
        *,
        prompt: str,
        output_path: str | Path,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
    ) -> Path:
        config = self.registry.role(self.profile, role)
        if config.adapter != "image_diffusion":
            raise ModelConfigurationError(
                f"Role {role!r} does not expose a local image diffusion adapter."
            )
        adapter = ImageDiffusionAdapter(config)
        with self._gpu_scope(True):
            return adapter.generate_image(
                prompt=prompt,
                output_path=Path(output_path),
                width=width,
                height=height,
                seed=seed,
            )

    def transcribe(self, role: str, audio_path: str | Path) -> str:
        config = self.registry.role(self.profile, role)
        if config.adapter != "speech":
            raise ModelConfigurationError(
                f"Role {role!r} does not expose a local speech adapter."
            )
        adapter = SpeechAdapter(config)
        with self._gpu_scope(config.exclusive_gpu):
            return adapter.transcribe(Path(audio_path))

    @staticmethod
    @contextmanager
    def _gpu_scope(exclusive: bool):
        if exclusive:
            with _GPU_EXCLUSIVE_LOCK:
                yield
        else:
            yield
