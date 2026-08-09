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
    VLLMAdapter,
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
        self._generation_lock = threading.RLock()
        self._active_generation_role: str | None = None
        self._active_generation_adapter: Any | None = None

    @contextmanager
    def generation_session(self, role: str):
        """Keep one text-generation backend alive for a bounded workflow.

        Only one role can be pinned on a router at a time.  This avoids an
        unbounded multi-model VRAM cache while allowing a paginated planner to
        reuse the same processor and weights until its complete plan succeeds or
        raises.  Direct ``generate_text`` calls outside this context retain their
        existing load-generate-release lifetime.
        """

        config = self.registry.role(self.profile, role)
        adapter = self._new_text_adapter(config, role=role)
        with self._generation_lock:
            if self._active_generation_adapter is not None:
                raise ModelConfigurationError(
                    "A generation session is already active for role "
                    f"{self._active_generation_role!r}."
                )
            self._active_generation_role = role
            self._active_generation_adapter = adapter
            session_factory = getattr(adapter, "generation_session", None)
            try:
                if callable(session_factory):
                    with session_factory():
                        yield self
                else:
                    try:
                        yield self
                    finally:
                        adapter.close()
            finally:
                self._active_generation_adapter = None
                self._active_generation_role = None

    def generate_text(
        self,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        media_paths: Sequence[str | Path] = (),
        response_format: str = "text",
    ) -> str:
        with self._generation_lock:
            config = self.registry.role(self.profile, role)
            if self._active_generation_adapter is not None:
                if role != self._active_generation_role:
                    raise ModelConfigurationError(
                        "Generation session for role "
                        f"{self._active_generation_role!r} cannot serve role "
                        f"{role!r}."
                    )
                adapter = self._active_generation_adapter
            else:
                adapter = self._new_text_adapter(config, role=role)
            request = GenerationRequest(
                messages=messages,
                media_paths=tuple(Path(path) for path in media_paths),
                response_format=response_format,
            )
            with self._gpu_scope(config.exclusive_gpu):
                return adapter.generate(request)

    @staticmethod
    def _new_text_adapter(config, *, role: str):
        if config.adapter == "vllm":
            return VLLMAdapter(config)
        if config.adapter == "transformers_text":
            return TransformersTextAdapter(config)
        if config.adapter == "transformers_multimodal":
            return TransformersMultimodalAdapter(config)
        if config.adapter == "openai_compatible":
            return OpenAICompatibleAdapter(config)
        raise ModelConfigurationError(
            f"Role {role!r} cannot generate text with adapter {config.adapter!r}."
        )

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
        if config.adapter == "image_diffusion":
            adapter = ImageDiffusionAdapter(config)
        elif config.adapter == "openai_compatible":
            adapter = OpenAICompatibleAdapter(config)
        else:
            raise ModelConfigurationError(
                f"Role {role!r} cannot generate images with adapter "
                f"{config.adapter!r}."
            )
        with self._gpu_scope(config.exclusive_gpu):
            return adapter.generate_image(
                prompt=prompt,
                output_path=Path(output_path),
                width=width,
                height=height,
                seed=seed,
            )

    def transcribe(self, role: str, audio_path: str | Path) -> str:
        config = self.registry.role(self.profile, role)
        if config.adapter == "speech":
            adapter = SpeechAdapter(config)
        elif config.adapter == "openai_compatible":
            adapter = OpenAICompatibleAdapter(config)
        else:
            raise ModelConfigurationError(
                f"Role {role!r} cannot transcribe audio with adapter "
                f"{config.adapter!r}."
            )
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
