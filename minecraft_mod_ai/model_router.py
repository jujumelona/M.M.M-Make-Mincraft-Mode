from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
_ROLE_TOOL_STAGE = {
    "planner": "planning",
    "researcher": "research",
    "coder": "generation",
    "coder_safe": "quality",
    "visual_critic": "quality",
}
_NATIVE_TOOL_ADAPTERS = frozenset({"llama_cpp", "vllm", "openai_compatible"})
_DEFAULT_MAX_TOOL_ROUNDS = 8
_DEFAULT_MAX_TOOL_CALLS = 24


class ModelRouter:
    """Role router with strict profile selection and no silent backend fallback."""

    def __init__(
        self,
        *,
        profile: str = "t4_local",
        registry: ModelRegistry | None = None,
        agent_tool_runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.profile = profile
        self.registry.load_profile(profile)
        self._generation_lock = threading.RLock()
        self._active_generation_role: str | None = None
        self._active_generation_adapter: Any | None = None
        self._agent_tool_runtime_factory = agent_tool_runtime_factory
        self._agent_tool_runtime: Any | None = None

    @contextmanager
    def generation_session(self, role: str):
        """Keep one text-generation backend alive for a bounded workflow.

        Only one role can be pinned on a router at a time. This avoids an
        unbounded multi-model VRAM cache while allowing a paginated planner to
        reuse the same processor and weights until its complete plan succeeds or
        raises. Direct ``generate_text`` calls outside this context retain their
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
        tool_stage: str | None = None,
        enable_tools: bool = True,
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

            stage = (tool_stage or _ROLE_TOOL_STAGE.get(role, "")).strip().lower()
            runtime = None
            tools: tuple[Mapping[str, Any], ...] = ()
            if self._tools_enabled(
                enable_tools=enable_tools,
                stage=stage,
                adapter_name=config.adapter,
            ):
                runtime = self._tool_runtime()
                tools = tuple(runtime.tool_schemas(stage))

            request = GenerationRequest(
                messages=messages,
                media_paths=tuple(Path(path) for path in media_paths),
                response_format=response_format,
                tools=tools,
                tool_choice="auto" if tools else None,
                parallel_tool_calls=True,
            )
            with self._gpu_scope(config.exclusive_gpu):
                if runtime is not None and tools:
                    return self._generate_with_tools(
                        adapter=adapter,
                        request=request,
                        runtime=runtime,
                        stage=stage,
                    )
                return adapter.generate(request)

    def _generate_with_tools(
        self,
        *,
        adapter: Any,
        request: GenerationRequest,
        runtime: Any,
        stage: str,
    ) -> str:
        messages: list[dict[str, Any]] = [dict(message) for message in request.messages]
        total_calls = 0
        max_rounds = _positive_env_int(
            "MMM_AGENT_MAX_TOOL_ROUNDS",
            _DEFAULT_MAX_TOOL_ROUNDS,
        )
        max_calls = _positive_env_int(
            "MMM_AGENT_MAX_TOOL_CALLS",
            _DEFAULT_MAX_TOOL_CALLS,
        )

        for round_index in range(max_rounds + 1):
            turn_request = GenerationRequest(
                messages=messages,
                media_paths=request.media_paths if round_index == 0 else (),
                response_format=request.response_format,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
            turn = adapter.generate_turn(turn_request)
            if not turn.tool_calls:
                if not turn.content.strip():
                    raise ModelConfigurationError(
                        "Tool-capable model returned an empty final response."
                    )
                return turn.content.strip()

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.raw_arguments
                            or json.dumps(
                                dict(call.arguments),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for call in turn.tool_calls
                ],
            }
            messages.append(assistant_message)

            for call in turn.tool_calls:
                total_calls += 1
                if total_calls > max_calls:
                    raise ModelConfigurationError(
                        f"Agent exceeded the tool-call limit ({max_calls})."
                    )
                try:
                    result = runtime.call(stage, call.name, call.arguments)
                    tool_payload: Mapping[str, Any] = {
                        "ok": True,
                        "tool": call.name,
                        "result": result,
                    }
                except Exception as exc:
                    # Tool failures are observations, not a reason to kill the agent.
                    # Feeding the exact error back lets Qwen repair arguments or select
                    # another stage-allowed tool on the next turn.
                    tool_payload = {
                        "ok": False,
                        "tool": call.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": json.dumps(
                            tool_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )

        raise ModelConfigurationError(
            f"Agent did not finish after {max_rounds} tool rounds."
        )

    def _tool_runtime(self) -> Any:
        if self._agent_tool_runtime is not None:
            return self._agent_tool_runtime
        if self._agent_tool_runtime_factory is not None:
            self._agent_tool_runtime = self._agent_tool_runtime_factory(
                profile=self.profile
            )
        else:
            from .agent_tool_runtime import AgentToolRuntime

            self._agent_tool_runtime = AgentToolRuntime(profile=self.profile)
        return self._agent_tool_runtime

    @staticmethod
    def _tools_enabled(
        *,
        enable_tools: bool,
        stage: str,
        adapter_name: str,
    ) -> bool:
        if not enable_tools or not stage:
            return False
        if adapter_name not in _NATIVE_TOOL_ADAPTERS:
            return False
        if os.environ.get("MMM_AGENT_TOOL_CHILD", "").strip() == "1":
            return False
        raw = os.environ.get("MMM_AGENT_TOOLS", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    @staticmethod
    def _new_text_adapter(config, *, role: str):
        if config.adapter == "transformers_text":
            return TransformersTextAdapter(config)
        if config.adapter == "transformers_multimodal":
            return TransformersMultimodalAdapter(config)
        if config.adapter in ("llama_cpp", "vllm"):
            from .model_adapters.llama_cpp_adapter import LlamaCppAdapter
            return LlamaCppAdapter(config)
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
            "Retrieve the Minecraft modding evidence that directly answers the "
            "query for the caller-selected platform target. Do not prefer or infer "
            "a different Minecraft version or mapping namespace."
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


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
