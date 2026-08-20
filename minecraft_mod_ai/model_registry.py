from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config_paths import config_path
from .model_adapters import AdapterConfig, ModelConfigurationError


LEGACY_REQUIRED_ROLES = frozenset(
    {"planner", "researcher", "coder", "visual_critic", "image_generator"}
)
REQUIRED_ROLES = frozenset(
    {
        "planner",
        "researcher",
        "coder",
        "coder_safe",
        "visual_critic",
        "embedding",
        "reranker",
        "image_generator",
    }
)
ALLOWED_ADAPTERS = frozenset(
    {
        "mock",
        "transformers_text",
        "transformers_multimodal",
        "llama_cpp",
        "vllm",
        "openai_compatible",
        "image_diffusion",
        "embedding",
        "reranker",
    }
)
SUPPORTED_SCHEMAS = frozenset({"mmm/model-registry-v1", "mmm/model-registry-v2"})
_REGISTRY_CACHE_LOCK = threading.RLock()
_REGISTRY_SOURCE_CACHE: dict[tuple[str, int, int], tuple[str, dict[str, Any]]] = {}


@dataclass(frozen=True)
class ModelProfile:
    name: str
    description: str
    roles: Mapping[str, AdapterConfig]


def _read_registry_source(path: Path) -> tuple[str, dict[str, Any]]:
    """Parse an unchanged registry file once per process.

    Profile role resolution remains uncached because remote profiles intentionally read
    current environment variables on every load_profile() call.
    """
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    with _REGISTRY_CACHE_LOCK:
        cached = _REGISTRY_SOURCE_CACHE.get(key)
        if cached is not None:
            return cached

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") not in SUPPORTED_SCHEMAS:
        raise ModelConfigurationError("Unsupported or malformed model registry.")
    profiles = raw.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ModelConfigurationError("Model registry contains no profiles.")
    value = (str(raw["schema_version"]), profiles)
    with _REGISTRY_CACHE_LOCK:
        _REGISTRY_SOURCE_CACHE[key] = value
        while len(_REGISTRY_SOURCE_CACHE) > 8:
            _REGISTRY_SOURCE_CACHE.pop(next(iter(_REGISTRY_SOURCE_CACHE)))
    return value


class ModelRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path).expanduser().resolve()
            if path is not None
            else config_path("model_registry.yaml")
        )
        if not self.path.is_file():
            raise ModelConfigurationError(f"Model registry not found: {self.path}")
        self.schema_version, self._raw_profiles = _read_registry_source(self.path)

    def profile_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._raw_profiles))

    def load_profile(self, name: str) -> ModelProfile:
        raw_profile = self._raw_profiles.get(name)
        if not isinstance(raw_profile, dict):
            raise ModelConfigurationError(
                f"Unknown model profile {name!r}; available: {', '.join(self.profile_names())}"
            )
        raw_roles = raw_profile.get("roles")
        if not isinstance(raw_roles, dict):
            raise ModelConfigurationError(f"Profile {name!r} has no roles mapping.")
        required_roles = (
            LEGACY_REQUIRED_ROLES
            if self.schema_version == "mmm/model-registry-v1"
            else REQUIRED_ROLES
        )
        missing = required_roles - set(raw_roles)
        if missing:
            raise ModelConfigurationError(
                f"Profile {name!r} is missing roles: {sorted(missing)}"
            )
        roles = {
            role: self._resolve_role(role, config)
            for role, config in raw_roles.items()
        }
        profile = ModelProfile(
            name=name,
            description=str(raw_profile.get("description", "")),
            roles=roles,
        )
        from .parallel_runtime_contract import prefetch_profile

        prefetch_profile(profile)
        return profile

    def role(self, profile: str, role: str) -> AdapterConfig:
        loaded = self.load_profile(profile)
        try:
            return loaded.roles[role]
        except KeyError as exc:
            raise ModelConfigurationError(
                f"Profile {profile!r} has no role {role!r}."
            ) from exc

    @staticmethod
    def _resolve_role(role: str, raw: Any) -> AdapterConfig:
        if not isinstance(raw, dict):
            raise ModelConfigurationError(f"Role {role!r} must be a mapping.")
        adapter = str(raw.get("adapter", "")).strip()
        if adapter not in ALLOWED_ADAPTERS:
            raise ModelConfigurationError(
                f"Role {role!r} uses unsupported adapter {adapter!r}."
            )
        provider = str(raw.get("provider", "local")).strip() or "local"
        model_id = str(raw.get("model_id", "")).strip()
        base_url = ""
        api_key = ""
        if provider == "openai_compatible":
            model_id = _required_env(raw, "model_env", role)
            base_url = _required_env(raw, "base_url_env", role)
            api_key = _required_env(raw, "api_key_env", role)
        elif not model_id:
            raise ModelConfigurationError(f"Local role {role!r} has no model_id.")
        known = {
            "model_id",
            "provider",
            "adapter",
            "quantization",
            "torch_dtype",
            "max_context",
            "max_input_tokens",
            "max_new_tokens",
            "min_free_vram_mb",
            "exclusive_gpu",
            "cpu_offload",
            "model_env",
            "base_url_env",
            "api_key_env",
        }
        return AdapterConfig(
            role=role,
            adapter=adapter,
            model_id=model_id,
            provider=provider,
            quantization=(str(raw["quantization"]) if raw.get("quantization") else None),
            torch_dtype=str(raw.get("torch_dtype", "auto")),
            max_context=_nonnegative_int(
                raw.get("max_context", 0), f"{role}.max_context"
            ),
            max_input_tokens=_nonnegative_int(
                raw.get("max_input_tokens", 0), f"{role}.max_input_tokens"
            ),
            max_new_tokens=_positive_int(
                raw.get("max_new_tokens", 1200), f"{role}.max_new_tokens"
            ),
            min_free_vram_mb=_nonnegative_int(
                raw.get("min_free_vram_mb", 0), f"{role}.min_free_vram_mb"
            ),
            exclusive_gpu=bool(raw.get("exclusive_gpu", False)),
            cpu_offload=bool(raw.get("cpu_offload", False)),
            base_url=base_url,
            api_key=api_key,
            extra={key: value for key, value in raw.items() if key not in known},
        )

    def to_public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": "mmm/model-registry-public-v2",
            "source_schema_version": self.schema_version,
            "profiles": {},
        }
        for name in self.profile_names():
            raw = self._raw_profiles[name]
            roles: dict[str, Any] = {}
            for role, config in raw["roles"].items():
                roles[role] = {
                    key: value
                    for key, value in config.items()
                    if key not in {"api_key_env"}
                }
            result["profiles"][name] = {
                "description": raw.get("description", ""),
                "roles": roles,
            }
        return result


def _required_env(raw: Mapping[str, Any], key: str, role: str) -> str:
    env_name = str(raw.get(key, "")).strip()
    if not env_name:
        raise ModelConfigurationError(f"Remote role {role!r} has no {key}.")
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ModelConfigurationError(
            f"Environment variable {env_name} is required for role {role!r}."
        )
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ModelConfigurationError(f"{field} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ModelConfigurationError(f"{field} must be a non-negative integer.")
    return value
