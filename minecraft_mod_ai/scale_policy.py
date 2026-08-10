from __future__ import annotations

import os
from dataclasses import dataclass


class ScalePolicyError(ValueError):
    pass


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    value = default if raw is None or not raw.strip() else int(raw)
    if value < minimum:
        raise ScalePolicyError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class ScalePolicy:
    """Operational resource policy, deliberately separate from feature semantics.

    There is no global module/content count cap. Large projects are processed through
    deterministic shards and byte-budgeted retrieval. Limits here protect the host
    process and can be raised explicitly without changing a proposal schema.
    """

    java_shard_size: int = 48
    entity_shard_size: int = 24
    function_shard_size: int = 128
    world_placements_per_tick: int = 4
    model_context_bytes: int = 1_500_000
    mcp_page_bytes: int = 256 * 1024
    max_patch_bytes: int = 32 * 1024 * 1024
    max_single_file_bytes: int = 64 * 1024 * 1024
    max_texture_dimension: int = 8192
    max_audio_seconds: int = 3600
    repair_attempts: int = 8
    nbt_piece_axis: int = 32
    nbt_piece_volume: int = 32768
    gradle_min_heap_mb: int = 2048
    gradle_max_heap_mb: int = 12288
    runtime_min_heap_mb: int = 3072
    runtime_max_heap_mb: int = 16384
    fast_mode: bool = False

    def validate(self) -> None:
        positive = {
            "java_shard_size": self.java_shard_size,
            "entity_shard_size": self.entity_shard_size,
            "function_shard_size": self.function_shard_size,
            "world_placements_per_tick": self.world_placements_per_tick,
            "model_context_bytes": self.model_context_bytes,
            "mcp_page_bytes": self.mcp_page_bytes,
            "max_patch_bytes": self.max_patch_bytes,
            "max_single_file_bytes": self.max_single_file_bytes,
            "max_texture_dimension": self.max_texture_dimension,
            "max_audio_seconds": self.max_audio_seconds,
            "repair_attempts": self.repair_attempts,
            "nbt_piece_axis": self.nbt_piece_axis,
            "nbt_piece_volume": self.nbt_piece_volume,
            "gradle_min_heap_mb": self.gradle_min_heap_mb,
            "gradle_max_heap_mb": self.gradle_max_heap_mb,
            "runtime_min_heap_mb": self.runtime_min_heap_mb,
            "runtime_max_heap_mb": self.runtime_max_heap_mb,
        }
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ScalePolicyError(f"{name} must be a positive integer")
        if self.gradle_min_heap_mb > self.gradle_max_heap_mb:
            raise ScalePolicyError("Gradle minimum heap exceeds maximum heap")
        if self.runtime_min_heap_mb > self.runtime_max_heap_mb:
            raise ScalePolicyError("Runtime minimum heap exceeds maximum heap")
        if self.nbt_piece_axis > 48:
            raise ScalePolicyError(
                "nbt_piece_axis may not exceed the reviewed vanilla structure-template axis of 48"
            )
        if self.nbt_piece_axis**3 > self.nbt_piece_volume:
            raise ScalePolicyError(
                "nbt_piece_axis creates pieces larger than nbt_piece_volume; lower the axis or raise volume"
            )

    @classmethod
    def from_environment(cls) -> "ScalePolicy":
        policy = cls(
            java_shard_size=_env_int("MMM_JAVA_SHARD_SIZE", 48),
            entity_shard_size=_env_int("MMM_ENTITY_SHARD_SIZE", 24),
            function_shard_size=_env_int("MMM_FUNCTION_SHARD_SIZE", 128),
            world_placements_per_tick=_env_int(
                "MMM_WORLD_PLACEMENTS_PER_TICK",
                4,
            ),
            model_context_bytes=_env_int("MMM_MODEL_CONTEXT_BYTES", 1_500_000),
            mcp_page_bytes=_env_int("MMM_MCP_PAGE_BYTES", 256 * 1024),
            max_patch_bytes=_env_int("MMM_MAX_PATCH_BYTES", 32 * 1024 * 1024),
            max_single_file_bytes=_env_int("MMM_MAX_SINGLE_FILE_BYTES", 64 * 1024 * 1024),
            max_texture_dimension=_env_int("MMM_MAX_TEXTURE_DIMENSION", 8192),
            max_audio_seconds=_env_int("MMM_MAX_AUDIO_SECONDS", 3600),
            repair_attempts=_env_int("MMM_REPAIR_ATTEMPTS", 8),
            nbt_piece_axis=_env_int("MMM_NBT_PIECE_AXIS", 32),
            nbt_piece_volume=_env_int("MMM_NBT_PIECE_VOLUME", 32768),
            gradle_min_heap_mb=_env_int("MMM_GRADLE_MIN_HEAP_MB", 2048),
            gradle_max_heap_mb=_env_int("MMM_GRADLE_MAX_HEAP_MB", 12288),
            runtime_min_heap_mb=_env_int("MMM_RUNTIME_MIN_HEAP_MB", 3072),
            runtime_max_heap_mb=_env_int("MMM_RUNTIME_MAX_HEAP_MB", 16384),
        )
        policy.validate()
        return policy

    def gradle_heap_mb(self, *, module_count: int, source_file_count: int = 0) -> int:
        # Resource sizing, not a feature cap. The project remains complete if the host
        # cannot provide this heap; the build gate reports the resource requirement.
        estimate = self.gradle_min_heap_mb + module_count * 6 + source_file_count * 2
        return max(self.gradle_min_heap_mb, min(self.gradle_max_heap_mb, estimate))

    def runtime_heap_mb(self, *, module_count: int, entity_count: int, structure_count: int) -> int:
        estimate = (
            self.runtime_min_heap_mb
            + module_count * 3
            + entity_count * 24
            + structure_count * 16
        )
        return max(self.runtime_min_heap_mb, min(self.runtime_max_heap_mb, estimate))
