#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


PATH = Path("minecraft_mod_ai/planner.py")


REPLACEMENT = '''class LocalTransformersPlanner:
    """Compatibility wrapper around the role-based model registry.

    Direct model identifiers and fallback planners are intentionally rejected so
    every local backend uses ``config/model_registry.yaml`` and failures remain
    visible to the caller.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        fallback: Planner | None = None,
        max_new_tokens: int | None = None,
        profile: str = "t4_local",
    ) -> None:
        if model_id is not None:
            raise SpecValidationError(
                "Direct model_id overrides are disabled. Configure the model in "
                "config/model_registry.yaml and select a profile."
            )
        if fallback is not None:
            raise SpecValidationError(
                "Silent or automatic planner fallback is disabled."
            )
        if max_new_tokens is not None:
            raise SpecValidationError(
                "Per-call max_new_tokens overrides are disabled. Configure the role "
                "limit in config/model_registry.yaml."
            )
        self.profile = profile
        self.last_backend = f"role-router:{profile}"

    def plan(self, prompt: str) -> Proposal:
        from .routed_planner import RoutedPlanner

        return RoutedPlanner(profile=self.profile).plan(prompt)


class OpenAICompatiblePlanner:'''


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    migrated, count = re.subn(
        r"class LocalTransformersPlanner:.*?\n\nclass OpenAICompatiblePlanner:",
        REPLACEMENT,
        text,
        count=1,
        flags=re.S,
    )
    if count == 0:
        required = (
            "class LocalTransformersPlanner:",
            "role-router:{profile}",
            "Silent or automatic planner fallback is disabled.",
        )
        if all(marker in text for marker in required):
            return
        raise SystemExit("LocalTransformersPlanner migration target was not found.")
    forbidden = (
        "Qwen/Qwen3.5-9B-Instruct",
        'self.last_backend = "deterministic-fallback"',
        "return self.fallback.plan(prompt)",
    )
    found = [marker for marker in forbidden if marker in migrated]
    if found:
        raise SystemExit(f"Forbidden legacy planner markers remain: {found}")
    PATH.write_text(migrated, encoding="utf-8")


if __name__ == "__main__":
    main()
