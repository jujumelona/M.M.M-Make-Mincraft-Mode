from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .capability_plugins import plugin_manifest
from .model_router import ModelRouter
from .planner import _proposal_from_model_data
from .spec import Proposal, SpecValidationError


class GameDesignPlanner:
    """Multimodal planner that emits the full design plus a bootstrap Fabric spec.

    ``build_slice`` is only the initial project bootstrap consumed by the deterministic
    Fabric generator. It is not the feature scope: CompleteGameDesignPlanner converts
    every requested system into the paginated complete module graph.
    """

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
    ) -> tuple[dict[str, Any], Proposal]:
        prompt = prompt.strip()
        if not prompt:
            raise SpecValidationError("프롬프트를 입력해 주세요.")
        messages = [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt},
        ]
        text = self.router.generate_text(
            "planner",
            messages,
            media_paths=media_paths,
            response_format="json",
        )
        payload = _extract_json(text)
        if set(payload) != {"game_design", "build_slice"}:
            raise SpecValidationError(
                "Planner output must contain exactly game_design and build_slice."
            )
        design = payload["game_design"]
        build_slice = payload["build_slice"]
        if not isinstance(design, dict) or not isinstance(build_slice, dict):
            raise SpecValidationError(
                "Planner output fields must be JSON objects."
            )
        _validate_design(design)
        proposal = _proposal_from_model_data(prompt, build_slice)
        proposal.validate()
        return design, proposal


def _system_prompt() -> str:
    manifest = json.dumps(
        plugin_manifest(), ensure_ascii=False, sort_keys=True
    )
    return f"""
You are GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.
Return exactly one JSON object and no markdown. Use reference images when provided.
Describe every requested system in game_design. The build_slice is only a deterministic
bootstrap project, never the total feature scope. The complete planner will paginate and
compile all systems after this stage, so do not omit features because the project is large.

Current executable plugin manifest:
{manifest}

Output contract:
{{
  "game_design": {{
    "title": "string",
    "pitch": "string",
    "core_loop": ["ordered actions"],
    "progression": ["milestones"],
    "combat": {{"player_verbs": ["..."], "enemy_roles": ["..."]}},
    "world": {{"regions": [{{"id":"snake_case","purpose":"...","links":["region_id"]}}]}},
    "modules": [{{"plugin_id":"from manifest or custom","status":"implemented|custom","reason":"..."}}],
    "assets": [{{"id":"snake_case","kind":"item|block|entity|gui|environment","brief":"..."}}],
    "acceptance_tests": ["observable test"]
  }},
  "build_slice": {{
    "mod_id": "lowercase_snake_case",
    "mod_name": "English name",
    "package_name": "lowercase.dotted.package",
    "summary": "short summary",
    "contents": [
      {{"content_id":"lowercase_snake_case","kind":"item or block","display_name_en":"English","display_name_ko":"Korean","color":"#RRGGBB","recipe":true}}
    ],
    "deferred_capabilities": []
  }}
}}
Choose only the bootstrap item/block entries genuinely needed before complete-module
compilation. There is no numeric content cap and no requested feature may be hidden.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecValidationError("Planner did not return a JSON object.")


def _validate_design(design: dict[str, Any]) -> None:
    required = {
        "title",
        "pitch",
        "core_loop",
        "progression",
        "combat",
        "world",
        "modules",
        "assets",
        "acceptance_tests",
    }
    if set(design) != required:
        raise SpecValidationError(
            f"game_design keys must be exactly {sorted(required)}."
        )
    for field in (
        "core_loop",
        "progression",
        "modules",
        "assets",
        "acceptance_tests",
    ):
        if not isinstance(design[field], list):
            raise SpecValidationError(
                f"game_design.{field} must be a list."
            )
    if not isinstance(design["combat"], dict) or not isinstance(
        design["world"], dict
    ):
        raise SpecValidationError(
            "game_design combat and world must be objects."
        )
