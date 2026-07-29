from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .complete_spec import (
    AssetRequest,
    AudioRequest,
    CompleteProposal,
    ProductionModule,
    complete_proposal_from_parts,
)
from .game_design import GameDesignPlanner
from .model_router import ModelRouter
from .spec import SpecValidationError


class CompleteGameDesignPlanner:
    """Create a complete production proposal rather than an item/block-only slice."""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def plan(
        self,
        prompt: str,
        *,
        media_paths: Sequence[str | Path] = (),
        existing_input_sha256: str = "",
    ) -> CompleteProposal:
        game_design, base_proposal = GameDesignPlanner(self.router).plan(
            prompt,
            media_paths=media_paths,
        )
        implementation_prompt = _implementation_prompt(prompt, game_design)
        text = self.router.generate_text(
            "planner",
            [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT,
                },
                {"role": "user", "content": implementation_prompt},
            ],
            media_paths=media_paths,
            response_format="json",
        )
        payload = _extract_json(text)
        required = {"modules", "world_ir", "assets", "audio", "acceptance_tests"}
        if set(payload) != required:
            raise SpecValidationError(
                f"Complete planner output keys must be exactly {sorted(required)}."
            )
        modules = tuple(_module(item) for item in _list(payload, "modules"))
        assets = tuple(_asset(item) for item in _list(payload, "assets"))
        audio = tuple(_audio(item) for item in _list(payload, "audio"))
        acceptance_tests = tuple(str(value).strip() for value in _list(payload, "acceptance_tests"))
        if any(not value for value in acceptance_tests):
            raise SpecValidationError("Complete planner acceptance tests must be non-empty strings.")
        world_ir = payload["world_ir"]
        if world_ir is not None and not isinstance(world_ir, dict):
            raise SpecValidationError("Complete planner world_ir must be an object or null.")
        return complete_proposal_from_parts(
            requested_prompt=prompt,
            base_proposal=base_proposal,
            game_design=game_design,
            modules=modules,
            world_ir=world_ir,
            assets=assets,
            audio=audio,
            acceptance_tests=acceptance_tests,
            existing_input_sha256=existing_input_sha256,
        )


_SYSTEM_PROMPT = """
You are the complete production planner for Minecraft Java 1.20.1 Fabric.
Return exactly one JSON object and no markdown. Every requested feature must be
represented as an executable production module. Do not reduce the request to an
item/block slice and do not mark work complete merely because a contract file was
created. Use kind=custom_java for unusual Fabric features that do not match a named
kind. Dependencies must form an acyclic graph.

Output contract:
{
  "modules": [
    {
      "module_id": "snake_case",
      "kind": "item|block|tool|weapon|armor|food|crop|fluid|machine|recipe|effect|enchantment|entity|boss|npc|quest|class|skill|economy|shop|gui|networking|party|guild|command|structure|biome|dimension|world_event|advancement|loot|audio|integration|custom_java",
      "config": {"implementation details": "JSON values"},
      "depends_on": ["module_id"],
      "required_gates": ["observable build/runtime gate"]
    }
  ],
  "world_ir": null or {
    "schema_version": "mmm/world-ir-v1",
    "regions": [{"id":"snake_case","purpose":"..."}],
    "routes": [{"from":"region","to":"region","travel_mode":"road"}],
    "structures": [{"id":"snake_case","region_id":"region","kind":"village|dungeon|tower|road|arena","brief":"...","size":[9,6,9],"palette":["minecraft:stone_bricks","minecraft:air"]}],
    "quests": [{"id":"snake_case","start_region":"region","end_region":"region","objective":"..."}],
    "constraints": []
  },
  "assets": [
    {"asset_id":"snake_case","kind":"item|block|entity|gui|environment|icon","prompt":"...","target_path":"src/main/resources/...png","width":16,"height":16}
  ],
  "audio": [
    {"sound_id":"snake_case","kind":"effect|ambient|music|ui","duration_seconds":1.0,"frequency_hz":440.0,"volume":0.8,"loop":false,"subtitle_en":"...","subtitle_ko":"..."}
  ],
  "acceptance_tests": ["observable test covering every requested system"]
}
""".strip()


def _implementation_prompt(prompt: str, game_design: dict[str, Any]) -> str:
    return (
        "Original request:\n"
        + prompt.strip()
        + "\n\nApproved design candidate:\n"
        + json.dumps(game_design, ensure_ascii=False, sort_keys=True)
        + "\n\nCreate the complete implementation graph. Include server authority, persistence, "
        "networking, client UI, resources, runtime playtests and release gates whenever relevant."
    )


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
    raise SpecValidationError("Complete planner did not return a JSON object.")


def _list(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload[field]
    if not isinstance(value, list):
        raise SpecValidationError(f"Complete planner field {field} must be a list.")
    return value


def _module(value: Any) -> ProductionModule:
    if not isinstance(value, dict):
        raise SpecValidationError("Every production module must be an object.")
    allowed = {"module_id", "kind", "config", "depends_on", "required_gates"}
    if set(value) != allowed:
        raise SpecValidationError("Production module fields are invalid.")
    return ProductionModule(
        module_id=str(value["module_id"]),
        kind=str(value["kind"]),
        config=dict(value["config"]),
        depends_on=tuple(str(item) for item in value["depends_on"]),
        required_gates=tuple(str(item) for item in value["required_gates"]),
    )


def _asset(value: Any) -> AssetRequest:
    if not isinstance(value, dict):
        raise SpecValidationError("Every asset request must be an object.")
    return AssetRequest(
        asset_id=str(value["asset_id"]),
        kind=str(value["kind"]),
        prompt=str(value["prompt"]),
        target_path=str(value["target_path"]),
        width=int(value.get("width", 16)),
        height=int(value.get("height", 16)),
    )


def _audio(value: Any) -> AudioRequest:
    if not isinstance(value, dict):
        raise SpecValidationError("Every audio request must be an object.")
    return AudioRequest(
        sound_id=str(value["sound_id"]),
        kind=str(value["kind"]),
        duration_seconds=float(value["duration_seconds"]),
        frequency_hz=float(value.get("frequency_hz", 440.0)),
        volume=float(value.get("volume", 0.8)),
        loop=bool(value.get("loop", False)),
        subtitle_en=str(value.get("subtitle_en", "")),
        subtitle_ko=str(value.get("subtitle_ko", "")),
    )
