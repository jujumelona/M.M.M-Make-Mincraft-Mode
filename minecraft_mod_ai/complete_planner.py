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
    """Create a complete production graph with paginated module planning.

    The old one-response contract remains accepted for compatibility. Large designs use
    module batches and cursors so model context length is not treated as a feature cap.
    """

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
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": implementation_prompt},
            ],
            media_paths=media_paths,
            response_format="json",
        )
        payload = _extract_json(text)
        common = {"world_ir", "assets", "audio", "acceptance_tests"}
        if set(payload) == common | {"modules"}:
            modules = tuple(_module(item) for item in _list(payload, "modules"))
        elif set(payload) == common | {"module_batches"}:
            modules = self._expand_batches(
                prompt=prompt,
                game_design=game_design,
                batches=_list(payload, "module_batches"),
                media_paths=media_paths,
            )
        else:
            raise SpecValidationError(
                "Complete planner output must use modules or module_batches plus world/assets/audio/tests."
            )

        assets = tuple(_asset(item) for item in _list(payload, "assets"))
        audio = tuple(_audio(item) for item in _list(payload, "audio"))
        acceptance_tests = tuple(
            str(value).strip() for value in _list(payload, "acceptance_tests")
        )
        if any(not value for value in acceptance_tests):
            raise SpecValidationError(
                "Complete planner acceptance tests must be non-empty strings."
            )
        world_ir = payload["world_ir"]
        if world_ir is not None and not isinstance(world_ir, dict):
            raise SpecValidationError("Complete planner world_ir must be an object or null.")
        modules = _remove_bootstrap_duplicates(modules, base_proposal)
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

    def _expand_batches(
        self,
        *,
        prompt: str,
        game_design: dict[str, Any],
        batches: list[Any],
        media_paths: Sequence[str | Path],
    ) -> tuple[ProductionModule, ...]:
        result: list[ProductionModule] = []
        batch_ids: set[str] = set()
        for raw in batches:
            if not isinstance(raw, dict) or set(raw) != {
                "batch_id",
                "scope",
                "depends_on_batches",
            }:
                raise SpecValidationError("Every module batch has invalid fields.")
            batch_id = str(raw["batch_id"])
            if not batch_id or batch_id in batch_ids:
                raise SpecValidationError(f"Invalid or duplicate module batch: {batch_id!r}")
            batch_ids.add(batch_id)
            scope = str(raw["scope"]).strip()
            if not scope:
                raise SpecValidationError(f"Module batch scope is empty: {batch_id}")
            dependencies = raw["depends_on_batches"]
            if not isinstance(dependencies, list):
                raise SpecValidationError("depends_on_batches must be a list.")
            result.extend(
                self._expand_one_batch(
                    prompt=prompt,
                    game_design=game_design,
                    batch_id=batch_id,
                    scope=scope,
                    dependencies=[str(value) for value in dependencies],
                    already_generated=result,
                    media_paths=media_paths,
                )
            )
        ids = [module.module_id for module in result]
        if len(ids) != len(set(ids)):
            raise SpecValidationError("Paginated planner returned duplicate module IDs.")
        return tuple(result)

    def _expand_one_batch(
        self,
        *,
        prompt: str,
        game_design: dict[str, Any],
        batch_id: str,
        scope: str,
        dependencies: list[str],
        already_generated: list[ProductionModule],
        media_paths: Sequence[str | Path],
    ) -> list[ProductionModule]:
        cursor = ""
        seen_cursors: set[str] = set()
        generated: list[ProductionModule] = []
        while True:
            request = {
                "original_request": prompt,
                "game_design": game_design,
                "batch": {
                    "batch_id": batch_id,
                    "scope": scope,
                    "depends_on_batches": dependencies,
                },
                "known_module_ids": [
                    module.module_id for module in [*already_generated, *generated]
                ],
                "cursor": cursor,
                "contract": {
                    "modules": [
                        {
                            "module_id": "snake_case",
                            "kind": "supported production kind",
                            "config": {},
                            "depends_on": [],
                            "required_gates": [],
                        }
                    ],
                    "complete": True,
                    "next_cursor": "",
                },
            }
            text = self.router.generate_text(
                "planner",
                [
                    {
                        "role": "system",
                        "content": (
                            "Return exactly one JSON object with modules, complete and next_cursor. "
                            "Cover the entire requested batch. If more output is required, set complete=false "
                            "and return a new opaque cursor. Never repeat a module ID."
                        ),
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                media_paths=media_paths,
                response_format="json",
            )
            page = _extract_json(text)
            if set(page) != {"modules", "complete", "next_cursor"}:
                raise SpecValidationError("Module batch page fields are invalid.")
            raw_modules = page["modules"]
            if not isinstance(raw_modules, list) or not raw_modules:
                raise SpecValidationError("Every incomplete/complete module page must contain modules.")
            generated.extend(_module(item) for item in raw_modules)
            complete = page["complete"]
            next_cursor = page["next_cursor"]
            if type(complete) is not bool or not isinstance(next_cursor, str):
                raise SpecValidationError("Module batch pagination contract is invalid.")
            if complete:
                if next_cursor:
                    raise SpecValidationError("Complete module page may not have next_cursor.")
                break
            if not next_cursor or next_cursor in seen_cursors:
                raise SpecValidationError("Module batch pagination did not advance.")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return generated


_SYSTEM_PROMPT = """
You are the complete production planner for Minecraft Java 1.20.1 Fabric.
Return exactly one JSON object and no markdown. Every requested feature must be
represented as an executable production module. Do not reduce the request to an
item/block slice and do not mark work complete merely because a contract file was
created. Use kind=custom_java for unusual Fabric features that do not match a named
kind. Dependencies must form an acyclic graph.

For a small design, return modules directly. For a large design, return module_batches;
each batch is fetched through a cursor and therefore there is no module-count limit.

One-shot output:
{
  "modules": [{"module_id":"snake_case","kind":"item|block|tool|weapon|armor|food|crop|fluid|machine|recipe|effect|enchantment|entity|boss|npc|quest|class|skill|economy|shop|gui|networking|party|guild|command|structure|biome|dimension|world_event|advancement|loot|audio|integration|custom_java","config":{},"depends_on":[],"required_gates":[]}],
  "world_ir": null,
  "assets": [],
  "audio": [],
  "acceptance_tests": ["observable test"]
}

Large output replaces modules with:
"module_batches": [
  {"batch_id":"content_core","scope":"complete scope description","depends_on_batches":[]}
]

world_ir uses mmm/world-ir-v1 and may describe logical structures larger than one
vanilla template; the compiler partitions them. Asset width/height are positive integer
pixels and are controlled by host resource policy rather than a fixed enum.
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


def _remove_bootstrap_duplicates(
    modules: tuple[ProductionModule, ...], base_proposal
) -> tuple[ProductionModule, ...]:
    base_ids = {content.content_id: content.kind.value for content in base_proposal.spec.contents}
    result: list[ProductionModule] = []
    for module in modules:
        base_kind = base_ids.get(module.module_id)
        if base_kind is None:
            result.append(module)
            continue
        if module.kind == base_kind:
            continue
        raise SpecValidationError(
            f"Complete module {module.module_id} collides with bootstrap {base_kind}."
        )
    if not result:
        # A bootstrap-only request still needs one executable graph node.
        result.append(
            ProductionModule(
                module_id="bootstrap_integration",
                kind="integration",
                config={"uses_base_content": sorted(base_ids)},
                required_gates=("Gradle", "GameTest"),
            )
        )
    return tuple(result)
