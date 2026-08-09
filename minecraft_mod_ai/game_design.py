from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_plugins import plugin_manifest
from .central_research import normalize_research_brief
from .model_router import ModelRouter
from .planner import HeuristicPlanner, _proposal_from_model_data
from .spec import Proposal, SpecValidationError


_GAME_DESIGN_FIELDS = (
    "title",
    "pitch",
    "core_loop",
    "progression",
    "combat",
    "mod_context",
    "modules",
    "assets",
    "acceptance_tests",
)
_OPTIONAL_GAME_DESIGN_FIELDS = ("art_direction",)


class GameDesignPlanner:
    """Emit a compact reader-facing design, then derive the bootstrap locally.

    The model is deliberately responsible for one bounded artifact: ``game_design``.
    It never has to repeat that design inside a second, project-sized envelope merely
    to create a Fabric identity.  ``build_slice`` is derived deterministically from
    the unchanged original request and the validated design.  CompleteGameDesignPlanner
    then turns the same request into its paginated production outline and module graph.
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
        try:
            design = _extract_valid_game_design(text)
        except SpecValidationError as initial_error:
            repaired_text = self.router.generate_text(
                "planner",
                _repair_messages(prompt),
                media_paths=media_paths,
                response_format="json",
            )
            try:
                design = _extract_valid_game_design(repaired_text)
            except SpecValidationError as repair_error:
                raise SpecValidationError(
                    "Planner could not return a complete game_design after one "
                    "automatic repair of that stage. "
                    f"Initial response: {initial_error} "
                    f"Repair response: {repair_error}"
                ) from repair_error

        design = _canonical_game_design(design)
        # Research classification is derived locally from the authoritative request
        # and validated design.  Asking the model to duplicate it here made this
        # supposedly small stage grow with the size of the whole project.
        research_brief = normalize_research_brief(prompt, design)
        design = {
            **design,
            "_research_brief": research_brief,
        }
        build_slice = _deterministic_bootstrap(prompt, design)
        proposal = _proposal_from_model_data(prompt, build_slice)
        proposal.validate()
        return design, proposal


def _system_prompt() -> str:
    manifest = json.dumps(
        _planner_plugin_manifest(), ensure_ascii=False, sort_keys=True
    )
    return f"""
You are GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.
Return exactly one small JSON object with one top-level game_design field and no
markdown or analysis. Use reference images when provided. This is a readable design
overview, not the complete implementation catalog. State the player fantasy, core
loop, progression, requested systems, vanilla integration boundaries, art direction
when requested, and observable quality goals. Preserve every distinct requested
system, but group large repeated catalogs into named families instead of enumerating
every member. The unchanged original request is passed to a later paginated production
planner, so project scale must not make this response grow without bound.

Do not create build_slice, research_brief, production modules, code, or implementation
pages in this response. Those are separate deterministic or paginated stages.
Do not insert combat, bosses, maps, villages, dungeons, voice, AI, or any other content
merely to fill a category. Empty combat lists are correct for a non-combat request.
Treat a named commercial game as a request to understand mechanics and experience,
not as permission to copy its proprietary code, characters, logos, textures, models,
audio, writing, maps, or other protected material. Plan original assets unless the
user supplies authorized material or a third-party artifact passes the origin license.

Current planner plugin catalog (the executable registry retains full details):
{manifest}

Output contract:
{{
  "game_design": {{
    "title": "string",
    "pitch": "string",
    "core_loop": ["ordered actions"],
    "progression": ["milestones"],
    "combat": {{"player_verbs": ["..."], "enemy_roles": ["..."]}},
    "mod_context": {{"vanilla_integration": ["..."], "compatibility_targets": ["..."]}},
    "art_direction": {{"visual_tone": "...", "texture_guidance": ["..."], "model_animation_guidance": ["..."]}},
    "modules": [{{"plugin_id":"from manifest or custom","status":"implemented|custom","reason":"..."}}],
    "assets": [{{"id":"snake_case","kind":"item|block|entity|gui|environment","brief":"..."}}],
    "acceptance_tests": ["observable test"]
  }}
}}
If combat or mod integration details are not requested, keep those lists empty; their
presence in this JSON shape is not permission to invent them. modules describes broad
requested systems only, not every implementation unit. assets describes grouped asset
families needed to communicate the design, not a project-wide manifest.
combat and mod_context must be JSON objects whose values, when present, are arrays of
non-empty strings. Use an empty object when the request has no relevant details; never
replace an array with a scalar string or a nested object.
art_direction is optional: include it only for requested visual direction, and omit it otherwise.
""".strip()


def _planner_plugin_manifest() -> dict[str, Any]:
    """Keep only planner-selection fields from the full executable manifest."""

    manifest = plugin_manifest()
    return {
        "product_scope": manifest["product_scope"],
        "standalone_map_generation": manifest["standalone_map_generation"],
        "plugins": [
            {
                "plugin_id": plugin["plugin_id"],
                "status": plugin["status"],
            }
            for plugin in manifest["plugins"]
        ],
    }


def _extract_valid_game_design(text: str) -> dict[str, Any]:
    """Extract the final complete design spine from a model response.

    The current contract is ``{"game_design": {...}}``.  A bare design object and
    the former combined envelope remain readable so saved/captured responses do not
    break, but any model-authored build slice or research brief is intentionally
    ignored.  The last complete candidate wins because reasoning-capable models can
    emit a smaller draft before their final JSON object.
    """

    design = _last_complete_standalone_design(text)
    if design is not None:
        return design

    candidates = tuple(_json_objects(text))
    for candidate in reversed(candidates):
        nested = candidate.get("game_design")
        possible = nested if isinstance(nested, dict) else candidate
        if not isinstance(possible, dict):
            continue
        if set(possible) & set(_GAME_DESIGN_FIELDS):
            _validate_design(possible)
    if candidates:
        raise SpecValidationError(
            "Planner response is incomplete. Include one complete game_design "
            "object, then retry that stage."
        )
    raise SpecValidationError("Planner did not return a JSON object for game_design.")


def _repair_messages(prompt: str) -> list[dict[str, str]]:
    """Retry from the authoritative request with a compact response contract.

    The malformed model response is deliberately excluded.  It is untrusted,
    non-authoritative context and may itself be as large as the model's output
    allowance.  Re-attaching it made the repair request larger than the initial
    request and could overflow an otherwise valid context window.
    """

    return [
        {"role": "system", "content": _repair_system_prompt()},
        {"role": "user", "content": prompt},
    ]


def _repair_system_prompt() -> str:
    """Compact repair contract that fits small local-model context windows."""

    return """
Retry only the compact game-design stage from the unchanged original request. Return
exactly one JSON object and no analysis or markdown. Preserve every distinct requested
system, grouping large repeated catalogs into families. Do not add unrequested systems.
Do not return build_slice, research_brief, production modules, code, or later pages.

Required shape:
{
  "game_design": {
    "title": "string", "pitch": "string", "core_loop": [], "progression": [],
    "combat": {}, "mod_context": {}, "modules": [], "assets": [],
    "acceptance_tests": []
  }
}
combat and mod_context must be JSON objects whose values, when present, are arrays of
non-empty strings. Use an empty object when the request has no relevant details; never
replace an array with a scalar string or a nested object.
""".strip()


def _last_complete_standalone_design(text: str) -> dict[str, Any] | None:
    designs: list[dict[str, Any]] = []
    for candidate in _json_objects(text):
        nested = candidate.get("game_design")
        possible = nested if isinstance(nested, dict) else candidate
        if not set(_GAME_DESIGN_FIELDS) <= set(possible):
            continue
        try:
            _validate_design(possible)
        except SpecValidationError:
            continue
        designs.append(possible)
    return designs[-1] if designs else None


def _deterministic_bootstrap(
    prompt: str,
    design: dict[str, Any],
) -> dict[str, Any]:
    """Translate the request-derived heuristic proposal into the bootstrap schema."""

    proposal = HeuristicPlanner().plan(prompt)
    spec = proposal.spec
    title = str(design.get("title", "")).strip() or spec.mod_name
    pitch = str(design.get("pitch", "")).strip() or spec.summary
    normalized_title = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in title.lower()
    )
    title_stem = "_".join(
        part for part in normalized_title.split("_") if part
    )
    if not title_stem:
        title_stem = f"mmm_{hashlib.sha256(title.encode('utf-8')).hexdigest()[:10]}"
    if not title_stem[0].isalpha():
        title_stem = f"mmm_{title_stem}"
    mod_id = f"{title_stem[:55].rstrip('_')}_mod"
    return {
        "mod_id": mod_id,
        "mod_name": title,
        "package_name": f"ai.minecraft.generated.{mod_id}",
        "summary": pitch,
        "contents": [
            {
                "content_id": content.content_id,
                "kind": content.kind.value,
                "display_name_en": content.display_name_en,
                "display_name_ko": content.display_name_ko,
                "color": content.color,
                "recipe": content.recipe,
            }
            for content in spec.contents
        ],
        "deferred_capabilities": [
            deferred.capability for deferred in proposal.deferred_requests
        ],
    }


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _canonical_game_design(design: dict[str, Any]) -> dict[str, Any]:
    """Keep the reader-facing contract free of model-side trace metadata."""

    canonical = {field: design[field] for field in _GAME_DESIGN_FIELDS}
    for field in _OPTIONAL_GAME_DESIGN_FIELDS:
        if field in design:
            canonical[field] = design[field]
    return canonical


def _validate_design(design: dict[str, Any]) -> None:
    missing = sorted(set(_GAME_DESIGN_FIELDS) - set(design))
    if missing:
        raise SpecValidationError(
            "Planner response is incomplete: game_design is missing "
            + ", ".join(missing)
            + ". Please retry the plan."
        )
    for field in ("title", "pitch"):
        value = design[field]
        if not isinstance(value, str) or not value.strip():
            raise SpecValidationError(
                f"game_design.{field} must be a non-empty string."
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
    for field in ("core_loop", "progression", "acceptance_tests"):
        if any(not isinstance(value, str) or not value.strip() for value in design[field]):
            raise SpecValidationError(
                f"game_design.{field} must contain only non-empty strings."
            )
    for field, required in (
        ("modules", frozenset({"plugin_id", "status", "reason"})),
        ("assets", frozenset({"id", "kind", "brief"})),
    ):
        for value in design[field]:
            if not isinstance(value, dict) or not required <= set(value):
                raise SpecValidationError(
                    f"game_design.{field} entries must contain "
                    + ", ".join(sorted(required))
                    + "."
                )
            if any(
                not isinstance(value[key], str) or not value[key].strip()
                for key in required
            ):
                raise SpecValidationError(
                    f"game_design.{field} required values must be non-empty strings."
                )
    if not isinstance(design["combat"], dict) or not isinstance(
        design["mod_context"], dict
    ):
        raise SpecValidationError(
            "game_design combat and mod_context must be objects."
        )
    for field in ("combat", "mod_context"):
        if any(
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            for values in design[field].values()
        ):
            raise SpecValidationError(
                f"game_design.{field} values must be lists of non-empty strings."
            )
    if "art_direction" in design and not isinstance(design["art_direction"], dict):
        raise SpecValidationError("game_design.art_direction must be an object.")
