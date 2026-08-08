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


_PLAN_FIELDS = frozenset({"game_design", "build_slice"})
_OPTIONAL_PLAN_FIELDS = frozenset({"research_brief"})
_RESPONSE_WRAPPER_KEYS = (
    "response",
    "result",
    "data",
    "plan",
    "output",
    "payload",
    "message",
    "content",
)
_RESPONSE_LIST_WRAPPER_KEYS = ("choices", "results", "outputs")
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
_BUILD_SLICE_FIELDS = (
    "mod_id",
    "mod_name",
    "package_name",
    "summary",
    "contents",
    "deferred_capabilities",
)
_RESEARCH_BRIEF_FIELDS = (
    "summary",
    "domains",
    "unresolved_questions",
)
_RESEARCH_DOMAIN_FIELDS = (
    "domain_id",
    "objective",
    "requirements",
    "evidence_kinds",
    "queries",
    "providers",
    "depends_on",
)


class GameDesignPlanner:
    """Emit a reader-facing design overview plus a bootstrap Fabric spec.

    ``build_slice`` is only the initial project bootstrap consumed by the deterministic
    Fabric generator. It is not the feature scope: CompleteGameDesignPlanner converts
    the original brief into a paginated coverage outline and complete module graph.
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
            payload = _extract_valid_plan_payload(prompt, text)
        except SpecValidationError as initial_error:
            payload = _recover_complete_design_with_safe_bootstrap(prompt, text)
            if payload is None:
                repaired_text = self.router.generate_text(
                    "planner",
                    _repair_messages(prompt, text),
                    media_paths=(),
                    response_format="json",
                )
                try:
                    payload = _extract_valid_plan_payload(prompt, repaired_text)
                except SpecValidationError as repair_error:
                    payload = _recover_complete_design_with_safe_bootstrap(
                        prompt,
                        repaired_text,
                    )
                    if payload is None:
                        raise SpecValidationError(
                            "Planner could not return a complete game design after one "
                            "automatic JSON-format repair. The essential game_design "
                            "and build_slice contract is still incomplete. "
                            f"Initial response: {initial_error} "
                            f"Repair response: {repair_error}"
                        ) from repair_error
        design = payload["game_design"]
        build_slice = payload["build_slice"]
        if not isinstance(design, dict) or not isinstance(build_slice, dict):
            raise SpecValidationError(
                "Planner response has an invalid game_design or build_slice. "
                "Please retry the plan."
            )
        _validate_design(design)
        design = _canonical_game_design(design)
        build_slice = _canonical_build_slice(build_slice)
        classification_issue = ""
        try:
            research_brief = normalize_research_brief(
                prompt,
                design,
                _canonical_research_brief(payload.get("research_brief")),
            )
        except SpecValidationError as exc:
            research_brief = normalize_research_brief(prompt, design)
            classification_issue = (
                "Planner research classification was rejected and replaced "
                f"by the safe request-derived fallback: {exc}"
            )
        design = {
            **design,
            "_research_brief": research_brief,
        }
        if classification_issue:
            design["_research_brief_issue"] = classification_issue
        proposal = _proposal_from_model_data(prompt, build_slice)
        proposal.validate()
        return design, proposal


def _system_prompt() -> str:
    manifest = json.dumps(
        _planner_plugin_manifest(), ensure_ascii=False, sort_keys=True
    )
    return f"""
You are GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.
Return exactly one JSON object and no markdown. Use reference images when provided.
Make game_design a coherent mod-design overview: player fantasy, loop, progression,
requested mod systems, vanilla integration boundaries, art direction, and observable quality goals. Group
repetitive catalogs instead of trying to enumerate an enormous project in this one
response. Preserve every distinct requested system in that overview. The original brief
is passed unchanged to the complete planner, which creates a paginated production
outline and compiles every deliverable. The build_slice is only a deterministic
bootstrap project, never the total feature scope.

Also act as the central research classifier. Derive research domains from this request,
not from a preset genre or example. A domain may cover any requested mechanic,
simulation, sport, social system, entity AI, UI, networking, storage,
visual family, 3D/animation/VFX, audio, accessibility, performance, compatibility,
license or test concern. Do not insert combat, bosses, maps, villages, dungeons or any
other content merely to fill a category. Route every domain to the evidence types and
providers it actually needs. Group huge repeated catalogs, because later production
batches expand them without a project-wide count limit.
When the request actually needs AI or speech, decompose the pipeline instead of using
one vague "voice" or "AI" bucket. Classify inference/tool use, ASR, VAD, TTS,
translation, transport, optional voice adaptation, runtime placement, model license,
dataset provenance, consent/privacy, latency and fallbacks as separate requirements.
Do not add any of them to an unrelated request. Do not name a model merely because it
is recent or popular; later evidence must prove exact revision, format, license,
hardware fit, measured quality and the Minecraft-side integration boundary.
Treat a named commercial game as a request to understand mechanics and experience,
not as permission to copy its proprietary code, characters, logos, textures, models,
audio, writing, maps, or other protected material. Plan original assets unless the
user supplies authorized material or a third-party artifact passes the origin license.

Current planner plugin catalog (the executable registry retains full details):
{manifest}

Return required game_design and build_slice first. research_brief is optional and, when
included, must be the final top-level field.

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
  }},
  "research_brief": {{
    "summary": "plain-language description of what must be understood",
    "domains": [
      {{
        "domain_id": "lowercase_snake_case",
        "objective": "what this research establishes",
        "requirements": ["request-derived requirements, not examples"],
        "evidence_kinds": ["minecraft_api", "dependency", "testing"],
        "queries": ["specific search query including the exact capability"],
        "providers": ["official_docs", "project_rag", "modrinth", "github"],
        "depends_on": ["other_domain_id"]
      }}
    ],
    "unresolved_questions": ["facts that evidence must resolve"]
  }}
}}
Allowed evidence_kinds are minecraft_api, dependency, source_code,
gameplay_reference, visual_reference, texture, model_3d, animation, audio,
license, compatibility, runtime_behavior, performance, accessibility,
local_project, testing, release, ai_inference, agent_tool_use,
speech_recognition, voice_activity_detection, speech_synthesis,
voice_adaptation, voice_conversion, translation, model_runtime, model_license,
dataset_provenance, consent_privacy, latency_budget, and scholarly_reference. Allowed providers are official_docs,
project_rag, modrinth, github, openverse_images, openverse_audio, wikipedia,
blockbench, runtime, huggingface_models, openalex_works, and crossref_works.
Current documentation is discovery evidence only. Implementation claims must be
translated back to exact Minecraft 1.20.1, Fabric, Yarn and Java 17 evidence before
they become code.
Choose only the bootstrap item/block entries genuinely needed before complete-module
compilation. There is no numeric content cap and no requested feature may be hidden.
If combat or mod integration details are not requested, keep those lists empty; their presence in this
JSON shape is not permission to invent them.
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


def _extract_json(text: str) -> dict[str, Any]:
    candidates = tuple(_json_objects(text))
    incomplete_envelope: dict[str, Any] | None = None
    for candidate in candidates:
        envelope = _planner_envelope(candidate)
        if envelope is not None:
            if _has_complete_canonical_fields(envelope):
                return envelope
            if incomplete_envelope is None:
                incomplete_envelope = envelope
    if incomplete_envelope is not None:
        return incomplete_envelope
    if candidates:
        raise SpecValidationError(
            "Planner response is incomplete. Include game_design and build_slice "
            "JSON objects, then retry the plan."
        )
    raise SpecValidationError("Planner did not return a JSON object.")


def _extract_valid_plan_payload(prompt: str, text: str) -> dict[str, Any]:
    """Extract a structurally complete plan before downstream canonicalization."""

    payload = _extract_json(text)
    design = payload.get("game_design")
    build_slice = payload.get("build_slice")
    if not isinstance(design, dict) or not isinstance(build_slice, dict):
        raise SpecValidationError(
            "Planner response has an invalid game_design or build_slice."
        )
    _validate_design(design)
    canonical_build_slice = _canonical_build_slice(build_slice)
    try:
        proposal = _proposal_from_model_data(prompt, canonical_build_slice)
        proposal.validate()
    except SpecValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecValidationError(
            f"Planner build_slice contains invalid values: {exc}"
        ) from exc
    return payload


def _repair_messages(prompt: str, malformed_text: str) -> list[dict[str, str]]:
    """Ask the planner once to repair only its response contract."""

    return [
        {"role": "system", "content": _repair_system_prompt()},
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": _bounded_response_excerpt(malformed_text),
        },
        {
            "role": "user",
            "content": (
                "Your previous answer was incomplete or malformed. Return one complete "
                "JSON object matching the exact game_design and build_slice contract. "
                "Preserve the requested design and every distinct requested system. "
                "Do not include analysis or markdown."
            ),
        },
    ]


def _repair_system_prompt() -> str:
    """Compact repair contract that fits small local-model context windows."""

    return """
Repair an incomplete Minecraft mod-planner response. Return exactly one JSON object
and no analysis or markdown. Preserve the original request and completed design;
do not add unrequested systems. Omit research_brief during repair.

Required shape:
{
  "game_design": {
    "title": "string", "pitch": "string", "core_loop": [], "progression": [],
    "combat": {}, "mod_context": {}, "modules": [], "assets": [],
    "acceptance_tests": []
  },
  "build_slice": {
    "mod_id": "lowercase_snake_case", "mod_name": "string",
    "package_name": "lowercase.dotted.package", "summary": "string",
    "contents": [], "deferred_capabilities": []
  }
}
build_slice is only a small deterministic Fabric bootstrap. Its contents may contain
only explicitly requested item or block entries with content_id, kind,
display_name_en, display_name_ko, color, and a JSON-boolean recipe field.
""".strip()


def _bounded_response_excerpt(text: str, *, limit: int = 8_000) -> str:
    """Keep repair context bounded while retaining both response boundaries."""

    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + "\n[...middle of malformed response omitted...]\n"
        + text[-half:]
    )


def _recover_complete_design_with_safe_bootstrap(
    prompt: str,
    text: str,
) -> dict[str, Any] | None:
    """Recover a truncated envelope without inventing missing game design.

    A deterministic bootstrap is safe only when the model completed the full,
    validated reader-facing design but lost or malformed the outer envelope/build
    slice. If no complete design survived, fail closed so model-authored design is
    never replaced by a generic template.
    """

    design = _last_complete_standalone_design(text)
    if design is None:
        return None
    return {
        "game_design": design,
        "build_slice": _deterministic_bootstrap(prompt, design),
    }


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


def _planner_envelope(value: dict[str, Any]) -> dict[str, Any] | None:
    """Find the canonical plan inside harmless model/API response wrappers."""

    pending: list[dict[str, Any]] = [value]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)
        if _PLAN_FIELDS <= set(candidate):
            return {
                key: candidate[key]
                for key in _PLAN_FIELDS | _OPTIONAL_PLAN_FIELDS
                if key in candidate
            }
        for key in _RESPONSE_WRAPPER_KEYS:
            nested = candidate.get(key)
            if isinstance(nested, dict):
                pending.append(nested)
            elif isinstance(nested, str):
                pending.extend(_json_objects(nested))
        for key in _RESPONSE_LIST_WRAPPER_KEYS:
            nested = candidate.get(key)
            if isinstance(nested, list):
                pending.extend(item for item in nested if isinstance(item, dict))
    return None


def _has_complete_canonical_fields(envelope: dict[str, Any]) -> bool:
    design = envelope["game_design"]
    build_slice = envelope["build_slice"]
    return (
        isinstance(design, dict)
        and isinstance(build_slice, dict)
        and set(_GAME_DESIGN_FIELDS) <= set(design)
        and set(_BUILD_SLICE_FIELDS) <= set(build_slice)
    )


def _canonical_build_slice(build_slice: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _BUILD_SLICE_FIELDS if field not in build_slice]
    if missing:
        raise SpecValidationError(
            "Planner response is incomplete: build_slice is missing "
            + ", ".join(missing)
            + ". Please retry the plan."
        )
    # Bootstrap proposals are an executable contract, so ignore prose metadata
    # rather than passing unexpected keys into the strict deterministic parser.
    return {field: build_slice[field] for field in _BUILD_SLICE_FIELDS}


def _canonical_game_design(design: dict[str, Any]) -> dict[str, Any]:
    """Keep the reader-facing contract free of model-side trace metadata."""

    canonical = {field: design[field] for field in _GAME_DESIGN_FIELDS}
    for field in _OPTIONAL_GAME_DESIGN_FIELDS:
        if field in design:
            canonical[field] = design[field]
    return canonical


def _canonical_research_brief(value: Any) -> Any:
    """Drop explanatory metadata while leaving required research validation intact."""

    if not isinstance(value, dict):
        return value
    brief = {
        field: value[field]
        for field in _RESEARCH_BRIEF_FIELDS
        if field in value
    }
    domains = brief.get("domains")
    if isinstance(domains, list):
        brief["domains"] = [
            {
                field: domain[field]
                for field in _RESEARCH_DOMAIN_FIELDS
                if field in domain
            }
            if isinstance(domain, dict)
            else domain
            for domain in domains
        ]
    return brief


def _validate_design(design: dict[str, Any]) -> None:
    missing = sorted(set(_GAME_DESIGN_FIELDS) - set(design))
    if missing:
        raise SpecValidationError(
            "Planner response is incomplete: game_design is missing "
            + ", ".join(missing)
            + ". Please retry the plan."
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
        design["mod_context"], dict
    ):
        raise SpecValidationError(
            "game_design combat and mod_context must be objects."
        )
    if "art_direction" in design and not isinstance(design["art_direction"], dict):
        raise SpecValidationError("game_design.art_direction must be an object.")
