from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .capability_plugins import plugin_manifest
from .central_research import normalize_research_brief
from .model_router import ModelRouter
from .planner import _proposal_from_model_data
from .spec import Proposal, SpecValidationError


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
        payload = _extract_json(text)
        allowed_shapes = (
            {"game_design", "build_slice"},
            {"game_design", "build_slice", "research_brief"},
        )
        if set(payload) not in allowed_shapes:
            raise SpecValidationError(
                "Planner output must contain game_design, build_slice and the "
                "optional research_brief."
            )
        design = payload["game_design"]
        build_slice = payload["build_slice"]
        if not isinstance(design, dict) or not isinstance(build_slice, dict):
            raise SpecValidationError(
                "Planner output fields must be JSON objects."
            )
        _validate_design(design)
        classification_issue = ""
        try:
            research_brief = normalize_research_brief(
                prompt,
                design,
                payload.get("research_brief"),
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
        plugin_manifest(), ensure_ascii=False, sort_keys=True
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
    "mod_context": {{"vanilla_integration": ["..."], "compatibility_targets": ["..."]}},
    "modules": [{{"plugin_id":"from manifest or custom","status":"implemented|custom","reason":"..."}}],
    "assets": [{{"id":"snake_case","kind":"item|block|entity|gui|environment","brief":"..."}}],
    "acceptance_tests": ["observable test"]
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
        "mod_context",
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
        design["mod_context"], dict
    ):
        raise SpecValidationError(
            "game_design combat and mod_context must be objects."
        )
