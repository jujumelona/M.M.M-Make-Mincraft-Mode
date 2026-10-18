"""Small, target-aware API shortlist admission for a local Java operation.

A bytecode symbol being accessible and type-correct does not mean its behavior
implements the current task. Reject unrelated method names before MiMo chooses.
This is a ranking/admission aid, never gameplay verification.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# Only task verbs whose *callable* intent is reasonably unambiguous. Ordinary
# methods such as get/set and intermediate value producers remain type-driven.
_ACTIONS = {
    "register": "regis", "registration": "regis", "registering": "regis",
    "spawn": "spawn", "spawning": "spawn",
    "create": "creat", "creation": "creat", "creating": "creat",
    "craft": "craft", "crafting": "craft",
    "attack": "attac", "attacking": "attac",
    "render": "rende", "rendering": "rende",
    "remove": "remov", "removal": "remov", "removing": "remov",
    "update": "updat", "updating": "updat",
    "teleport": "telep", "teleporting": "telep",
    "launch": "launc", "launching": "launc",
    "initialize": "initi", "initialization": "initi",
    "subscribe": "subsc", "subscription": "subsc",
    "trade": "trad", "trading": "trad",
    "mine": "min", "mining": "min",
}
# Task nouns that cannot distinguish a behavior, but may legitimately qualify
# an API with a generic name such as registerItem or createEntity.
_NEUTRAL = frozenset({
    "minecraft", "fabric", "java", "game", "mod", "mods", "plugin",
    "function", "method", "value", "values", "object", "logic", "code",
    "required", "provided", "target", "current", "this", "that", "the",
    "with", "from", "into", "only", "should", "which", "have", "using",
    "input", "output", "type", "types", "system", "behavior", "feature",
})


def _words(value: object) -> tuple[str, ...]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return tuple(re.findall(r"[a-z][a-z0-9]*", value.casefold()))


def _stem(term: str) -> str:
    if term in _ACTIONS:
        return _ACTIONS[term]
    if term.endswith("ies"):
        return term[:-3] + "y"
    if term.endswith("es") and len(term) >= 5:
        return term[:-2]
    if term.endswith("s") and len(term) >= 5:
        return term[:-1]
    return term


def root_action_stems(purpose: str) -> frozenset[str]:
    """Recognize executable goal verbs, rather than incidental type words."""
    return frozenset(_ACTIONS[word] for word in _words(purpose)
                     if word in _ACTIONS)


# Root registration must target a registry for the requested domain.
# Matching the word "register" alone admits RecipeType/RuleTestType for entities.
_DOMAINS = {
    "entity": {"entity", "entities", "alien", "mob", "mobs", "creature", "creatures"},
    "item": {"item", "items"},
    "block": {"block", "blocks"},
    "recipe": {"recipe", "recipes", "crafting", "smelting"},
    "sound": {"sound", "sounds", "audio"},
    "fluid": {"fluid", "fluids"},
    "biome": {"biome", "biomes"},
    "particle": {"particle", "particles"},
    "attribute": {"attribute", "attributes"},
    "component": {"component", "components"},
}
_GENERIC_REGISTRY_OWNERS = frozenset({
    "registry", "registries", "builtinregistries", "deferredregister",
})


def _registration_owner_matches(purpose: str, row: Mapping[str, object]) -> bool:
    words = set(_words(purpose))
    requested = {domain for domain, aliases in _DOMAINS.items() if words & aliases}
    if not requested:
        return True
    owner = str(row.get("owner") or "").rsplit(".", 1)[-1].rsplit("$", 1)[-1]
    if owner.casefold() in _GENERIC_REGISTRY_OWNERS:
        return True
    owner_terms = set(_words(owner))
    owner_domains = {domain for domain, aliases in _DOMAINS.items()
                     if owner_terms & aliases}
    # A specific registrar for the wrong domain can never implement this one.
    if not owner_domains or not owner_domains & requested:
        return False
    # EntityRendererRegistry registers renderers, not entity types. Likewise
    # "client" registrations cannot satisfy an ordinary server-side entity
    # registration task.
    qualifier_roles = {
        "renderer", "renderers", "rendering", "client", "screen", "model",
        "data", "serializer", "selector", "option", "debug", "subscription",
        "fluid", "interaction",
    }
    if owner_terms & qualifier_roles and not words & qualifier_roles:
        return False
    return True


def api_semantic_admissible(
    op: str, purpose: str, row: Mapping[str, object],
    requirements: Sequence[Mapping[str, object]] = (),
) -> bool:
    """For a root *action* call, require matching callable behavior evidence.

    Typed child expressions remain governed by the exact slot type contract.
    Generic utilities cannot become registration operations just because their
    owner/package contains Entity, Registry or Minecraft.
    """
    if op != "call" or requirements:
        return True
    requested = _words(purpose)
    action_stems = root_action_stems(purpose)
    if not action_stems:
        return True  # No explicit action: preserve normal target API retrieval.
    method = _words(row.get("name", ""))
    if not method:
        return False
    # Match actual callable names, not package/return types.
    matching_actions = {
        action for action in action_stems
        if any(part.startswith(action) for part in method)
    }
    if not matching_actions:
        return False
    # An action with unrelated extra qualifiers is a different API. In
    # particular registerDebugValues must not implement register alien entity.
    rest = [
        _stem(term) for term in method
        if not any(term.startswith(action) for action in matching_actions)
        and term not in _NEUTRAL
    ]
    intent = {_stem(term) for term in requested if term not in _NEUTRAL}
    # Generic register(...) still requires the correct domain owner.
    if rest and not any(term in intent for term in rest):
        return False
    if "regis" in matching_actions and not _registration_owner_matches(purpose, row):
        return False
    return True
