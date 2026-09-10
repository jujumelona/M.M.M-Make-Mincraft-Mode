from __future__ import annotations

"""Compile artifact responsibilities into tasks without game-specific builders."""
from dataclasses import dataclass

from .minecraft_template_catalog import (
    FEATURE_CLIENT, FEATURE_DATAGEN, FEATURE_NETWORK, FEATURE_PERSISTENCE,
    FEATURE_WORLDGEN, MinecraftTemplateProfile,
)
from .task_template_catalog import load_template

ROOT_PROVIDE = "target:frozen"


@dataclass(frozen=True)
class TemplateStep:
    name: str
    outcome: str
    consumes: tuple[str, ...]
    provides: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    branch_features: tuple[str, ...] = ()


# Translation of already-declared artifact obligations, never prompt keyword matching.
_ARTIFACT_KIND = {
    "item_model": "model", "block_model": "model", "entity_model": "model",
    "blockstate": "block", "loot_table": "loot", "lang": "language",
    "recipe": "recipe", "tag": "tag", "dimension_data": "dimension",
    "worldgen_data": "worldgen", "gametest": None, "benchmark": None,
}
_FEATURE_ARTIFACT = {
    FEATURE_PERSISTENCE: "saved_data", FEATURE_NETWORK: "network_payload",
    FEATURE_WORLDGEN: "worldgen",
}


def steps_for_profile(profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    capability = profile.capability
    steps = []
    previous = ROOT_PROVIDE

    def append(name, task, anchors=("symbol", "test"), branches=()):
        nonlocal previous
        output = f"{name}:{capability}"
        steps.append(TemplateStep(name, task + f" for {capability}", (previous,), (output,), tuple(anchors), tuple(branches)))
        previous = output

    # Authored gameplay meaning remains in the requirement, not in a game-named builder.
    for name, task in (
        ("trigger", "Bind the authored trigger to its verified entry point"),
        ("input", "Validate the declared input contract"),
        ("state", "Declare the authored state with its explicit owner and defaults"),
        ("transition", "Implement exactly the authored state transition"),
        ("output", "Expose the declared observable output"),
        ("failure", "Implement the declared rejection behavior and preserved state"),
    ):
        append(name, task)

    artifacts = []
    for kind in profile.artifact_kinds:
        if kind not in _ARTIFACT_KIND:
            raise ValueError(f"TEMPLATE_ARTIFACT: unmapped artifact obligation {kind!r}")
        artifact = _ARTIFACT_KIND[kind]
        if artifact:
            artifacts.append((kind if artifact == "model" else artifact, artifact))
    # A model resource alone does not implement its runtime owner.
    for kind, owner in (("entity_model", "entity"), ("item_model", "item"), ("blockstate", "block")):
        if kind in profile.artifact_kinds and (owner, owner) not in artifacts:
            artifacts.insert(0, (owner, owner))
    for feature, artifact in _FEATURE_ARTIFACT.items():
        if feature in profile.features and (artifact, artifact) not in artifacts:
            artifacts.append((artifact, artifact))
    for instance, artifact in dict.fromkeys(artifacts):
        branches = []
        if artifact in {"model", "texture", "animation", "language", "recipe", "loot", "tag", "datagen"}:
            branches.append(FEATURE_DATAGEN)
        if artifact == "network_payload":
            branches.append(FEATURE_NETWORK)
        if artifact in {"worldgen", "dimension", "biome", "structure"}:
            branches.append(FEATURE_WORLDGEN)
        if artifact == "saved_data":
            branches.append(FEATURE_PERSISTENCE)
        for identifier in load_template(f"minecraft/{artifact}")["steps"]:
            task = load_template(identifier)
            append(identifier.replace(f"minecraft/{artifact}/", f"minecraft/{instance}/").replace("/", "_"), task["task"] + " " + " ".join(task["rules"]), task["anchor_kinds"], branches)
    if FEATURE_CLIENT in profile.features:
        for name, task in (("client_projection", "Read only the declared authoritative state projection"),
                           ("client_input", "Bind the declared client input without granting gameplay authority"),
                           ("client_render", "Render the declared client presentation on its verified surface")):
            append(name, task, branches=(FEATURE_CLIENT,))
    append("integration", "Connect only the declared producer and consumer interfaces")
    steps.append(TemplateStep("runtime_scenario", f"Verify the authored observable acceptance scenarios for {capability}", (previous,), (capability,), ("test",)))
    return tuple(steps)


__all__ = ["ROOT_PROVIDE", "TemplateStep", "steps_for_profile"]
