from __future__ import annotations

"""Composable host-owned Minecraft implementation task templates.

Each template materializes a small, independently verifiable DAG.  Core Minecraft
responsibilities are decomposed before the coder model runs, so a small model receives a
single concrete implementation outcome instead of being asked to invent architecture.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .minecraft_template_catalog import (
    FEATURE_CLIENT,
    FEATURE_DATAGEN,
    FEATURE_NETWORK,
    FEATURE_PERSISTENCE,
    FEATURE_REGISTRY,
    FEATURE_WORLDGEN,
    MinecraftTemplateProfile,
)

ROOT_PROVIDE = "target:frozen"


@dataclass(frozen=True)
class TemplateStep:
    name: str
    outcome: str
    consumes: tuple[str, ...]
    provides: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    branch_features: tuple[str, ...] = ()


def _token(kind: str, capability: str) -> str:
    return f"{kind}:{capability}"


def _step(
    name: str,
    outcome: str,
    consumes: Sequence[str],
    provides: Sequence[str],
    anchors: Sequence[str],
    *branches: str,
) -> TemplateStep:
    return TemplateStep(
        name=name,
        outcome=outcome,
        consumes=tuple(consumes),
        provides=tuple(provides),
        anchor_kinds=tuple(anchors),
        branch_features=tuple(dict.fromkeys(branches)),
    )


def _semantic_contract(capability: str) -> tuple[TemplateStep, str]:
    contract = _token("semantic_contract", capability)
    return (
        _step(
            "semantic_contract",
            (
                f"Freeze the authored inputs, authoritative state owner, state transitions, "
                f"integration boundary and observable success/rejection contract for {capability}"
            ),
            (ROOT_PROVIDE,),
            (contract,),
            ("symbol", "test"),
        ),
        contract,
    )


def _standard_extensions(
    capability: str,
    profile: MinecraftTemplateProfile,
    *,
    core_steps: Sequence[TemplateStep],
    contract: str,
    behavior: str,
    registry: str | None = None,
    handled_features: frozenset[str] = frozenset(),
) -> tuple[TemplateStep, ...]:
    steps = list(core_steps)
    terminal_inputs: list[str] = [behavior]

    if FEATURE_PERSISTENCE in profile.features and FEATURE_PERSISTENCE not in handled_features:
        codec = _token("state_codec", capability)
        persisted = _token("persisted_state", capability)
        steps.extend(
            (
                _step(
                    "state_codec",
                    f"Define the versioned/default-safe persistence codec and state scope for {capability}",
                    (contract,),
                    (codec,),
                    ("symbol", "test"),
                    FEATURE_PERSISTENCE,
                ),
                _step(
                    "persistence_binding",
                    f"Persist mutations, mark/change state correctly, reload it and reject stale or invalid state for {capability}",
                    (codec, behavior),
                    (persisted,),
                    ("symbol", "test"),
                    FEATURE_PERSISTENCE,
                ),
            )
        )
        terminal_inputs.append(persisted)

    network_output: str | None = None
    if FEATURE_NETWORK in profile.features and FEATURE_NETWORK not in handled_features:
        payload = _token("payload_codec", capability)
        network_output = _token("network_sync", capability)
        steps.extend(
            (
                _step(
                    "payload_contract",
                    f"Define typed payload direction, identifier and stream/packet codec for {capability}",
                    (contract,),
                    (payload,),
                    ("symbol", "registry_id", "test"),
                    FEATURE_NETWORK,
                ),
                _step(
                    "server_handler_sync",
                    f"Register logical-side handlers, validate client-originated input on the server and synchronize authoritative state for {capability}",
                    (payload, behavior),
                    (network_output,),
                    ("symbol", "test"),
                    FEATURE_NETWORK,
                ),
            )
        )
        terminal_inputs.append(network_output)

    resource_output: str | None = None
    if FEATURE_DATAGEN in profile.features and FEATURE_DATAGEN not in handled_features:
        resource_output = _token("generated_resources", capability)
        data_input = registry or contract
        steps.append(
            _step(
                "data_resource_binding",
                f"Generate and validate every required recipe, tag, loot, model, language or other data/resource artifact for {capability}",
                (data_input,),
                (resource_output,),
                ("resource", "test"),
                FEATURE_DATAGEN,
            )
        )
        terminal_inputs.append(resource_output)

    world_output: str | None = None
    if FEATURE_WORLDGEN in profile.features and FEATURE_WORLDGEN not in handled_features:
        configured = _token("configured_feature", capability)
        placed = _token("placed_feature", capability)
        world_output = _token("world_binding", capability)
        steps.extend(
            (
                _step(
                    "configured_feature",
                    f"Define the data-driven configured world-generation contract for {capability}",
                    (registry or contract,),
                    (configured,),
                    ("symbol", "resource", "test"),
                    FEATURE_WORLDGEN,
                    FEATURE_DATAGEN,
                ),
                _step(
                    "placed_feature",
                    f"Define placement modifiers, density/height constraints and placement key for {capability}",
                    (configured,),
                    (placed,),
                    ("symbol", "resource", "test"),
                    FEATURE_WORLDGEN,
                    FEATURE_DATAGEN,
                ),
                _step(
                    "world_target_binding",
                    f"Bind {capability} only to its approved biome/dimension/world-generation targets",
                    (placed,),
                    (world_output,),
                    ("symbol", "resource", "test"),
                    FEATURE_WORLDGEN,
                ),
            )
        )
        terminal_inputs.append(world_output)

    if FEATURE_CLIENT in profile.features and FEATURE_CLIENT not in handled_features:
        client_contract = _token("client_contract", capability)
        client_surface = _token("client_surface", capability)
        client_input = network_output or behavior
        steps.extend(
            (
                _step(
                    "client_contract",
                    f"Define the side-safe presentation contract and authoritative state projection for {capability}",
                    (contract, client_input),
                    (client_contract,),
                    ("symbol",),
                    FEATURE_CLIENT,
                ),
                _step(
                    "client_surface",
                    f"Bind and render the requested client presentation without moving gameplay authority to the client for {capability}",
                    (client_contract,),
                    (client_surface,),
                    ("symbol", "resource", "test"),
                    FEATURE_CLIENT,
                ),
            )
        )
        terminal_inputs.append(client_surface)

    failure = _token("failure_contract", capability)
    steps.append(
        _step(
            "failure_contract",
            f"Implement explicit invalid-input, missing-resource, boundary-state and cleanup/recovery behavior for {capability}",
            (behavior,),
            (failure,),
            ("symbol", "test"),
        )
    )
    terminal_inputs.append(failure)

    steps.append(
        _step(
            "runtime_scenario",
            f"Verify the complete authored success, rejection, state-change and reload/side-visible runtime scenarios for {capability}",
            tuple(dict.fromkeys(terminal_inputs)),
            (capability,),
            ("test",),
        )
    )
    return tuple(steps)


def _item(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("registry_id", capability)
    behavior = _token("item_behavior", capability)
    core = (
        *base,
        _step(
            "registry_identity",
            f"Reserve stable namespaced item/component registry identities for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol"),
            FEATURE_REGISTRY,
        ),
        _step(
            "item_behavior",
            f"Implement server-safe item properties, component/state handling and authored interactions for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _combat_item(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("registry_id", capability)
    behavior = _token("combat_behavior", capability)
    core = (
        *base,
        _step(
            "registry_identity",
            f"Register stable weapon/item identities and authored combat data for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol"),
            FEATURE_REGISTRY,
        ),
        _step(
            "combat_behavior",
            f"Implement server-authoritative attack validation, damage/effect application, cooldown or durability changes only as authored for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _combat_service(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    behavior = _token("combat_service", capability)
    core = (
        *base,
        _step(
            "server_combat_service",
            f"Implement authoritative targeting, attack eligibility, damage calculation and terminal combat state for {capability}",
            (contract,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _entity(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("entity_registry", capability)
    behavior = _token("entity_lifecycle", capability)
    core = (
        *base,
        _step(
            "entity_type_attributes",
            f"Register the entity type, dimensions/tracking contract and required attributes for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "entity_lifecycle",
            f"Implement spawn/creation, tick or goal lifecycle, interaction and removal/death cleanup for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _hostile_entity_combat(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("combat_entity", capability)
    behavior = _token("combat_behavior", capability)
    core = (
        *base,
        _step(
            "entity_attributes_spawn",
            f"Register entity identity/attributes and only the authored planet/biome spawn conditions for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "ai_damage_death",
            f"Implement server-authoritative AI goals, attacks, damage, death and cleanup for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _boss(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("boss_registry", capability)
    behavior = _token("boss_behavior", capability)
    core = (
        *base,
        _step(
            "boss_identity_attributes",
            f"Register boss entity identity, attributes and encounter initialization for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "boss_phase_combat",
            f"Implement authoritative boss phase/state transitions, attacks, damage rules and death outcome for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _crafting(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    recipe = _token("recipe_data", capability)
    behavior = _token("crafting_behavior", capability)
    core = (
        *base,
        _step(
            "recipe_definition",
            f"Define deterministic data-driven ingredients, output and unlock/visibility contract for {capability}",
            (contract,),
            (recipe,),
            ("resource", "test"),
            FEATURE_DATAGEN,
        ),
        _step(
            "recipe_server_validation",
            f"Validate inputs and produce/consume inventory state atomically for {capability}",
            (recipe, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        handled_features=frozenset({FEATURE_DATAGEN}),
    )


def _data_resource(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    behavior = _token("data_resource", capability)
    core = (
        *base,
        _step(
            "data_definition",
            f"Generate the exact namespaced data/resource definition and validate every referenced registry key for {capability}",
            (contract,),
            (behavior,),
            ("resource", "test"),
            FEATURE_DATAGEN,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        handled_features=frozenset({FEATURE_DATAGEN}),
    )


def _inventory(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    behavior = _token("inventory_transaction", capability)
    core = (
        *base,
        _step(
            "inventory_transaction",
            f"Implement atomic slot eligibility, insertion/extraction, stack-count and full-inventory rejection for {capability}",
            (contract,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _network(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    codec = _token("payload_codec", capability)
    behavior = _token("network_sync", capability)
    core = (
        *base,
        _step(
            "payload_codec_registration",
            f"Define and register payload identifier, direction and codec for {capability}",
            (contract,),
            (codec,),
            ("registry_id", "symbol", "test"),
            FEATURE_NETWORK,
        ),
        _step(
            "logical_side_handler",
            f"Handle the payload on the correct logical side, validate serverbound data and synchronize authoritative results for {capability}",
            (codec, contract),
            (behavior,),
            ("symbol", "test"),
            FEATURE_NETWORK,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        handled_features=frozenset({FEATURE_NETWORK}),
    )


def _persistence(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    codec = _token("state_codec", capability)
    behavior = _token("persisted_state", capability)
    core = (
        *base,
        _step(
            "state_codec",
            f"Define state scope, default construction and versioned codec/serializer for {capability}",
            (contract,),
            (codec,),
            ("symbol", "test"),
            FEATURE_PERSISTENCE,
        ),
        _step(
            "saved_state_binding",
            f"Bind state to its correct world/entity/block/item owner, mark mutations and prove save/reload behavior for {capability}",
            (codec, contract),
            (behavior,),
            ("symbol", "test"),
            FEATURE_PERSISTENCE,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        handled_features=frozenset({FEATURE_PERSISTENCE}),
    )


def _client_ui(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    behavior = _token("client_surface", capability)
    core = (
        *base,
        _step(
            "client_surface",
            f"Implement screen/HUD/widget lifecycle, rendering and close/back behavior using only client-safe projected state for {capability}",
            (contract,),
            (behavior,),
            ("symbol", "resource", "test"),
            FEATURE_CLIENT,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        handled_features=frozenset({FEATURE_CLIENT}),
    )


def _container_ui(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("menu_registry", capability)
    server_menu = _token("server_menu", capability)
    behavior = _token("client_surface", capability)
    core = (
        *base,
        _step(
            "menu_registry_contract",
            f"Register menu/container identity and authoritative slot/property schema for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "server_menu_validation",
            f"Implement server-owned slot transfer, button/action validation and state synchronization contract for {capability}",
            (registry, contract),
            (server_menu,),
            ("symbol", "test"),
        ),
        _step(
            "client_screen",
            f"Render the client screen from synchronized menu state without granting client authority for {capability}",
            (server_menu,),
            (behavior,),
            ("symbol", "resource", "test"),
            FEATURE_CLIENT,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        registry=registry,
        handled_features=frozenset({FEATURE_CLIENT}),
    )


def _economy(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("economy_model", capability)
    behavior = _token("transaction_service", capability)
    core = (
        *base,
        _step(
            "economy_model",
            f"Define currency/resource representation, price/offer/stock rules and transaction invariants for {capability}",
            (contract,),
            (schema,),
            ("symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "transaction_service",
            f"Implement one atomic server-authoritative debit/credit, inventory transfer and rejection service for {capability}",
            (schema, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=schema)


def _machine(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("machine_registry", capability)
    state = _token("block_entity_state", capability)
    behavior = _token("machine_behavior", capability)
    core = (
        *base,
        _step(
            "block_blockentity_registry",
            f"Register block, block item and block-entity type identities for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource"),
            FEATURE_REGISTRY,
        ),
        _step(
            "blockentity_state_owner",
            f"Define block-entity owned state, inventory/capacity invariants and ticker eligibility for {capability}",
            (registry, contract),
            (state,),
            ("symbol", "test"),
        ),
        _step(
            "server_tick_behavior",
            f"Implement authoritative tick/processing transitions, input consumption and output production for {capability}",
            (state,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _stateful_service(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("state_schema", capability)
    behavior = _token("stateful_service", capability)
    core = (
        *base,
        _step(
            "state_schema",
            f"Define authoritative values, bounds, mutation triggers and ownership for {capability}",
            (contract,),
            (schema,),
            ("symbol", "test"),
        ),
        _step(
            "state_transition_service",
            f"Implement bounded server-owned state changes and rejection behavior for {capability}",
            (schema,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _vehicle(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("vehicle_registry", capability)
    state = _token("vehicle_state", capability)
    behavior = _token("vehicle_control", capability)
    core = (
        *base,
        _step(
            "vehicle_entity_registry",
            f"Register vehicle entity/type identity, dimensions/tracking and required item/resource identities for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "vehicle_state_schema",
            f"Define driver/occupant, movement/control and authored vehicle state invariants for {capability}",
            (registry, contract),
            (state,),
            ("symbol", "test"),
        ),
        _step(
            "server_vehicle_control",
            f"Validate control inputs and mutate movement/vehicle state on the logical server for {capability}",
            (state,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _dimension_worldgen(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("dimension_registry", capability)
    generated = _token("dimension_generation", capability)
    behavior = _token("destination_access", capability)
    core = (
        *base,
        _step(
            "dimension_registry_data",
            f"Define namespaced dimension/worldgen registry keys and data-driven dimension contract for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "resource", "test"),
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_WORLDGEN,
        ),
        _step(
            "dimension_generation",
            f"Bind terrain/generator, biome and required world-generation data for {capability}",
            (registry,),
            (generated,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
            FEATURE_DATAGEN,
        ),
        _step(
            "destination_access",
            f"Implement server-safe access, arrival placement and return/world linkage for {capability}",
            (generated, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        registry=registry,
        handled_features=frozenset({FEATURE_DATAGEN, FEATURE_WORLDGEN}),
    )


def _structure_worldgen(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("worldgen_registry", capability)
    configured = _token("configured_feature", capability)
    placed = _token("placed_feature", capability)
    behavior = _token("world_binding", capability)
    core = (
        *base,
        _step(
            "worldgen_registry_data",
            f"Define namespaced structure/feature registry and data contract for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "resource", "test"),
            FEATURE_REGISTRY,
            FEATURE_DATAGEN,
            FEATURE_WORLDGEN,
        ),
        _step(
            "configured_generation",
            f"Define configured structure/feature contents and generation parameters for {capability}",
            (registry,),
            (configured,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
            FEATURE_DATAGEN,
        ),
        _step(
            "placed_generation",
            f"Define placement constraints and placement key for {capability}",
            (configured,),
            (placed,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
            FEATURE_DATAGEN,
        ),
        _step(
            "world_target_binding",
            f"Bind {capability} to approved biome/dimension targets and prove generation visibility",
            (placed,),
            (behavior,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        registry=registry,
        handled_features=frozenset({FEATURE_DATAGEN, FEATURE_WORLDGEN}),
    )


def _resource(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("resource_registry", capability)
    behavior = _token("resource_acquisition", capability)
    core = (
        *base,
        _step(
            "resource_registry",
            f"Register resource item/block identity and authoritative acquisition accounting for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "acquisition_consumption",
            f"Implement authored obtain/drop/collect, inventory transfer, consumption and rejection behavior for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _worldgen_resource(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("resource_registry", capability)
    configured = _token("configured_feature", capability)
    placed = _token("placed_feature", capability)
    world = _token("world_binding", capability)
    behavior = _token("resource_acquisition", capability)
    core = (
        *base,
        _step(
            "resource_registry",
            f"Register the special resource block/item identities and mine/drop contract for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "configured_feature",
            f"Define ore/resource configured-feature targets and vein/content parameters for {capability}",
            (registry,),
            (configured,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
            FEATURE_DATAGEN,
        ),
        _step(
            "placed_feature",
            f"Define placement count/rarity/height and placement key for {capability}",
            (configured,),
            (placed,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
            FEATURE_DATAGEN,
        ),
        _step(
            "biome_dimension_binding",
            f"Bind the placed resource only to its approved planet/biome/dimension targets for {capability}",
            (placed,),
            (world,),
            ("symbol", "resource", "test"),
            FEATURE_WORLDGEN,
        ),
        _step(
            "mining_loot_acquisition",
            f"Bind mining/drop/loot/tag behavior and prove inventory acquisition for {capability}",
            (registry, world),
            (behavior,),
            ("symbol", "resource", "test"),
            FEATURE_DATAGEN,
        ),
    )
    return _standard_extensions(
        capability,
        profile,
        core_steps=core,
        contract=contract,
        behavior=behavior,
        registry=registry,
        handled_features=frozenset({FEATURE_DATAGEN, FEATURE_WORLDGEN}),
    )


def _progression(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("progression_schema", capability)
    behavior = _token("progression_transition", capability)
    core = (
        *base,
        _step(
            "progression_state_schema",
            f"Define progression state, bounds, milestones and only the authored unlock/effect rules for {capability}",
            (contract,),
            (schema,),
            ("symbol", "test"),
        ),
        _step(
            "progression_transition_service",
            f"Implement authoritative gain/loss/advance transitions and rejection behavior for {capability}",
            (schema,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _quest(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("quest_schema", capability)
    behavior = _token("quest_progression", capability)
    core = (
        *base,
        _step(
            "quest_state_schema",
            f"Define quest/objective states, triggers and completion/reward contract for {capability}",
            (contract,),
            (schema,),
            ("symbol", "test"),
        ),
        _step(
            "objective_transition_service",
            f"Implement authoritative objective tracking, step transitions, completion and duplicate/rejection handling for {capability}",
            (schema,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _ability(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("ability_schema", capability)
    behavior = _token("ability_execution", capability)
    core = (
        *base,
        _step(
            "ability_state_schema",
            f"Define authored cost/cooldown/target/effect state and validation rules for {capability}",
            (contract,),
            (schema,),
            ("symbol", "test"),
        ),
        _step(
            "server_ability_execution",
            f"Validate and execute the ability/effect on the logical server, including rejection and terminal state for {capability}",
            (schema,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _spacecraft_assembly(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("spacecraft_component_schema", capability)
    behavior = _token("spacecraft_assembly", capability)
    core = (
        *base,
        _step(
            "component_slot_schema",
            f"Define spacecraft part registry identities, slots, compatibility and assembly-completion invariants for {capability}",
            (contract,),
            (schema,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "assembly_transaction_service",
            f"Acquire/consume authored inputs atomically, validate part compatibility and mutate assembled spacecraft state for {capability}",
            (schema, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=schema)


def _spacecraft_upgrade(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    schema = _token("spacecraft_upgrade_schema", capability)
    behavior = _token("spacecraft_upgrade", capability)
    core = (
        *base,
        _step(
            "upgrade_stat_schema",
            f"Define only the authored spacecraft stat/module dimensions, tiers, caps and compatibility rules for {capability}",
            (contract,),
            (schema,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "upgrade_transaction_service",
            f"Validate prerequisites/costs, atomically purchase/install/remove the upgrade and recompute authoritative spacecraft state for {capability}",
            (schema, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=schema)


def _crew(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("crew_schema", capability)
    behavior = _token("crew_lifecycle", capability)
    core = (
        *base,
        _step(
            "crew_role_skill_schema",
            f"Define crew identity, authored roles/skills, assignments and registry/resource identities for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "recruit_assign_replace_lifecycle",
            f"Implement authoritative hire/recruit, assign, remove/death and replacement transitions for {capability}",
            (registry, contract),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _space_travel(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    policy = _token("launch_policy", capability)
    transaction = _token("travel_transaction", capability)
    behavior = _token("world_transition", capability)
    core = (
        *base,
        _step(
            "launch_unlock_policy",
            f"Freeze required versus optional launch prerequisites and destination eligibility for {capability}",
            (contract,),
            (policy,),
            ("symbol", "test"),
        ),
        _step(
            "fuel_destination_transaction",
            f"Validate launch state, destination and fuel; consume fuel atomically only on an accepted transition for {capability}",
            (policy,),
            (transaction,),
            ("symbol", "registry_id", "test"),
        ),
        _step(
            "world_transition",
            f"Perform authoritative destination transfer, safe arrival/return placement and world linkage for {capability}",
            (transaction,),
            (behavior,),
            ("symbol", "resource", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior)


def _colony(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    registry = _token("colony_identity", capability)
    state = _token("colony_state", capability)
    behavior = _token("colony_progression", capability)
    core = (
        *base,
        _step(
            "placement_ownership",
            f"Validate placement/world constraints and establish server-owned colony identity/ownership for {capability}",
            (contract,),
            (registry,),
            ("registry_id", "symbol", "resource", "test"),
            FEATURE_REGISTRY,
        ),
        _step(
            "colony_state_schema",
            f"Define storage/world linkage and only the authored development state for {capability}",
            (registry, contract),
            (state,),
            ("symbol", "test"),
        ),
        _step(
            "colony_progression_service",
            f"Implement authoritative development/resource-cost transitions, ownership checks and storage mutation for {capability}",
            (state,),
            (behavior,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=behavior, registry=registry)


def _software_quality(capability: str, _profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    baseline = _token("baseline", capability)
    patch = _token("patch", capability)
    return (
        _step(
            "baseline_contract",
            f"Capture behavior and explicitly authored performance/resource baselines for {capability}",
            (ROOT_PROVIDE,),
            (baseline,),
            ("test",),
        ),
        _step(
            "compatibility_patch",
            f"Apply the smallest target-compatible implementation change for {capability}",
            (baseline,),
            (patch,),
            ("symbol", "build_config"),
        ),
        _step(
            "regression_proof",
            f"Prove behavior equivalence plus the authored performance/resource criterion for {capability}",
            (patch,),
            (capability,),
            ("test",),
        ),
    )


def _custom_gameplay(capability: str, profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    base, contract = _semantic_contract(capability)
    behavior = _token("authoritative_behavior", capability)
    integration = _token("integration", capability)
    core = (
        *base,
        _step(
            "authoritative_behavior",
            f"Implement only the user-authored state transition and authoritative validation for {capability}",
            (contract,),
            (behavior,),
            ("symbol", "test"),
        ),
        _step(
            "integration_binding",
            f"Bind {capability} to its existing Minecraft event/service/interaction boundary without inventing an unrelated subsystem",
            (behavior,),
            (integration,),
            ("symbol", "test"),
        ),
    )
    return _standard_extensions(capability, profile, core_steps=core, contract=contract, behavior=integration)


_BUILDERS: dict[str, Callable[[str, MinecraftTemplateProfile], tuple[TemplateStep, ...]]] = {
    "item": _item,
    "combat_item": _combat_item,
    "combat_service": _combat_service,
    "entity": _entity,
    "hostile_entity_combat": _hostile_entity_combat,
    "boss_combat": _boss,
    "crafting": _crafting,
    "data_resource": _data_resource,
    "inventory_service": _inventory,
    "network": _network,
    "persistence": _persistence,
    "client_ui": _client_ui,
    "container_ui": _container_ui,
    "economy": _economy,
    "machine": _machine,
    "stateful_service": _stateful_service,
    "vehicle": _vehicle,
    "dimension_worldgen": _dimension_worldgen,
    "structure_worldgen": _structure_worldgen,
    "resource": _resource,
    "worldgen_resource": _worldgen_resource,
    "progression": _progression,
    "quest": _quest,
    "ability": _ability,
    "spacecraft_assembly": _spacecraft_assembly,
    "spacecraft_upgrade": _spacecraft_upgrade,
    "crew": _crew,
    "space_travel": _space_travel,
    "colony": _colony,
    "software_quality": _software_quality,
    "custom_gameplay": _custom_gameplay,
}


def steps_for_profile(profile: MinecraftTemplateProfile) -> tuple[TemplateStep, ...]:
    builder = _BUILDERS.get(profile.template_id)
    if builder is None:
        raise ValueError(f"unknown Minecraft template id: {profile.template_id!r}")
    steps = builder(profile.capability, profile)
    if not steps or profile.capability not in steps[-1].provides:
        raise ValueError(
            f"Minecraft template {profile.template_id!r} does not terminate in {profile.capability!r}"
        )
    return steps


__all__ = ["ROOT_PROVIDE", "TemplateStep", "steps_for_profile"]
