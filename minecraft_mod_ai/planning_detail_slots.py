"""Fixed record layouts for every engineering concern (no model-selected keys)."""

# Each concern is a required array of records with exactly these string fields.
# Empty arrays are allowed only with a concrete reason in inapplicable_concerns.
DETAIL_RECORDS = {
    "behavior_contract": {
        "actors": "name role authority",
        "entry_conditions": "trigger owner",
        "preconditions": "condition rejection",
        "inputs": "name type unit range default source",
        "outputs": "name type visibility side_effect",
        "success_postconditions": "condition observation",
        "rejection_postconditions": "condition preserved_state observation",
        "ordering_and_timing": "event order frequency cooldown",
        "boundaries": "limit non_goal",
    },
    "state_model": {
        "variables": "name owner type unit default domain",
        "transitions": "from_state trigger guard mutation to_state",
        "invariants": "condition enforcement",
        "initialization": "owner trigger initial_state",
        "updates": "trigger mutation owner",
        "cleanup": "event action retained_state",
        "concurrency": "entry_path ownership reentrancy_rule",
    },
    "algorithm": {
        "steps": "operation input output next_step",
        "branches": "predicate on_true on_false",
        "formulae": "expression unit rounding clamping",
        "iterations": "order termination work_bound",
        "randomness": "source seed_owner determinism",
        "edge_cases": "condition result",
        "atomic_mutations": "mutations commit rollback",
        "cost_bounds": "operation time_bound memory_bound",
    },
    "integration": {
        "entry_points": "boundary trigger owner",
        "responsibilities": "caller callee contract",
        "target_bindings": "requirement verified_symbol evidence_ref",
        "initialization_order": "component prerequisite order",
        "module_interfaces": "producer consumer input output",
        "side_placement": "component logical_side restriction",
        "compatibility": "assumption extension_point",
        "missing_hooks": "condition fallback",
    },
    "authority_and_network": {
        "decisions": "decision authoritative_side",
        "client_boundaries": "presentation prediction reconciliation",
        "packets": "purpose direction",
        "payloads": "field type validation trust_boundary",
        "security_checks": "condition rejection",
        "synchronization": "state recipients trigger",
        "reconnection": "event resync_action",
        "invalid_packets": "condition action preserved_state",
    },
    "persistence": {
        "stored_state": "state owner scope",
        "serialization": "key type binding_requirement",
        "missing_defaults": "field default",
        "save_triggers": "trigger dirty_rule action",
        "load_behavior": "event action",
        "migration": "source_version destination_version transformation",
        "malformed_data": "condition recovery preserved_state",
        "transfers": "event copy_rule ownership",
    },
    "resources_and_ui": {
        "registries": "purpose identifier binding_requirement",
        "data_resources": "kind purpose owner",
        "assets": "kind purpose production_owner",
        "interactions": "trigger server_action client_feedback",
        "displayed_state": "state authority synchronization",
        "paths": "resource path_binding_requirement",
        "missing_resources": "condition fallback validation",
        "accessibility": "observation localization feedback",
    },
    "failure_and_limits": {
        "invalid_inputs": "condition rejection preserved_state",
        "missing_dependencies": "dependency detection action",
        "repeated_calls": "condition deduplication result",
        "partial_failures": "failure rollback cleanup",
        "interruptions": "event recovery retained_state",
        "concurrency_hazards": "hazard prevention",
        "bounds": "resource limit unit enforcement",
        "diagnostics": "failure signal context",
        "fail_closed": "condition stop_reason",
    },
    "reuse_assessment": {
        "sources": "evidence_ref pattern",
        "unchanged_parts": "part contract",
        "adaptations": "part required_change",
        "compatibility": "api version loader mappings constraint",
        "dependencies": "dependency transitive_impact",
        "provenance": "source constraint",
        "collisions": "owner path_risk resolution",
        "evidence_gaps": "missing_fact blocked_reuse",
        "verdicts": "candidate verdict reason",
    },
    "verification": {
        "success_cases": "given when then measurement",
        "failure_cases": "given when then measurement",
        "boundary_cases": "given when then measurement",
        "invariant_cases": "invariant given when then measurement",
        "persistence_cases": "given when then measurement",
        "multiplayer_cases": "given when then measurement",
        "resource_cases": "given when then measurement",
        "static_gates": "gate expected_result",
        "coverage": "behavior_invariant check",
    },
}


def specification_schema(section):
    records = DETAIL_RECORDS[section]
    properties = {}
    for concern, columns in records.items():
        fields = columns.split()
        properties[concern] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {field: {"type": "string", "minLength": 1} for field in fields},
                "required": fields,
                "additionalProperties": False,
            },
        }
    properties["inapplicable_concerns"] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(records)},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["concern", "reason"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
