from __future__ import annotations

"""Immutable exhaustive micro-template pipeline and completion ledger.

The pipeline has no router and no template chooser. Every declared stage is invoked exactly
once in fixed order for every project run. Applicability is a stage result:
``not_required`` / ``not_applicable`` are successful terminal accounting states, not skips.
Failures do not short-circuit later stages, so validation, repair, revalidation, packaging,
and completion accounting are still executed and visible.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

PIPELINE_SCHEMA = "mmm/exhaustive-fixed-pipeline-v1"
LEDGER_SCHEMA = "mmm/exhaustive-execution-ledger-v1"

TERMINAL_STATUSES = frozenset(
    {"pass", "required", "not_required", "not_applicable", "fail"}
)
SUCCESS_STATUSES = frozenset(
    {"pass", "required", "not_required", "not_applicable"}
)


@dataclass(frozen=True)
class StageDefinition:
    stage_id: str
    phase: str
    subject: str
    operation: str
    purpose: str
    model_slot_allowed: bool = True


@dataclass(frozen=True)
class StageExecution:
    stage_id: str
    status: str
    artifacts: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineDefinition:
    schema: str
    stages: tuple[StageDefinition, ...]

    @property
    def definition_sha256(self) -> str:
        payload = [
            {
                "stage_id": stage.stage_id,
                "phase": stage.phase,
                "subject": stage.subject,
                "operation": stage.operation,
                "purpose": stage.purpose,
                "model_slot_allowed": stage.model_slot_allowed,
            }
            for stage in self.stages
        ]
        encoded = json.dumps(
            {"schema": self.schema, "stages": payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        ids = tuple(stage.stage_id for stage in self.stages)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("fixed pipeline stage IDs must be non-empty and unique")
        for stage in self.stages:
            if not all(
                (
                    stage.stage_id.strip(),
                    stage.phase.strip(),
                    stage.subject.strip(),
                    stage.operation.strip(),
                    stage.purpose.strip(),
                )
            ):
                raise ValueError(f"incomplete fixed stage definition: {stage!r}")


class ExecutionLedger:
    """One terminal execution record for every fixed pipeline stage."""

    def __init__(self, definition: PipelineDefinition):
        definition.validate()
        self.definition = definition
        self._expected = tuple(stage.stage_id for stage in definition.stages)
        self._expected_set = frozenset(self._expected)
        self._records: dict[str, StageExecution] = {}

    @property
    def records(self) -> tuple[StageExecution, ...]:
        return tuple(
            self._records[stage_id]
            for stage_id in self._expected
            if stage_id in self._records
        )

    @property
    def unexecuted(self) -> tuple[str, ...]:
        return tuple(stage_id for stage_id in self._expected if stage_id not in self._records)

    def record(self, execution: StageExecution) -> None:
        if execution.stage_id not in self._expected_set:
            raise ValueError(f"unknown fixed stage: {execution.stage_id}")
        if execution.stage_id in self._records:
            raise ValueError(f"fixed stage executed more than once: {execution.stage_id}")
        if execution.status not in TERMINAL_STATUSES:
            raise ValueError(
                f"stage {execution.stage_id} has non-terminal status {execution.status!r}"
            )
        self._records[execution.stage_id] = execution

    def summary(self) -> dict[str, Any]:
        counts = Counter(record.status for record in self._records.values())
        failed = counts["fail"]
        return {
            "schema": LEDGER_SCHEMA,
            "pipeline_schema": self.definition.schema,
            "pipeline_sha256": self.definition.definition_sha256,
            "total": len(self._expected),
            "executed": len(self._records),
            "unexecuted": len(self.unexecuted),
            "failed": failed,
            "status_counts": dict(sorted(counts.items())),
            "project_complete": self.project_complete,
        }

    @property
    def project_complete(self) -> bool:
        return (
            len(self._records) == len(self._expected)
            and not self.unexecuted
            and all(record.status in SUCCESS_STATUSES for record in self._records.values())
        )


StageExecutor = Callable[
    [StageDefinition, Mapping[str, Any]],
    StageExecution | Mapping[str, Any] | str,
]


def _fixed(
    prefix: str,
    phase: str,
    operations: Sequence[str],
    *,
    model_slot_allowed: bool = True,
) -> tuple[StageDefinition, ...]:
    return tuple(
        StageDefinition(
            stage_id=f"{prefix}-{index:03d}",
            phase=phase,
            subject=phase,
            operation=operation,
            purpose=f"{operation.replace('_', ' ')} for the fixed {phase} contract",
            model_slot_allowed=model_slot_allowed,
        )
        for index, operation in enumerate(operations, 1)
    )


def _matrix(
    prefix: str,
    phase: str,
    subjects: Sequence[str],
    operations: Sequence[str],
    *,
    model_slot_allowed: bool = True,
) -> tuple[StageDefinition, ...]:
    output: list[StageDefinition] = []
    for subject_index, subject in enumerate(subjects, 1):
        for operation_index, operation in enumerate(operations, 1):
            output.append(
                StageDefinition(
                    stage_id=f"{prefix}-{subject_index:02d}-{operation_index:02d}",
                    phase=phase,
                    subject=subject,
                    operation=operation,
                    purpose=f"{operation.replace('_', ' ')} for {subject.replace('_', ' ')}",
                    model_slot_allowed=model_slot_allowed,
                )
            )
    return tuple(output)


_CAPTURE = (
    "raw_request_receipt", "source_span_index", "attachment_inventory",
    "existing_project_presence", "user_constraint_receipt", "must_preserve_receipt",
    "forbidden_change_receipt", "completion_expectation_receipt", "explicit_version_receipt",
    "explicit_loader_receipt", "explicit_language_receipt", "reference_asset_receipt",
)

_INTENT = (
    "primary_goal_extract", "requested_artifact_extract", "action_verb_extract",
    "named_entity_extract", "external_franchise_detect", "target_game_detect",
    "target_mod_platform_detect", "version_detect", "loader_detect",
    "programming_language_detect", "visual_style_extract", "gameplay_style_extract",
    "feature_mentions_extract", "quantity_constraints_extract", "quality_constraints_extract",
    "compatibility_constraints_extract", "user_reference_extract", "forbidden_change_extract",
    "must_preserve_extract", "completion_expectation_extract", "unspecified_field_detect",
    "contradiction_detect", "external_knowledge_need_detect", "source_requirement_detect",
    "existing_project_detect", "asset_requirement_detect", "tool_requirement_detect",
    "project_spec_finalize",
)

_DOMAIN = (
    "entity_identity_research", "canonical_description_research", "genre_research",
    "core_gameplay_loop_research", "character_system_research", "combat_system_research",
    "class_system_research", "skill_system_research", "item_system_research",
    "equipment_system_research", "monster_system_research", "boss_system_research",
    "progression_system_research", "map_environment_research", "ui_visual_language_research",
    "art_style_research", "color_palette_research", "iconic_content_research",
    "terminology_research", "feature_relationship_research", "canonical_scope_research",
    "reference_asset_research", "mechanic_invariant_research", "copyright_boundary_research",
    "source_conflict_research", "domain_brief_finalize",
)

_REQUIREMENT_SUBJECTS = (
    "gameplay", "content", "world", "entity", "item", "block", "weapon", "armor",
    "skill", "status", "progression", "gui", "hud", "particle", "animation", "sound",
    "texture", "model", "config", "network", "save", "compatibility", "localization",
    "acceptance", "command", "keybind", "recipe", "loot", "tag", "advancement",
    "worldgen", "dimension", "biome", "structure",
)
_REQUIREMENT_OPS = (
    "presence", "instance_list", "identity", "role", "state", "behavior", "dependencies",
    "inputs", "outputs", "failure_modes", "persistence", "network_authority",
    "client_presentation", "resource_obligations", "validation_contract", "acceptance_contract",
)

_REUSE_SUBJECTS = (
    "requirements", "analysis", "domain_brief", "feature_graph", "plan", "file_structure",
    "code", "assets", "image_prompts", "validators", "repairs", "tests",
    "registry_patterns", "network_patterns", "persistence_patterns", "gui_patterns",
)
_REUSE_OPS = (
    "search", "candidate_normalize", "version_compare", "platform_compare",
    "api_signature_compare", "provenance_verify", "exact_reuse_decide", "adaptation_decide",
    "discard_decide", "merge", "post_merge_conflict_validate",
)

_PLAN = (
    "requirement_set_freeze", "feature_list_expand", "subfeature_expand", "capability_expand",
    "artifact_census", "code_artifact_census", "json_artifact_census", "image_artifact_census",
    "model_artifact_census", "sound_artifact_census", "localization_artifact_census",
    "data_artifact_census", "dependency_extract", "prerequisite_extract",
    "parallelism_calculate", "implementation_order", "validation_order", "repair_relationships",
    "feature_dag_construct", "feature_to_epics", "epic_to_tasks", "task_to_microtasks",
    "microtask_atomicity", "input_availability", "precondition_generate",
    "postcondition_generate", "validator_assign", "repair_contract_assign", "reuse_binding",
    "side_ownership", "source_set_ownership", "resource_namespace_ownership",
    "task_anchor_reserve", "execution_graph_finalize",
)

_ARCH_SUBJECTS = (
    "project", "module", "source_set", "package", "registry", "event", "network",
    "persistence", "config", "datagen", "resource", "client", "server", "test", "build",
    "dependency",
)
_ARCH_OPS = (
    "inventory", "ownership", "boundary", "dependency", "naming", "path", "interface",
    "failure_contract", "version_binding", "compatibility_check",
)

_IMPLEMENTATION_SUBJECTS = (
    "mod_bootstrap", "registry", "block", "block_entity", "item", "tool", "weapon", "armor",
    "food", "component", "entity", "mob", "boss", "projectile", "entity_attribute",
    "entity_ai_goal", "entity_spawn", "entity_renderer", "entity_model", "menu", "screen",
    "hud", "widget", "tooltip", "event", "command", "keybind", "network_packet",
    "client_handler", "server_handler", "save_data", "config", "codec", "worldgen", "biome",
    "feature", "structure", "dimension", "particle", "sound", "animation", "recipe_logic",
    "loot_logic", "advancement_logic",
)
_IMPLEMENTATION_OPS = (
    "requirement_read", "target_api_read", "existing_pattern_read", "responsibility_freeze",
    "identity_freeze", "dependency_freeze", "imports", "type_shell", "state_fields",
    "constructor", "behavior_member", "registration", "event_binding", "resource_reference",
    "error_handling", "side_safety", "compile_validation", "resource_validation",
    "runtime_validation",
)

_ASSET_SUBJECTS = (
    "item_icon", "block_texture", "tileable_block", "entity_skin", "entity_sprite",
    "armor_texture", "weapon_texture", "projectile_texture", "particle_sprite", "gui_panel",
    "gui_icon", "button", "hud", "background", "logo", "status_effect_icon", "skill_icon",
    "biome_reference", "structure_reference", "model_concept",
)
_ASSET_OPS = (
    "requirement_presence", "subject", "reference", "style", "palette", "resolution", "alpha",
    "layout", "uv_constraints", "consistency", "positive_prompt", "negative_prompt",
    "generation_parameters", "generation", "dimension_validation", "alpha_validation",
    "visual_validation", "correction_prompt", "corrected_generation", "final_conversion",
    "path_assignment", "reference_connection",
)

_RESOURCE_SUBJECTS = (
    "blockstate", "block_model", "item_model", "client_item", "texture", "particle",
    "sound_definition", "lang", "atlas", "equipment", "shader", "font", "recipe", "loot",
    "tag", "advancement", "worldgen", "damage_type", "dimension", "biome", "structure",
    "configured_feature", "placed_feature",
)
_RESOURCE_OPS = (
    "presence", "schema", "namespace", "path", "content", "reference_bind", "datagen_bind",
    "parse_validate", "semantic_validate",
)

_LINK = (
    "code_to_registry", "registry_to_id", "id_to_lang", "id_to_model", "model_to_texture",
    "entity_to_renderer", "entity_to_model", "entity_to_texture", "item_to_recipe",
    "block_to_loot", "gui_to_menu", "menu_to_network", "keybind_to_action",
    "feature_to_worldgen", "sound_to_event", "datagen_to_build", "config_to_runtime",
    "packet_to_codec", "packet_to_handler", "save_to_codec", "structure_to_worldgen",
    "biome_to_feature", "advancement_to_trigger", "tag_to_consumer", "reference_graph_finalize",
)

_VALIDATE = (
    "file_exists", "path", "namespace", "json_schema", "resource_reference", "registry",
    "import", "api_version", "side_separation", "compilation", "datagen", "unit_test",
    "game_test", "game_start", "mod_load", "feature_load", "entity_spawn", "item_obtain",
    "recipe", "loot", "ui", "texture_missing", "model_missing", "visual",
    "network_direction", "network_codec", "persistence_reload", "config_reload",
    "worldgen_visibility", "localization", "performance", "regression", "dangling_reference",
    "unowned_artifact", "placeholder", "todo_fixme", "duplicate_registration",
    "client_server_leak", "acceptance_coverage", "artifact_manifest",
)

_REPAIR_SUBJECTS = (
    "compiler", "import", "mapping", "api_signature", "missing_registration",
    "duplicate_registration", "null", "client_server", "packet", "codec", "json",
    "resource_path", "missing_texture", "missing_model", "blockstate", "datagen",
    "runtime_crash", "load_order", "dependency", "visual_mismatch", "asset_prompt",
    "performance", "regression", "persistence", "worldgen", "localization",
)
_REPAIR_OPS = (
    "failure_receipt", "minimal_patch", "post_patch_static_validation",
    "post_patch_runtime_validation",
)

_PACKAGE = (
    "artifact_manifest_freeze", "license_inventory", "third_party_notice",
    "generated_asset_inventory", "source_inventory", "resource_inventory", "test_inventory",
    "build_output_inventory", "version_metadata", "loader_metadata", "mod_metadata",
    "changelog", "readme", "install_instructions", "clean_build", "final_smoke_test",
    "ledger_coverage", "failed_stage_check", "unexecuted_stage_check", "project_complete_decision",
)


def build_pipeline_definition() -> PipelineDefinition:
    stages = (
        *_fixed("CAP", "capture", _CAPTURE, model_slot_allowed=False),
        *_fixed("INT", "intent", _INTENT),
        *_fixed("DOM", "domain", _DOMAIN),
        *_matrix("REQ", "requirements", _REQUIREMENT_SUBJECTS, _REQUIREMENT_OPS),
        *_matrix("REUSE", "reuse", _REUSE_SUBJECTS, _REUSE_OPS),
        *_fixed("PLAN", "planning", _PLAN),
        *_matrix("ARCH", "architecture", _ARCH_SUBJECTS, _ARCH_OPS),
        *_matrix("IMPL", "implementation", _IMPLEMENTATION_SUBJECTS, _IMPLEMENTATION_OPS),
        *_matrix("ASSET", "assets", _ASSET_SUBJECTS, _ASSET_OPS),
        *_matrix("RES", "resources", _RESOURCE_SUBJECTS, _RESOURCE_OPS),
        *_fixed("LINK", "linking", _LINK, model_slot_allowed=False),
        *_fixed("VAL", "validation", _VALIDATE, model_slot_allowed=False),
        *_matrix("FIX", "repair", _REPAIR_SUBJECTS, _REPAIR_OPS),
        *_fixed("PKG", "packaging", _PACKAGE, model_slot_allowed=False),
    )
    definition = PipelineDefinition(schema=PIPELINE_SCHEMA, stages=tuple(stages))
    definition.validate()
    return definition


FIXED_PIPELINE = build_pipeline_definition()


def _coerce_execution(
    stage: StageDefinition,
    value: StageExecution | Mapping[str, Any] | str,
) -> StageExecution:
    if isinstance(value, StageExecution):
        if value.stage_id != stage.stage_id:
            raise ValueError(f"executor returned {value.stage_id!r} for {stage.stage_id!r}")
        return value
    if isinstance(value, str):
        return StageExecution(stage_id=stage.stage_id, status=value)
    if isinstance(value, Mapping):
        status = str(value.get("status") or "")
        artifacts_raw = value.get("artifacts") or ()
        if isinstance(artifacts_raw, str):
            artifacts = (artifacts_raw,)
        else:
            artifacts = tuple(str(item) for item in artifacts_raw)
        details = value.get("details")
        return StageExecution(
            stage_id=stage.stage_id,
            status=status,
            artifacts=artifacts,
            details=dict(details) if isinstance(details, Mapping) else {},
        )
    raise TypeError(f"invalid executor result for {stage.stage_id}: {type(value)!r}")


def run_exhaustive_pipeline(
    context: Mapping[str, Any],
    executor: StageExecutor,
    *,
    definition: PipelineDefinition = FIXED_PIPELINE,
) -> ExecutionLedger:
    """Invoke every fixed stage exactly once; never route around a stage."""

    ledger = ExecutionLedger(definition)
    for stage in definition.stages:
        try:
            execution = _coerce_execution(stage, executor(stage, context))
        except Exception as exc:
            execution = StageExecution(
                stage_id=stage.stage_id,
                status="fail",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
        ledger.record(execution)
    return ledger


__all__ = [
    "ExecutionLedger", "FIXED_PIPELINE", "LEDGER_SCHEMA", "PIPELINE_SCHEMA",
    "PipelineDefinition", "SUCCESS_STATUSES", "StageDefinition", "StageExecution",
    "TERMINAL_STATUSES", "build_pipeline_definition", "run_exhaustive_pipeline",
]
