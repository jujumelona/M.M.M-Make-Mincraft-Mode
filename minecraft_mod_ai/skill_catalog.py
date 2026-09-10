from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CANONICAL_SKILLS = ('intake-mod-brief', 'research-minecraft-evidence', 'plan-game-design', 'freeze-approved-spec', 'inspect-existing-project', 'generate-fabric-core', 'generate-datagen', 'generate-worldgen', 'generate-geckolib-entity', 'generate-quest-progression', 'generate-gui-networking', 'generate-textures', 'model-with-blockbench', 'compile-and-repair', 'runtime-playtest', 'visual-review', 'release-security', 'execute-complete-production', 'patch-existing-project', 'publish-release', 'compile-massive-work-graph', 'gather-adaptive-minecraft-evidence', 'ground-production-with-live-evidence', 'resume-production-run', 'route-generic-game-research', 'select-compatible-ai-technique', 'converge-game-quality')
POLICY_NATIVE_SKILLS = frozenset({'compile-massive-work-graph', 'gather-adaptive-minecraft-evidence', 'resume-production-run', 'route-generic-game-research', 'select-compatible-ai-technique', 'execute-complete-production', 'converge-game-quality'})
REQUIRED_SECTIONS = ('activate_when:', 'inputs:', 'required_rag:', 'allowed_tools:', 'validators:', 'retry_policy:', 'approval_required:', 'forbidden_actions:', 'exit_conditions:')
REVIEWED_STAGES = frozenset({'frontdoor', 'planning', 'research', 'generation', 'quality', 'runtime', 'release', 'training'})
REVIEWED_TOOL_STAGES: dict[str, frozenset[str]] = {'discover_mmm_capabilities': REVIEWED_STAGES, 'plan_game': frozenset({'planning'}), 'plan_complete_game': frozenset({'frontdoor', 'planning'}), 'revise_plan': frozenset({'planning'}), 'revise_complete_plan': frozenset({'frontdoor', 'planning'}), 'approve_plan': frozenset({'planning', 'generation'}), 'approve_complete_plan': frozenset({'planning', 'generation'}), 'read_complete_plan_section': frozenset({'planning', 'generation'}), 'read_quality_contract': frozenset({'planning', 'generation', 'quality'}), 'quality_status': frozenset({'frontdoor', 'planning', 'generation', 'quality', 'release'}), 'discover_ecosystem_resources': frozenset({'frontdoor', 'planning', 'research', 'generation'}), 'inspect_modrinth_project': frozenset({'planning', 'research', 'generation'}), 'inspect_github_repository': frozenset({'planning', 'research', 'generation'}), 'inspect_huggingface_model': frozenset({'planning', 'research', 'generation'}), 'build_technology_radar': frozenset({'frontdoor', 'planning', 'research'}), 'assess_technology_compatibility': frozenset({'planning', 'research', 'generation'}), 'search_project_rag': frozenset({'frontdoor', 'planning', 'research', 'generation', 'quality'}), 'search_code_rag': frozenset({'research', 'generation', 'quality'}), 'read_reuse_source': frozenset({'generation'}), 'index_project_rag': frozenset({'research'}), 'inspect_existing_mod': frozenset({'frontdoor', 'planning', 'research', 'generation', 'quality'}), 'work_status': frozenset({'frontdoor', 'planning', 'generation', 'quality'}), 'work_tasks': frozenset({'frontdoor', 'planning', 'generation', 'quality'}), 'work_cancel_run': frozenset({'frontdoor', 'planning', 'generation'}), 'work_resume_run': frozenset({'frontdoor', 'planning', 'generation'}), 'execute_complete_project': frozenset({'generation'}), 'generate_fabric_project': frozenset({'generation'}), 'generate_assets': frozenset({'generation'}), 'generate_geckolib_entity': frozenset({'generation'}), 'generate_system_plugin': frozenset({'generation'}), 'apply_source_patch': frozenset({'generation'}), 'repair_project': frozenset({'quality'}), 'java_diagnostics': frozenset({'generation', 'quality'}), 'java_workspace_symbols': frozenset({'generation', 'quality'}), 'blockbench_list_tools': frozenset({'quality'}), 'blockbench_execute': frozenset({'quality'}), 'run_static_validation': frozenset({'quality'}), 'run_gradle_build': frozenset({'quality'}), 'run_gametest': frozenset({'quality'}), 'inspect_jar': frozenset({'quality', 'release'}), 'runtime_prepare_instance': frozenset({'runtime'}), 'runtime_start_server': frozenset({'runtime'}), 'runtime_start_client': frozenset({'runtime'}), 'runtime_send_command': frozenset({'runtime'}), 'runtime_logs': frozenset({'runtime'}), 'runtime_register_screenshot': frozenset({'runtime'}), 'runtime_status': frozenset({'runtime'}), 'runtime_stop': frozenset({'runtime'}), 'mineflayer_connect': frozenset({'runtime'}), 'mineflayer_status': frozenset({'runtime'}), 'mineflayer_walk_to': frozenset({'runtime'}), 'mineflayer_interact_block': frozenset({'runtime'}), 'mineflayer_inventory': frozenset({'runtime'}), 'mineflayer_disconnect': frozenset({'runtime'}), 'package_release': frozenset({'release'}), 'run_model_smoke': frozenset({'training'})}
MUTATING_TOOLS = frozenset({'approve_plan', 'approve_complete_plan', 'index_project_rag', 'work_cancel_run', 'work_resume_run', 'execute_complete_project', 'generate_fabric_project', 'generate_assets', 'generate_geckolib_entity', 'generate_system_plugin', 'apply_source_patch', 'repair_project', 'blockbench_execute', 'run_gradle_build', 'run_gametest', 'runtime_prepare_instance', 'runtime_start_server', 'runtime_start_client', 'runtime_send_command', 'runtime_register_screenshot', 'runtime_stop', 'mineflayer_connect', 'mineflayer_walk_to', 'mineflayer_interact_block', 'mineflayer_disconnect', 'package_release', 'run_model_smoke'})
_LEGACY_VALIDATOR_ALIASES = {
    'bounded duration, frequency, volume and file size': 'audio_bounds',
    'client playback and loop review': 'playback_review',
    'complete proposal hash and existing-input hash': 'proposal_identity',
    'exact archive and file hash preconditions': 'input_hashes',
    'exact SHA-256 patch preconditions': 'input_hashes',
    'existing functionality remains present': 'feature_preservation',
    'game version and loader are pinned': 'version_lock',
    'immutable approval and path containment': 'approval_and_path',
    'JAR bytes match the validated SHA-256': 'jar_hash',
    'Java diagnostics and structured resource validation where applicable': 'source_validation',
    'JDT, Gradle, GameTest and JAR gates': 'full_build_gates',
    'loader, version and mappings consistency': 'version_lock',
    'loader/version/mapping consistency': 'version_lock',
    'no advertised capability without its required build/runtime gate': 'capability_receipts',
    'no overwrite outside the approved project': 'path_containment',
    'no requested-functionality deletion': 'feature_preservation',
    'OGG file existence and deterministic registration': 'audio_binding',
    'path containment and no symlinks': 'path_containment',
    'request fidelity and immutable approval hash': 'approval_and_fidelity',
    'required Blockbench, runtime, Mineflayer and visual gates': 'external_quality_gates',
    'source containment and transactional writes': 'transactional_writes',
    'token is read only at upload time': 'secret_handling',
    'transaction rollback on failure': 'transaction_atomic',
    'transactional rollback on any failed operation': 'transaction_atomic',
    'upload endpoint is HTTPS and reviewed': 'reviewed_https',
    'ZIP bomb, path traversal, symlink and credential rejection': 'archive_safety',
    'fabric.mod.json id, version, environment, entrypoint and dependency fields match the approved PlatformLock and proposal': 'version_lock',
    'source-set and client/server entrypoint placement prevents dedicated-server loading of client-only classes': 'source_validation',
    'every registry identifier is valid, unique and referenced by the intended registration path': 'source_validation',
    'mixin config, access widener and resource references resolve when present and are absent when not requested': 'source_validation',
    'Java package names, imports and mapping symbols match the exact approved mappings and Java target': 'source_validation',
    'static validation and Java diagnostics pass for every changed core source/resource path': 'source_validation',
    'no generated core capability is advertised as built or runtime-tested until those downstream gates execute': 'capability_receipts',
}
REVIEWED_VALIDATORS = frozenset(set(_LEGACY_VALIDATOR_ALIASES.values()) | {'bounded_shards', 'checkpoint_integrity', 'complete_dependency_coverage', 'downstream_invalidation', 'durable_ledger', 'exact_version_evidence', 'immutable_model_revision', 'separate_license_closure', 'execution_boundary', 'data_flow_and_consent', 'measured_runtime_quality', 'deterministic_fallback', 'final_receipts', 'graph_acyclic', 'no_duplicate_run', 'retrieval_coverage', 'retrieval_not_authority', 'source_provenance', 'requirement_traceability', 'quality_convergence', 'evidence_freshness', 'no_self_certification'})
_FRONTMATTER_RE = re.compile('\\A---\\r?\\n(?P<frontmatter>.*?)\\r?\\n---\\r?\\n(?P<body>.*)\\Z', re.DOTALL)
_YAML_FENCE_RE = re.compile('```yaml\\s*\\r?\\n(?P<yaml>.*?)```', re.DOTALL)
_STAGE_PRIORITY = ('frontdoor', 'planning', 'research', 'generation', 'quality', 'runtime', 'release', 'training')


class SkillPolicyError(ValueError):
    """Raised when a Skill cannot compile into a fail-closed runtime policy."""


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    skill: str
    tool: str
    stage: str

    def to_dict(self) -> dict[str, Any]:
        return {'allowed': self.allowed, 'reason': self.reason, 'skill': self.skill, 'tool': self.tool, 'stage': self.stage}


@dataclass(frozen=True)
class RetryContract:
    max_attempts: int | None
    strategy: str
    stop_on_repeated_error_signature: bool
    require_fresh_evidence: bool

    def allows_retry(self, *, attempts_started: int, error_signature: str | None, prior_error_signatures: Iterable[str]=(), fresh_evidence: bool) -> bool:
        if attempts_started < 0:
            raise ValueError('attempts_started cannot be negative.')
        if self.max_attempts is not None and attempts_started >= self.max_attempts:
            return False
        if self.require_fresh_evidence and (not fresh_evidence):
            return False
        if self.stop_on_repeated_error_signature and error_signature and (error_signature in frozenset(prior_error_signatures)):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {'max_attempts': self.max_attempts, 'strategy': self.strategy, 'stop_on_repeated_error_signature': self.stop_on_repeated_error_signature, 'require_fresh_evidence': self.require_fresh_evidence}


@dataclass(frozen=True)
class ExitContract:
    success: tuple[str, ...]
    blocked: tuple[str, ...]
    failed: tuple[str, ...]

    def resolve(self, *, validators_passed: bool, receipts_complete: bool, unresolved_external: Iterable[str]=(), attempts_exhausted: bool=False, safety_violation: bool=False) -> str:
        if tuple(unresolved_external):
            return 'blocked'
        if attempts_exhausted or safety_violation:
            return 'failed'
        if validators_passed and receipts_complete:
            return 'success'
        return 'in_progress'

    def to_dict(self) -> dict[str, list[str]]:
        return {'success': list(self.success), 'blocked': list(self.blocked), 'failed': list(self.failed)}


@dataclass(frozen=True)
class SkillContract:
    name: str
    description: str
    activate_when: tuple[str, ...]
    stages: tuple[str, ...]
    required_rag: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    tool_routes: Mapping[str, str]
    validators: tuple[str, ...]
    retry: RetryContract
    approvals: Mapping[str, bool]
    forbidden_actions: tuple[str, ...]
    exit: ExitContract

    def authorize_tool(self, tool: str, stage: str, *, write_approved: bool=False, runtime_approved: bool=False, runtime_requested: bool=False) -> PolicyDecision:
        if tool not in self.allowed_tools:
            return PolicyDecision(False, 'tool_not_allowlisted', self.name, tool, stage)
        if stage not in self.stages:
            return PolicyDecision(False, 'stage_not_allowlisted', self.name, tool, stage)
        if stage not in REVIEWED_TOOL_STAGES.get(tool, frozenset()):
            return PolicyDecision(False, 'tool_not_exposed_in_stage', self.name, tool, stage)
        if tool in MUTATING_TOOLS and self.approvals.get('writes', False) and (not write_approved):
            return PolicyDecision(False, 'write_approval_required', self.name, tool, stage)
        if (stage == 'runtime' or runtime_requested) and self.approvals.get('runtime', False) and (not runtime_approved):
            return PolicyDecision(False, 'runtime_approval_required', self.name, tool, stage)
        return PolicyDecision(True, 'allowlisted', self.name, tool, stage)

    def failed_validators(self, context: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(validator for validator in self.validators if context.get(validator) is not True)

    def to_dict(self) -> dict[str, Any]:
        return {'name': self.name, 'description': self.description, 'activate_when': list(self.activate_when), 'stages': list(self.stages), 'required_rag': list(self.required_rag), 'allowed_tools': list(self.allowed_tools), 'tool_routes': dict(self.tool_routes), 'validators': list(self.validators), 'retry': self.retry.to_dict(), 'approvals': dict(self.approvals), 'forbidden_actions': list(self.forbidden_actions), 'exit': self.exit.to_dict()}


def compile_skill_contract(skill: str, root: str | Path | None=None) -> SkillContract:
    if skill not in CANONICAL_SKILLS:
        raise SkillPolicyError(f'Unknown canonical skill: {skill}')
    text = _skill_texts(root).get(skill)
    if text is None:
        raise SkillPolicyError(f'Missing Skill file: {skill}')
    frontmatter, policy = _parse_skill(text, skill)
    required_rag = _string_tuple(policy.get('required_rag'), 'required_rag', skill)
    tools = _string_tuple(policy.get('allowed_tools'), 'allowed_tools', skill)
    unknown_tools = sorted(set(tools) - set(REVIEWED_TOOL_STAGES))
    if unknown_tools:
        raise SkillPolicyError(f"{skill} contains unreviewed tools: {', '.join(unknown_tools)}")
    declared_stages = policy.get('stages')
    if declared_stages is None:
        stage_set = {stage for tool in tools for stage in REVIEWED_TOOL_STAGES[tool] if stage != 'frontdoor'}
        if not stage_set:
            stage_set = {stage for tool in tools for stage in REVIEWED_TOOL_STAGES[tool]}
    else:
        stage_set = set(_string_tuple(declared_stages, 'stages', skill))
    unknown_stages = sorted(stage_set - REVIEWED_STAGES)
    if unknown_stages:
        raise SkillPolicyError(f"{skill} contains unreviewed stages: {', '.join(unknown_stages)}")
    if not stage_set:
        raise SkillPolicyError(f'{skill} must allow at least one stage.')
    tool_routes: dict[str, str] = {}
    for tool in tools:
        candidates = REVIEWED_TOOL_STAGES[tool] & stage_set
        if not candidates:
            raise SkillPolicyError(f'{skill} allows {tool}, but none of its reviewed stages are enabled.')
    for tool in tools:
        candidates = REVIEWED_TOOL_STAGES[tool] & stage_set
        selected = next((stage for stage in _STAGE_PRIORITY if stage in candidates), None)
        if selected is None:
            raise SkillPolicyError(f'{skill} has no deterministic route for {tool}.')
        tool_routes[tool] = selected
    validators = tuple(_LEGACY_VALIDATOR_ALIASES.get(item, item) for item in _string_tuple(policy.get('validators'), 'validators', skill))
    unknown_validators = sorted(set(validators) - REVIEWED_VALIDATORS)
    if unknown_validators:
        raise SkillPolicyError(f"{skill} contains unreviewed validators: {', '.join(unknown_validators)}")
    retry = _retry_contract(policy.get('retry_policy'), skill)
    approvals = _approval_contract(policy.get('approval_required'), skill)
    exit_contract = _exit_contract(policy.get('exit_conditions'), skill)
    return SkillContract(name=frontmatter['name'], description=frontmatter['description'], activate_when=_string_tuple(policy.get('activate_when'), 'activate_when', skill), stages=tuple(stage for stage in _STAGE_PRIORITY if stage in stage_set), required_rag=required_rag, allowed_tools=tools, tool_routes=tool_routes, validators=validators, retry=retry, approvals=approvals, forbidden_actions=_string_tuple(policy.get('forbidden_actions'), 'forbidden_actions', skill), exit=exit_contract)


def compile_skill_catalog(root: str | Path | None=None) -> dict[str, SkillContract]:
    texts = _skill_texts(root)
    missing = [name for name in CANONICAL_SKILLS if name not in texts]
    if missing:
        raise SkillPolicyError(f'Missing canonical Skills: {missing}')
    return {name: compile_skill_contract(name, root=root) for name in CANONICAL_SKILLS}


def load_skill_catalog(root: str | Path | None=None) -> dict[str, SkillContract]:
    return compile_skill_catalog(root)


def validate_skill_catalog(root: str | Path | None=None) -> dict[str, Any]:
    findings: list[str] = []
    contracts: dict[str, dict[str, Any]] = {}
    try:
        compiled = compile_skill_catalog(root)
    except (SkillPolicyError, TypeError, ValueError, yaml.YAMLError) as exc:
        return {
            "schema_version": "mmm/skill-catalog-validation-v2",
            "skills": list(CANONICAL_SKILLS),
            "contracts": {},
            "findings": [f"invalid-catalog:{exc}"],
            "passed": False,
        }
    for name, contract in compiled.items():
        contracts[name] = contract.to_dict()
    return {
        "schema_version": "mmm/skill-catalog-validation-v2",
        "skills": list(CANONICAL_SKILLS),
        "contracts": contracts,
        "findings": findings,
        "passed": True,
    }


def _skill_texts(root: str | Path | None) -> dict[str, str]:
    if root is None:
        from .packaged_skill_runtime import packaged_skill_texts
        return packaged_skill_texts()
    base = Path(root).expanduser().resolve()
    return {name: (base / name / 'SKILL.md').read_text(encoding='utf-8') for name in CANONICAL_SKILLS if (base / name / 'SKILL.md').is_file()}


def _parse_skill(text: str, skill: str) -> tuple[dict[str, str], dict[str, Any]]:
    matched = _FRONTMATTER_RE.match(text)
    if not matched:
        raise SkillPolicyError(f'{skill} must start with YAML frontmatter.')
    frontmatter_raw = yaml.safe_load(matched.group('frontmatter'))
    if not isinstance(frontmatter_raw, dict):
        raise SkillPolicyError(f'{skill} frontmatter must be an object.')
    if set(frontmatter_raw) != {'name', 'description'}:
        raise SkillPolicyError(f'{skill} frontmatter must contain exactly name and description.')
    name = frontmatter_raw['name']
    description = frontmatter_raw['description']
    if name != skill or not isinstance(description, str) or not description.strip():
        raise SkillPolicyError(f'{skill} frontmatter is invalid.')
    fences = list(_YAML_FENCE_RE.finditer(matched.group('body')))
    if len(fences) != 1:
        raise SkillPolicyError(f'{skill} must contain exactly one YAML policy block.')
    policy = yaml.safe_load(fences[0].group('yaml'))
    if not isinstance(policy, dict):
        raise SkillPolicyError(f'{skill} policy block must be an object.')
    missing_sections = [section for section in REQUIRED_SECTIONS if section[:-1] not in policy]
    if missing_sections:
        raise SkillPolicyError(f'{skill} is missing required policy sections: {missing_sections}')
    return {'name': name, 'description': description}, policy


def _string_tuple(value: Any, field: str, skill: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise SkillPolicyError(f'{skill} {field} must be a non-empty string list.')
    return tuple(value)


def _retry_contract(value: Any, skill: str) -> RetryContract:
    if not isinstance(value, dict):
        raise SkillPolicyError(f'{skill} retry_policy must be an object.')
    attempts = value.get('max_attempts')
    if attempts is not None and (type(attempts) is not int or attempts < 1):
        raise SkillPolicyError(f'{skill} retry_policy.max_attempts must be null or a positive integer.')
    strategy = value.get('strategy')
    if not isinstance(strategy, str) or not strategy.strip():
        raise SkillPolicyError(f'{skill} retry_policy.strategy must be a string.')
    stop = value.get('stop_on_repeated_error_signature')
    fresh = value.get('require_fresh_evidence')
    if type(stop) is not bool or type(fresh) is not bool:
        raise SkillPolicyError(f'{skill} retry policy booleans are invalid.')
    return RetryContract(attempts, strategy, stop, fresh)


def _approval_contract(value: Any, skill: str) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise SkillPolicyError(f'{skill} approval_required must be an object.')
    expected = {'writes', 'runtime', 'release'}
    if set(value) != expected or any(type(value[key]) is not bool for key in expected):
        raise SkillPolicyError(f'{skill} approval_required must contain boolean writes/runtime/release.')
    return {key: bool(value[key]) for key in sorted(expected)}


def _exit_contract(value: Any, skill: str) -> ExitContract:
    if not isinstance(value, dict) or set(value) != {'success', 'blocked', 'failed'}:
        raise SkillPolicyError(f'{skill} exit_conditions must contain success/blocked/failed.')
    return ExitContract(success=_string_tuple(value['success'], 'exit_conditions.success', skill), blocked=_string_tuple(value['blocked'], 'exit_conditions.blocked', skill), failed=_string_tuple(value['failed'], 'exit_conditions.failed', skill))
