from __future__ import annotations

import json
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
_LEGACY_VALIDATOR_ALIASES = {'bounded duration, frequency, volume and file size': 'audio_bounds', 'client playback and loop review': 'playback_review', 'complete proposal hash and existing-input hash': 'proposal_identity', 'exact archive and file hash preconditions': 'input_hashes', 'exact SHA-256 patch preconditions': 'input_hashes', 'existing functionality remains present': 'feature_preservation', 'game version and loader are pinned': 'version_lock', 'immutable approval and path containment': 'approval_and_path', 'JAR bytes match the validated SHA-256': 'jar_hash', 'Java diagnostics and structured resource validation where applicable': 'source_validation', 'JDT, Gradle, GameTest and JAR gates': 'full_build_gates', 'loader, version and mappings consistency': 'version_lock', 'loader/version/mapping consistency': 'version_lock', 'no advertised capability without its required build/runtime gate': 'capability_receipts', 'no overwrite outside the approved project': 'path_containment', 'no requested-functionality deletion': 'feature_preservation', 'OGG file existence and deterministic registration': 'audio_binding', 'path containment and no symlinks': 'path_containment', 'request fidelity and immutable approval hash': 'approval_and_fidelity', 'required Blockbench, runtime, Mineflayer and visual gates': 'external_quality_gates', 'source containment and transactional writes': 'transactional_writes', 'token is read only at upload time': 'secret_handling', 'transaction rollback on failure': 'transaction_atomic', 'transactional rollback on any failed operation': 'transaction_atomic', 'upload endpoint is HTTPS and reviewed': 'reviewed_https', 'ZIP bomb, path traversal, symlink and credential rejection': 'archive_safety'}
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
        tool_routes[tool] = next(stage for stage in _STAGE_PRIORITY if stage in candidates)
    validator_values = _string_tuple(policy.get('validators'), 'validators', skill)
    validator_ids: list[str] = []
    for value in validator_values:
        validator_id = value if value in REVIEWED_VALIDATORS else _LEGACY_VALIDATOR_ALIASES.get(value)
        if validator_id is None:
            raise SkillPolicyError(f'{skill} contains unreviewed validator: {value}')
        if validator_id not in validator_ids:
            validator_ids.append(validator_id)
    retry_raw = _mapping(policy.get('retry_policy'), 'retry_policy', skill)
    max_attempts = retry_raw.get('max_attempts')
    if max_attempts is not None:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool):
            raise SkillPolicyError(f'{skill} retry max_attempts must be null or an integer.')
        if not 1 <= max_attempts <= 10:
            raise SkillPolicyError(f'{skill} retry max_attempts must be between 1 and 10 when set.')
    strategy = retry_raw.get('strategy')
    if not isinstance(strategy, str) or not strategy.strip():
        raise SkillPolicyError(f'{skill} retry strategy must be non-empty.')
    retry = RetryContract(max_attempts=max_attempts, strategy=strategy.strip(), stop_on_repeated_error_signature=_bool_field(retry_raw, 'stop_on_repeated_error_signature', skill), require_fresh_evidence=bool(retry_raw.get('require_fresh_evidence', 'new ' in strategy.casefold() or 'diagnostic' in strategy.casefold())))
    approvals_raw = _mapping(policy.get('approval_required'), 'approval_required', skill)
    approvals = {key: _bool_field(approvals_raw, key, skill) for key in ('writes', 'runtime', 'read_only_research')}
    exit_raw = _mapping(policy.get('exit_conditions'), 'exit_conditions', skill)
    exit_contract = ExitContract(success=_string_tuple(exit_raw.get('success'), 'exit.success', skill), blocked=_string_tuple(exit_raw.get('blocked'), 'exit.blocked', skill), failed=_string_tuple(exit_raw.get('failed'), 'exit.failed', skill))
    return SkillContract(name=skill, description=str(frontmatter['description']).strip(), activate_when=_string_tuple(policy.get('activate_when'), 'activate_when', skill), stages=tuple(stage for stage in _STAGE_PRIORITY if stage in stage_set), required_rag=required_rag, allowed_tools=tools, tool_routes=tool_routes, validators=tuple(validator_ids), retry=retry, approvals=approvals, forbidden_actions=_string_tuple(policy.get('forbidden_actions'), 'forbidden_actions', skill), exit=exit_contract)

def compile_skill_catalog(root: str | Path | None=None) -> dict[str, SkillContract]:
    return {skill: compile_skill_contract(skill, root) for skill in CANONICAL_SKILLS}

def validate_skill_catalog(root: str | Path | None=None) -> dict[str, Any]:
    texts = _skill_texts(root)
    findings: list[str] = []
    contracts: dict[str, dict[str, Any]] = {}
    for skill in CANONICAL_SKILLS:
        text = texts.get(skill)
        if text is None:
            findings.append(f'missing:{skill}')
            continue
        if '[TODO' in text or 'TODO:' in text:
            findings.append(f'todo:{skill}')
        for section in REQUIRED_SECTIONS:
            if section not in text:
                findings.append(f'missing-section:{skill}:{section}')
        try:
            frontmatter, _ = _parse_skill(text, skill)
            if skill in POLICY_NATIVE_SKILLS and set(frontmatter) != {'name', 'description'}:
                findings.append(f'frontmatter-fields:{skill}')
            contracts[skill] = compile_skill_contract(skill, root).to_dict()
        except (SkillPolicyError, TypeError, ValueError, yaml.YAMLError) as exc:
            findings.append(f'invalid-contract:{skill}:{exc}')
    return {'schema_version': 'mmm/skill-catalog-validation-v2', 'skills': list(CANONICAL_SKILLS), 'contracts': contracts, 'findings': findings, 'passed': not findings}

def _parse_skill(text: str, expected_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillPolicyError(f'{expected_name} has invalid frontmatter boundaries.')
    frontmatter = yaml.safe_load(match.group('frontmatter'))
    if not isinstance(frontmatter, dict):
        raise SkillPolicyError(f'{expected_name} frontmatter must be a mapping.')
    if frontmatter.get('name') != expected_name:
        raise SkillPolicyError(f'{expected_name} frontmatter name does not match.')
    description = frontmatter.get('description')
    if not isinstance(description, str) or not description.strip():
        raise SkillPolicyError(f'{expected_name} description must be non-empty.')
    body = match.group('body')
    policy: dict[str, Any] | None = None
    for fenced in _YAML_FENCE_RE.finditer(body):
        candidate = yaml.safe_load(fenced.group('yaml'))
        if isinstance(candidate, dict) and candidate.get('schema_version') == 'mmm/skill-policy-v1':
            policy = candidate
            break
    if policy is None:
        candidate = yaml.safe_load(body)
        if isinstance(candidate, dict):
            policy = candidate
    if policy is None:
        raise SkillPolicyError(f'{expected_name} has no compilable runtime policy.')
    return (frontmatter, policy)

def _string_tuple(value: Any, field: str, skill: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SkillPolicyError(f'{skill} {field} must be a non-empty list.')
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SkillPolicyError(f'{skill} {field} contains an invalid item.')
        result.append(item.strip())
    return tuple(result)

def _mapping(value: Any, field: str, skill: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SkillPolicyError(f'{skill} {field} must be a mapping.')
    return value

def _bool_field(value: Mapping[str, Any], field: str, skill: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise SkillPolicyError(f'{skill} {field} must be boolean.')
    return result

def _skill_texts(root: str | Path | None) -> dict[str, str]:
    """Load packaged Skills, then overlay Skill files from a source checkout."""
    texts: dict[str, str] = {}
    packaged = Path(__file__).resolve().parent / 'packaged_skills.json'
    if root is None and packaged.is_file():
        raw = json.loads(packaged.read_text(encoding='utf-8'))
        skills = raw.get('skills', {})
        texts.update({str(name): str(text) for name, text in skills.items() if isinstance(name, str) and isinstance(text, str)})
    base = Path(root).expanduser().resolve() if root is not None else Path(__file__).resolve().parents[1] / 'skills'
    if base.is_dir():
        for skill in CANONICAL_SKILLS:
            path = base / skill / 'SKILL.md'
            if path.is_file():
                texts[skill] = path.read_text(encoding='utf-8')
    return texts
