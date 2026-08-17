from __future__ import annotations
from enum import Enum
from functools import wraps
from typing import Any

def _require_dict(value: Any, field: str, error_type: type[Exception]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f'{field} must be a JSON object.')
    return value

def _require_list(value: Any, field: str, error_type: type[Exception]) -> list[Any]:
    if not isinstance(value, list):
        raise error_type(f'{field} must be a JSON list.')
    return value

def _require_string(value: Any, field: str, error_type: type[Exception], *, empty: bool=False) -> str:
    if not isinstance(value, str) or (not empty and (not value.strip())):
        suffix = 'a string' if empty else 'a non-empty string'
        raise error_type(f'{field} must be {suffix}.')
    return value

def _string_list(value: Any, field: str, error_type: type[Exception], *, empty_items: bool=False) -> list[str]:
    values = _require_list(value, field, error_type)
    result: list[str] = []
    for index, item in enumerate(values):
        result.append(_require_string(item, f'{field}[{index}]', error_type, empty=empty_items))
    return result

def _json_native(value: Any) -> Any:
    """Normalize internal dataclass containers into JSON-native values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_native(item) for item in value]
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    return value

def install(spec_module: Any, complete_spec_module: Any) -> None:
    """Harden proposal serialization/deserialization at the JSON integrity boundary.

    Resume/approval data is an integrity boundary. Lists must not accept strings as
    character iterables, numeric ids must not become strings, and legacy
    ``ProductionModule`` normalization must not silently change persisted graph ids.
    ``Proposal.to_dict`` is normalized to real JSON arrays so data produced by the
    package is accepted by the same strict boundary it later consumes.
    """
    proposal_cls = spec_module.Proposal
    current_to_dict = proposal_cls.to_dict
    if not getattr(current_to_dict, '_mmm_json_native_serialization', False):

        @wraps(current_to_dict)
        def proposal_to_dict(self: Any) -> dict[str, Any]:
            value = _json_native(current_to_dict(self))
            if not isinstance(value, dict):
                raise spec_module.SpecValidationError('Proposal serialization must produce a JSON object.')
            return value
        proposal_to_dict._mmm_json_native_serialization = True
        proposal_to_dict.__wrapped__ = current_to_dict
        proposal_cls.to_dict = proposal_to_dict
    proposal_descriptor = proposal_cls.__dict__['from_dict']
    proposal_function = proposal_descriptor.__func__
    if not getattr(proposal_function, '_mmm_strict_deserialization', False):

        @classmethod
        @wraps(proposal_function)
        def proposal_from_dict(cls: Any, data: Any):
            error = spec_module.SpecValidationError
            raw = _require_dict(data, 'proposal', error)
            for field in ('schema_version', 'status', 'requested_prompt', 'approval_hash'):
                if field in raw:
                    _require_string(raw[field], field, error, empty=field == 'approval_hash')
            if 'proposal_version' in raw and (type(raw['proposal_version']) is not int or raw['proposal_version'] < 1):
                raise error('proposal_version must be a positive JSON integer.')
            for field in ('evidence_snapshot_hash', 'capability_manifest_hash', 'imported_source_snapshot_hash'):
                if field in raw:
                    _require_string(raw[field], field, error, empty=True)
            spec = _require_dict(raw.get('spec'), 'spec', error)
            for field in ('mod_id', 'mod_name', 'package_name', 'version', 'summary'):
                _require_string(spec.get(field), f'spec.{field}', error)
            platform = _require_dict(spec.get('platform'), 'spec.platform', error)
            for key, value in platform.items():
                _require_string(value, f'spec.platform.{key}', error)
            contents = _require_list(spec.get('contents'), 'spec.contents', error)
            for index, item in enumerate(contents):
                content = _require_dict(item, f'spec.contents[{index}]', error)
                for field in ('content_id', 'kind', 'display_name_en', 'display_name_ko'):
                    _require_string(content.get(field), f'spec.contents[{index}].{field}', error)
                if 'color' in content:
                    _require_string(content['color'], f'spec.contents[{index}].color', error)
                if 'recipe' in content and type(content['recipe']) is not bool:
                    raise error(f'spec.contents[{index}].recipe must be a JSON boolean.')
            boss = spec.get('boss')
            if boss is not None:
                boss = _require_dict(boss, 'spec.boss', error)
                for field in ('entity_id', 'display_name_en', 'display_name_ko', 'primary_color', 'secondary_color', 'model_kind'):
                    if field in boss:
                        _require_string(boss[field], f'spec.boss.{field}', error)
            for field in ('assumptions', 'exclusions', 'acceptance_tests', 'risk_approvals'):
                if field in raw:
                    _string_list(raw[field], field, error)
            deferred = _require_list(raw.get('deferred_requests'), 'deferred_requests', error)
            for index, item in enumerate(deferred):
                request = _require_dict(item, f'deferred_requests[{index}]', error)
                for field in ('capability', 'reason', 'suggested_phase'):
                    _require_string(request.get(field), f'deferred_requests[{index}].{field}', error)
            evidence = _require_list(raw.get('evidence_sources'), 'evidence_sources', error)
            for index, item in enumerate(evidence):
                source = _require_dict(item, f'evidence_sources[{index}]', error)
                for field, value in source.items():
                    _require_string(value, f'evidence_sources[{index}].{field}', error, empty=field == 'record_sha256')
            return proposal_function(cls, raw)
        proposal_from_dict.__func__._mmm_strict_deserialization = True
        proposal_cls.from_dict = proposal_from_dict
    complete_cls = complete_spec_module.CompleteProposal
    complete_descriptor = complete_cls.__dict__['from_dict']
    complete_function = complete_descriptor.__func__
    if getattr(complete_function, '_mmm_strict_deserialization', False):
        return

    @classmethod
    @wraps(complete_function)
    def complete_from_dict(cls: Any, data: Any):
        error = spec_module.SpecValidationError
        raw = _require_dict(data, 'complete proposal', error)
        for field in ('schema_version', 'status', 'requested_prompt', 'existing_input_sha256', 'approval_hash'):
            if field in raw:
                _require_string(raw[field], field, error, empty=field in {'existing_input_sha256', 'approval_hash'})
        if 'base_proposal' in raw:
            _require_dict(raw['base_proposal'], 'base_proposal', error)
        if 'game_design' in raw:
            _require_dict(raw['game_design'], 'game_design', error)
        if 'external_runtime_required' in raw and type(raw['external_runtime_required']) is not bool:
            raise error('external_runtime_required must be a JSON boolean.')
        if 'acceptance_tests' in raw:
            _string_list(raw['acceptance_tests'], 'acceptance_tests', error)
        modules = _require_list(raw.get('modules'), 'modules', error)
        for index, item in enumerate(modules):
            value = _require_dict(item, f'modules[{index}]', error)
            module_id = _require_string(value.get('module_id'), f'modules[{index}].module_id', error)
            if not complete_spec_module._ID.fullmatch(module_id):
                raise error(f'modules[{index}].module_id must already be lowercase snake_case.')
            _require_string(value.get('kind'), f'modules[{index}].kind', error)
            if 'config' in value:
                _require_dict(value['config'], f'modules[{index}].config', error)
            dependencies = _string_list(value.get('depends_on', []), f'modules[{index}].depends_on', error)
            invalid = [dep for dep in dependencies if not complete_spec_module._ID.fullmatch(dep)]
            if invalid:
                raise error(f'modules[{index}] has invalid dependency ids: {invalid[:4]}')
            _string_list(value.get('required_gates', []), f'modules[{index}].required_gates', error)
        assets = _require_list(raw.get('assets'), 'assets', error)
        for index, item in enumerate(assets):
            value = _require_dict(item, f'assets[{index}]', error)
            for field in ('asset_id', 'kind', 'prompt', 'target_path'):
                _require_string(value.get(field), f'assets[{index}].{field}', error)
            for field in ('width', 'height'):
                if field in value and type(value[field]) is not int:
                    raise error(f'assets[{index}].{field} must be a JSON integer.')
        return complete_function(cls, raw)
    complete_from_dict.__func__._mmm_strict_deserialization = True
    complete_cls.from_dict = complete_from_dict
__all__ = ['install']
