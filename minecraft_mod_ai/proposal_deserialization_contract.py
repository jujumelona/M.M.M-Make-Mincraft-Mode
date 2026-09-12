from __future__ import annotations

from dataclasses import fields
from enum import Enum
from functools import wraps
from typing import Any

from .spec import PlatformLock
from .target_contract import uses_native_names


def _require_dict(value: Any, field: str, error_type: type[Exception]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise error_type(f'{field} must be a JSON object.')
    return value


def _require_list(value: Any, field: str, error_type: type[Exception]) -> list[Any]:
    if not isinstance(value, list):
        raise error_type(f'{field} must be a JSON list.')
    return value


def _require_string(
    value: Any,
    field: str,
    error_type: type[Exception],
    *,
    empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not empty and (not value.strip())):
        suffix = 'a string' if empty else 'a non-empty string'
        raise error_type(f'{field} must be {suffix}.')
    return value


def _string_list(
    value: Any,
    field: str,
    error_type: type[Exception],
    *,
    empty_items: bool = False,
) -> list[str]:
    values = _require_list(value, field, error_type)
    result: list[str] = []
    for index, item in enumerate(values):
        result.append(
            _require_string(
                item,
                f'{field}[{index}]',
                error_type,
                empty=empty_items,
            )
        )
    return result


def _validate_platform_json(platform: dict[str, Any], error_type: type[Exception]) -> None:
    typed_fields = {'resource_pack_format', 'deterministic_module_kinds'}
    mapping_fields = {'mappings_kind', 'mappings_version', 'yarn_mappings'}
    optional_string_fields = {'host_facts_json'}
    canonical_fields = {item.name for item in fields(PlatformLock)}
    string_fields = canonical_fields - typed_fields
    unknown = sorted(set(platform) - canonical_fields)
    if unknown:
        raise error_type(f'spec.platform contains unsupported fields: {unknown[:8]}')

    native_names = False
    version = platform.get('minecraft_version')
    if isinstance(version, str) and version.strip():
        try:
            native_names = uses_native_names(version)
        except ValueError as exc:
            raise error_type(str(exc)) from exc

    for key in string_fields:
        if key in platform:
            _require_string(
                platform[key],
                f'spec.platform.{key}',
                error_type,
                empty=(
                    key in optional_string_fields
                    or (native_names and key in mapping_fields)
                ),
            )
    if 'resource_pack_format' in platform:
        value = platform['resource_pack_format']
        if type(value) is not int or value <= 0:
            raise error_type('spec.platform.resource_pack_format must be a positive JSON integer.')
    if 'deterministic_module_kinds' in platform:
        _string_list(
            platform['deterministic_module_kinds'],
            'spec.platform.deterministic_module_kinds',
            error_type,
        )


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


def _status_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _install_json_native_serializer(proposal_cls: Any) -> None:
    """Make every public proposal ``to_dict`` result valid strict JSON input."""

    current_to_dict = proposal_cls.to_dict
    if getattr(current_to_dict, '_mmm_json_native_serializer', False):
        return

    @wraps(current_to_dict)
    def to_dict(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        value = current_to_dict(self, *args, **kwargs)
        normalized = _json_native(value)
        if not isinstance(normalized, dict):
            raise TypeError('Proposal serializer must return a JSON object.')
        return normalized

    to_dict._mmm_json_native_serializer = True
    to_dict.__wrapped__ = current_to_dict
    proposal_cls.to_dict = to_dict


def _install_approval_authority(
    proposal_cls: Any,
    *,
    approved_status: str,
    error_type: type[Exception],
    label: str,
) -> None:
    current_validate = proposal_cls.validate
    if not getattr(current_validate, '_mmm_approval_receipt_required', False):

        @wraps(current_validate)
        def validate(self: Any, *args: Any, **kwargs: Any):
            if (
                _status_value(getattr(self, 'status', '')) == approved_status
                and not str(getattr(self, 'approval_hash', '') or '').strip()
            ):
                raise error_type(
                    f'{label} approved state requires its approval_hash integrity receipt.'
                )
            return current_validate(self, *args, **kwargs)

        validate._mmm_approval_receipt_required = True
        validate.__wrapped__ = current_validate
        proposal_cls.validate = validate

    current_approve = proposal_cls.approve
    if getattr(current_approve, '_mmm_approval_receipt_bound', False):
        return

    @wraps(current_approve)
    def approve(self: Any, supplied_hash: str, *args: Any, **kwargs: Any):
        expected = self.calculate_hash()
        approved = current_approve(self, supplied_hash, *args, **kwargs)
        if str(getattr(approved, 'approval_hash', '') or '') != expected:
            approved = type(approved)(
                **{**approved.__dict__, 'approval_hash': expected}
            )
        return approved

    approve._mmm_approval_receipt_bound = True
    approve.__wrapped__ = current_approve
    proposal_cls.approve = approve


def install_proposal_deserialization_contracts(
    *,
    proposal_cls: Any,
    proposal_status_cls: Any,
    spec_validation_error: type[Exception],
    content_spec_cls: Any,
    content_kind_cls: Any,
    boss_spec_cls: Any,
    mod_spec_cls: Any,
    deferred_request_cls: Any,
    evidence_source_cls: Any,
    capability_manifest_hash: Any,
    evidence_snapshot_hash: Any,
    json_bool: Any,
    complete_proposal_cls: Any,
    complete_proposal_status_cls: Any,
    production_module_cls: Any,
    asset_request_cls: Any,
) -> None:
    """Install strict JSON deserializers and approval-state receipt guards.

    This stays centralized so Proposal and CompleteProposal share one untrusted JSON
    boundary instead of maintaining subtly different handwritten parsers.
    """

    def proposal_from_dict(cls: Any, data: dict[str, Any]) -> Any:
        error = spec_validation_error
        raw = _require_dict(data, 'proposal', error)
        unknown = sorted(set(raw) - cls._TOP_LEVEL_KEYS)
        missing = sorted(cls._TOP_LEVEL_KEYS - cls._BACKWARD_COMPATIBLE_KEYS - set(raw))
        if unknown:
            raise error(f'Unknown proposal fields: {unknown}')
        if missing:
            raise error(f'Missing proposal fields: {missing}')
        schema_version = _require_string(raw['schema_version'], 'schema_version', error)
        proposal_version = raw['proposal_version']
        if type(proposal_version) is not int:
            raise error('proposal_version must be a JSON integer.')
        requested_prompt = _require_string(raw['requested_prompt'], 'requested_prompt', error)
        status_raw = _require_string(raw['status'], 'status', error)
        try:
            status = proposal_status_cls(status_raw)
        except ValueError as exc:
            raise error(f'Unsupported proposal status: {status_raw!r}') from exc

        spec_data = dict(_require_dict(raw['spec'], 'spec', error))
        if 'platform' not in spec_data:
            raise error('spec.platform is required.')
        platform_data = dict(_require_dict(spec_data.pop('platform'), 'spec.platform', error))
        _validate_platform_json(platform_data, error)
        try:
            platform = PlatformLock(**platform_data)
        except TypeError as exc:
            raise error('spec.platform is malformed.') from exc

        content_data = _require_list(spec_data.pop('contents', []), 'spec.contents', error)
        boss_data = spec_data.pop('boss', None)
        arena_data = spec_data.pop('arena', None)
        if arena_data is not None:
            raise error('spec.arena is not supported.')
        try:
            contents = tuple(
                content_spec_cls(
                    content_id=_require_string(item['content_id'], f'contents[{index}].content_id', error),
                    kind=content_kind_cls(item['kind']),
                    display_name_en=_require_string(item['display_name_en'], f'contents[{index}].display_name_en', error),
                    display_name_ko=_require_string(item['display_name_ko'], f'contents[{index}].display_name_ko', error),
                    color=item.get('color', '#74c7ec'),
                    recipe=json_bool(item.get('recipe', True), 'contents[].recipe'),
                )
                for index, item in enumerate(content_data)
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, error):
                raise
            raise error(f'Invalid content specification: {exc}') from exc
        try:
            spec = mod_spec_cls(
                contents=contents,
                boss=boss_spec_cls(**boss_data) if boss_data else None,
                platform=platform,
                **spec_data,
            )
        except TypeError as exc:
            raise error(f'Invalid mod specification: {exc}') from exc

        evidence_values = _require_list(raw['evidence_sources'], 'evidence_sources', error)
        try:
            evidence_sources = tuple(evidence_source_cls(**item) for item in evidence_values)
        except TypeError as exc:
            raise error(f'Invalid evidence source: {exc}') from exc
        proposal = cls(
            schema_version=schema_version,
            proposal_version=proposal_version,
            status=status,
            requested_prompt=requested_prompt,
            spec=spec,
            assumptions=tuple(_string_list(raw['assumptions'], 'assumptions', error)),
            exclusions=tuple(_string_list(raw['exclusions'], 'exclusions', error)),
            deferred_requests=tuple(
                deferred_request_cls(**item)
                for item in _require_list(raw['deferred_requests'], 'deferred_requests', error)
            ),
            acceptance_tests=tuple(_string_list(raw['acceptance_tests'], 'acceptance_tests', error)),
            evidence_sources=evidence_sources,
            evidence_snapshot_hash=raw.get(
                'evidence_snapshot_hash', evidence_snapshot_hash(evidence_sources)
            ),
            capability_manifest_hash=raw.get(
                'capability_manifest_hash', capability_manifest_hash()
            ),
            imported_source_snapshot_hash=raw.get('imported_source_snapshot_hash', ''),
            risk_approvals=tuple(_string_list(raw.get('risk_approvals', []), 'risk_approvals', error)),
            approval_hash=raw.get('approval_hash', ''),
        )
        proposal.validate()
        return proposal

    def complete_from_dict(cls: Any, data: dict[str, Any]) -> Any:
        error = spec_validation_error
        raw = _require_dict(data, 'complete_proposal', error)
        base_proposal = proposal_cls.from_dict(
            dict(_require_dict(raw['base_proposal'], 'base_proposal', error))
        )
        try:
            modules = tuple(
                production_module_cls(
                    module_id=item['module_id'],
                    kind=item['kind'],
                    config=dict(item.get('config', {})),
                    depends_on=tuple(item.get('depends_on', ())),
                    required_gates=tuple(item.get('required_gates', ())),
                )
                for item in _require_list(raw['modules'], 'modules', error)
            )
            assets = tuple(
                asset_request_cls(**item)
                for item in _require_list(raw.get('assets', []), 'assets', error)
            )
            complete = cls(
                schema_version=_require_string(raw['schema_version'], 'schema_version', error),
                proposal_version=raw['proposal_version'],
                status=complete_proposal_status_cls(
                    _require_string(raw['status'], 'status', error)
                ),
                requested_prompt=_require_string(raw['requested_prompt'], 'requested_prompt', error),
                base_proposal=base_proposal,
                game_design=dict(_require_dict(raw['game_design'], 'game_design', error)),
                modules=modules,
                assets=assets,
                acceptance_tests=tuple(
                    _string_list(raw['acceptance_tests'], 'acceptance_tests', error)
                ),
                external_runtime_required=raw.get('external_runtime_required', True),
                existing_input_sha256=raw.get('existing_input_sha256', ''),
                approval_hash=raw.get('approval_hash', ''),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, error):
                raise
            raise error(f'Invalid complete proposal: {exc}') from exc
        complete.validate()
        return complete

    proposal_cls.from_dict = classmethod(proposal_from_dict)
    complete_proposal_cls.from_dict = classmethod(complete_from_dict)
    _install_json_native_serializer(proposal_cls)
    _install_json_native_serializer(complete_proposal_cls)
    _install_approval_authority(
        proposal_cls,
        approved_status=proposal_status_cls.APPROVED.value,
        error_type=spec_validation_error,
        label='Proposal',
    )
    _install_approval_authority(
        complete_proposal_cls,
        approved_status=complete_proposal_status_cls.APPROVED.value,
        error_type=spec_validation_error,
        label='Complete proposal',
    )


def install(spec_module: Any, complete_spec_module: Any) -> None:
    """Bind the centralized deserialization contract into runtime modules."""
    from .capabilities import capability_manifest_hash
    from .knowledge import evidence_snapshot_hash

    install_proposal_deserialization_contracts(
        proposal_cls=spec_module.Proposal,
        proposal_status_cls=spec_module.ProposalStatus,
        spec_validation_error=spec_module.SpecValidationError,
        content_spec_cls=spec_module.ContentSpec,
        content_kind_cls=spec_module.ContentKind,
        boss_spec_cls=spec_module.BossSpec,
        mod_spec_cls=spec_module.ModSpec,
        deferred_request_cls=spec_module.DeferredRequest,
        evidence_source_cls=spec_module.EvidenceSource,
        capability_manifest_hash=capability_manifest_hash,
        evidence_snapshot_hash=evidence_snapshot_hash,
        json_bool=spec_module._json_bool,
        complete_proposal_cls=complete_spec_module.CompleteProposal,
        complete_proposal_status_cls=complete_spec_module.CompleteProposalStatus,
        production_module_cls=complete_spec_module.ProductionModule,
        asset_request_cls=complete_spec_module.AssetRequest,
    )


__all__ = ['install', 'install_proposal_deserialization_contracts']
