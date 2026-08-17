from __future__ import annotations
import re
from functools import wraps
from typing import Any

def _nonempty_string(value: Any, field: str, error_type: type[Exception]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f'{field} must be a non-empty string.')
    return value.strip()

def _string_sequence(value: Any, field: str, error_type: type[Exception]) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise error_type(f'{field} must be a list of strings.')
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise error_type(f'{field}[{index}] must be a non-empty string.')
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise error_type(f'{field} must not contain duplicates.')
    return tuple(result)

def install(module: Any) -> None:
    """Replace permissive production decoders with typed, fail-closed parsers.

    Planner pages are already page-locally repairable. Silently coercing malformed
    fields therefore makes correctness worse: invalid dependencies disappear,
    ``bool('false')`` becomes True, missing deliverables are fabricated, and fuzzy
    dependency repair can connect the wrong production batch. Reject malformed output
    at the parse boundary so the existing repair path can correct the exact field.
    """
    current_batch = module._production_batch
    if not getattr(current_batch, '_mmm_fail_closed_parser', False):

        @wraps(current_batch)
        def production_batch(value: Any):
            expected = {'batch_id', 'scope', 'depends_on_batches', 'deliverables', 'exports'}
            if not isinstance(value, dict) or set(value) != expected:
                raise module.SpecValidationError('Production batch descriptor fields are invalid.')
            batch_id = _nonempty_string(value['batch_id'], 'production batch.batch_id', module.SpecValidationError)
            if not module._BATCH_ID.fullmatch(batch_id):
                raise module.SpecValidationError(f'Invalid production batch id: {batch_id!r}')
            scope = _nonempty_string(value['scope'], f'production batch {batch_id}.scope', module.SpecValidationError)
            dependencies = _string_sequence(value['depends_on_batches'], f'production batch {batch_id}.depends_on_batches', module.SpecValidationError)
            deliverables = _string_sequence(value['deliverables'], f'production batch {batch_id}.deliverables', module.SpecValidationError)
            exports = _string_sequence(value['exports'], f'production batch {batch_id}.exports', module.SpecValidationError)
            if not deliverables:
                raise module.SpecValidationError(f'Production batch {batch_id} must declare at least one deliverable.')
            if batch_id in dependencies:
                raise module.SpecValidationError(f'Production batch {batch_id} may not depend on itself.')
            clean_exports: list[str] = []
            for item in exports:
                if module._BATCH_ID.fullmatch(item):
                    clean_exports.append(item)
                else:
                    sanitized = re.sub('[^a-z0-9_\\-]+', '_', item.lower()).strip('_')
                    if not sanitized or not sanitized[0].isalpha():
                        sanitized = f'exp_{sanitized}'
                    sanitized = sanitized[:63]
                    if module._BATCH_ID.fullmatch(sanitized):
                        clean_exports.append(sanitized)
                    else:
                        clean_exports.append(f'{batch_id}_export')
            return module._ProductionBatch(batch_id=batch_id, scope=scope, depends_on_batches=dependencies, deliverables=deliverables, exports=tuple(dict.fromkeys(clean_exports)))
        production_batch._mmm_fail_closed_parser = True
        module._production_batch = production_batch
    current_topological = module._topological_production_batches
    if not getattr(current_topological, '_mmm_fail_closed_graph', False):

        @wraps(current_topological)
        def topological_production_batches(batches: tuple[Any, ...]):
            ids = [batch.batch_id for batch in batches]
            if len(ids) != len(set(ids)):
                raise module.SpecValidationError('Production outline contains duplicate batch ids.')
            by_id = {batch.batch_id: batch for batch in batches}
            for batch in batches:
                missing = [dependency for dependency in batch.depends_on_batches if dependency not in by_id]
                if missing:
                    raise module.SpecValidationError(f'Production batch {batch.batch_id} references unknown dependencies: ' + ', '.join(missing[:4]))
                if batch.batch_id in batch.depends_on_batches:
                    raise module.SpecValidationError(f'Production batch {batch.batch_id} may not depend on itself.')
            outgoing: dict[str, list[str]] = {batch_id: [] for batch_id in by_id}
            indegree: dict[str, int] = {}
            for batch in batches:
                indegree[batch.batch_id] = len(batch.depends_on_batches)
                for dependency in batch.depends_on_batches:
                    outgoing[dependency].append(batch.batch_id)
            ready = [batch_id for batch_id, degree in indegree.items() if degree == 0]
            module.heapq.heapify(ready)
            ordered: list[Any] = []
            while ready:
                batch_id = module.heapq.heappop(ready)
                ordered.append(by_id[batch_id])
                for dependent in outgoing[batch_id]:
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        module.heapq.heappush(ready, dependent)
            if len(ordered) != len(batches):
                raise module.SpecValidationError('Production batch dependency cycle detected.')
            return tuple(ordered)
        topological_production_batches._mmm_fail_closed_graph = True
        module._topological_production_batches = topological_production_batches
    current_module = module._module
    if not getattr(current_module, '_mmm_fail_closed_parser', False):

        @wraps(current_module)
        def production_module(value: Any):
            if not isinstance(value, dict):
                raise module.SpecValidationError('Every production module must be an object.')
            module_id = value.get('module_id') or value.get('id') or value.get('name')
            kind = value.get('kind') or value.get('type')
            module_id = _nonempty_string(module_id, 'production module.module_id', module.SpecValidationError)
            kind = _nonempty_string(kind, f'production module {module_id}.kind', module.SpecValidationError)
            config = value.get('config')
            if not isinstance(config, dict):
                raise module.SpecValidationError(f'Production module {module_id}.config must be an object.')
            depends_on = _string_sequence(value.get('depends_on', []), f'production module {module_id}.depends_on', module.SpecValidationError)
            required_gates = _string_sequence(value.get('required_gates', []), f'production module {module_id}.required_gates', module.SpecValidationError)
            parsed = module.ProductionModule(module_id=module_id, kind=kind, config=dict(config), depends_on=depends_on, required_gates=required_gates)
            parsed.validate()
            return parsed
        production_module._mmm_fail_closed_parser = True
        module._module = production_module
    current_asset = module._asset
    if not getattr(current_asset, '_mmm_fail_closed_parser', False):

        @wraps(current_asset)
        def asset(value: Any):
            if not isinstance(value, dict):
                raise module.SpecValidationError('Every asset request must be an object.')
            asset_id = _nonempty_string(value.get('asset_id') or value.get('id'), 'asset.asset_id', module.SpecValidationError)
            kind = _nonempty_string(value.get('kind'), f'asset {asset_id}.kind', module.SpecValidationError)
            prompt = _nonempty_string(value.get('prompt') or value.get('description'), f'asset {asset_id}.prompt', module.SpecValidationError)
            target_path = _nonempty_string(value.get('target_path'), f'asset {asset_id}.target_path', module.SpecValidationError)
            width = value.get('width', 16)
            height = value.get('height', 16)
            if type(width) is not int or type(height) is not int:
                raise module.SpecValidationError(f'Asset {asset_id} width/height must be integers.')
            parsed = module.AssetRequest(asset_id=asset_id, kind=kind, prompt=prompt, target_path=target_path, width=width, height=height)
            parsed.validate()
            return parsed
        asset._mmm_fail_closed_parser = True
        module._asset = asset
__all__ = ['install']
