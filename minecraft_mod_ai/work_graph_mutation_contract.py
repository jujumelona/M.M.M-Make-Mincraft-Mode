from __future__ import annotations
from functools import wraps
from typing import Any, Iterable
_LOCAL_AI_SIDECAR = 'mmm_local_ai_sidecar'
_LLM_CAPABLE_STAGES = frozenset({'custom'})
_CPU_GENERATION_STAGES = frozenset({'content', 'system', 'entity'})

def install(work_graph_module: Any) -> None:
    original_stage = work_graph_module._module_stage
    if not getattr(original_stage, '_mmm_final_stage_contract', False):

        @wraps(original_stage)
        def final_module_stage(module: Any) -> str:
            stage = original_stage(module)
            if getattr(module, 'kind', '') == 'integration' and (not work_graph_module.is_research_shard(module)) and (getattr(module, 'config', {}).get('integration_type') != _LOCAL_AI_SIDECAR):
                return 'custom'
            return stage
        final_module_stage._mmm_final_stage_contract = True
        work_graph_module._module_stage = final_module_stage
    original = work_graph_module._node
    if getattr(original, '_mmm_module_mutation_contract', False):
        return

    @wraps(original)
    def mutation_safe_node(node_id: str, stage: str, dependencies: Iterable[str], payload: dict[str, Any]):
        normalized = dict(payload)
        kind = str(normalized.get('kind', ''))
        generation_stage = str(normalized.get('generation_stage', ''))
        if 'resource_class' not in normalized:
            if kind == 'module-shard':
                if generation_stage in _LLM_CAPABLE_STAGES:
                    normalized['resource_class'] = 'llm'
                elif generation_stage in _CPU_GENERATION_STAGES:
                    normalized['resource_class'] = 'cpu_io'
                else:
                    normalized['resource_class'] = 'commit'
            elif kind == 'asset-shard':
                normalized['resource_class'] = 'image_gpu'
        return original(node_id, stage, dependencies, normalized)
    mutation_safe_node._mmm_module_mutation_contract = True
    mutation_safe_node._mmm_deterministic_cpu_lanes = True
    work_graph_module._node = mutation_safe_node
__all__ = ['install']
