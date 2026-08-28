import pytest

from minecraft_mod_ai.model_registry import ModelRegistry

T4_QUANTIZED_QWEN_ROLES = {'planner', 'researcher', 'coder', 'coder_safe', 'visual_critic'}

def test_t4_registry_has_role_specific_real_model_ids() -> None:
    registry = ModelRegistry()
    profile = registry.load_profile('t4_local')
    assert 'Qwen' in profile.roles['planner'].model_id
    assert profile.roles['planner'].max_context == 262144
    assert profile.roles['planner'].max_new_tokens == 8192
    assert profile.roles['planner'].extra['dynamic_output_budget'] is True
    assert 'Qwen' in profile.roles['coder'].model_id
    assert profile.roles['coder'].extra['dynamic_output_budget'] is True
    assert 'Qwen' in profile.roles['coder_safe'].model_id
    assert profile.roles['embedding'].model_id == 'Qwen/Qwen3-Embedding-0.6B'
    assert profile.roles['reranker'].model_id == 'Qwen/Qwen3-Reranker-0.6B'
    assert profile.roles['image_generator'].exclusive_gpu is True

def test_t4_quality_non_planner_roles_match_t4_local_except_visual_critic() -> None:
    registry = ModelRegistry()
    local = registry.load_profile('t4_local')
    quality = registry.load_profile('t4_quality')
    for role in set(local.roles) - {'planner', 'visual_critic'}:
        assert quality.roles[role] == local.roles[role]
    assert 'Qwen' in quality.roles['visual_critic'].model_id
    assert quality.roles['visual_critic'].torch_dtype == 'float16'

@pytest.mark.parametrize('registry_path', ['config/model_registry.yaml', 'minecraft_mod_ai/config/model_registry.yaml'])
@pytest.mark.parametrize('profile_name', ['t4_local', 't4_quality'])
def test_t4_quantized_qwen_roles_force_fp16(registry_path: str, profile_name: str) -> None:
    profile = ModelRegistry(registry_path).load_profile(profile_name)
    quantized_qwen = {role: config for role, config in profile.roles.items() if 'Qwen' in config.model_id and config.adapter not in {'embedding', 'reranker'}}
    assert set(quantized_qwen).issuperset({'planner', 'coder', 'researcher'})
    assert {config.torch_dtype for config in quantized_qwen.values()} == {'float16'}
    assert {config.extra.get('dynamic_output_budget') for config in quantized_qwen.values()} == {True}

def test_repository_and_packaged_planner_budgets_stay_in_sync() -> None:
    repository = ModelRegistry('config/model_registry.yaml').role('t4_local', 'planner')
    packaged = ModelRegistry('minecraft_mod_ai/config/model_registry.yaml').role('t4_local', 'planner')
    assert repository.max_context == packaged.max_context == 262144
    assert repository.max_new_tokens == packaged.max_new_tokens == 8192
    assert repository.extra['dynamic_output_budget'] is packaged.extra['dynamic_output_budget'] is True
    repository_quality = ModelRegistry('config/model_registry.yaml').role('t4_quality', 'planner')
    packaged_quality = ModelRegistry('minecraft_mod_ai/config/model_registry.yaml').role('t4_quality', 'planner')
    assert repository_quality == packaged_quality
