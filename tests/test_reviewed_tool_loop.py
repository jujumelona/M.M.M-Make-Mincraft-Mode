from minecraft_mod_ai.reviewed_tool_loop import _is_host_owned_mutation


def test_project_mutation_stays_host_owned():
    assert _is_host_owned_mutation("apply_source_edit") is True
    assert _is_host_owned_mutation("generate_fabric_project") is True
    assert _is_host_owned_mutation("package_release") is True


def test_observation_tools_remain_model_callable():
    assert _is_host_owned_mutation("search_code_rag") is False
    assert _is_host_owned_mutation("java_diagnostics") is False
    assert _is_host_owned_mutation("runtime_logs") is False
