import json
import runpy
from pathlib import Path
from types import SimpleNamespace
import nbformat
import pytest
NOTEBOOK_PATH = Path('M.M.M_Make_Mincraft_Mode_Colab.ipynb')
OBSOLETE_NOTEBOOK_PATH = Path('Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb')

def _load_notebook():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook

def _cells() -> dict[str, str]:
    return {cell['id']: cell['source'] for cell in _load_notebook().cells}

def test_notebook_is_single_registry_driven_source() -> None:
    forbidden = ('Qwen/Qwen3.5-9B-Instruct', 'Qwen/Qwen3.5-4B-Instruct', 'built-in 폴백', 'deterministic-fallback', 'model_name_or_path=LOCAL_MODEL_ID')
    raw = NOTEBOOK_PATH.read_text(encoding='utf-8')
    assert 'MODEL_PROFILE' in raw
    assert 'config/model_registry.yaml' in raw
    assert all((token not in raw for token in forbidden))
    assert not OBSOLETE_NOTEBOOK_PATH.exists()

def test_notebook_has_stable_unique_cell_contract() -> None:
    notebook = _load_notebook()
    ids = [cell['id'] for cell in notebook.cells]
    assert ids == ['title', 'configuration', 'setup', 'existing-input', 'registry', 'plan', 'build', 'download', 'boundaries']
    assert len(ids) == len(set(ids))
    assert notebook.metadata['colab']['name'] == NOTEBOOK_PATH.name

def test_existing_zip_upload_is_explicit_and_bound_to_revise_mode() -> None:
    cells = _cells()
    run_modes = Path('minecraft_mod_ai/colab_run_modes.py').read_text(encoding='utf-8')
    assert 'RUN_MODE = "Full"' in cells['configuration']
    assert '"Revise"' in cells['configuration']
    assert 'prepare_existing_mod_input(RUN_MODE)' in cells['existing-input']
    assert 'PATCH_EXISTING' not in '\n'.join(cells.values())
    assert 'colab_files.upload()' in run_modes
    assert 'len(uploaded) != 1' in run_modes
    assert 'suffix=".zip"' in run_modes
    assert 'inspect_existing_project_archive(source)' in run_modes
    assert 'report.has_sources' in run_modes
    assert 'report.has_gradle_project' in run_modes
    assert 'existing_input=EXISTING_INPUT' in cells['plan']
    assert 'approval_hash' not in '\n'.join(cells.values())
    assert 'complete_proposal' not in '\n'.join(cells.values())

def test_setup_stops_managed_server_before_same_commit_engine_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup = runpy.run_path('tools/colab_runtime_setup.py')
    validate_checkout = setup['_validate_checkout']
    calls: list[str] = []
    package = SimpleNamespace(__file__=str(tmp_path / 'minecraft_mod_ai' / '__init__.py'))
    autotune = SimpleNamespace(_MANAGED_URL='http://127.0.0.1:8910', _shutdown_managed_server=lambda: calls.append('shutdown'))
    fake_modules = {'minecraft_mod_ai': package, 'minecraft_mod_ai.llama_server_autotune': autotune}
    fake_sys = SimpleNamespace(modules=fake_modules)
    fake_importlib = SimpleNamespace(invalidate_caches=lambda: calls.append('invalidate'))
    monkeypatch.setitem(validate_checkout.__globals__, 'sys', fake_sys)
    monkeypatch.setitem(validate_checkout.__globals__, 'importlib', fake_importlib)
    monkeypatch.setenv('LLAMA_SERVER_URL', autotune._MANAGED_URL)
    monkeypatch.setitem(validate_checkout.__globals__, '_git_head', lambda _path: 'a' * 40)
    monkeypatch.setitem(validate_checkout.__globals__, '_tracked_changes', lambda _path: '')
    (tmp_path / '.git').mkdir()
    validate_checkout(repo_dir=tmp_path, used_commit='a' * 40, previous_commit='a' * 40, engine_was_loaded=True, engine_module_file=package.__file__)
    assert calls == ['shutdown', 'invalidate']
    assert 'minecraft_mod_ai' not in fake_modules
    assert 'minecraft_mod_ai.llama_server_autotune' not in fake_modules
    assert 'LLAMA_SERVER_URL' not in __import__('os').environ

def test_notebook_checks_setup_fingerprint_and_prints_resolved_planner() -> None:
    cells = _cells()
    assert 'PERFORMANCE_MODE = "Auto"' in cells['configuration']
    assert 'os.environ["MMM_PERFORMANCE_MODE"] = performance_mode' in cells['configuration']
    assert 'os.environ["MMM_PERFORMANCE_MODE"]' not in cells['plan']
    assert 'def assert_current_colab_setup' in cells['registry']
    assert 'COLAB_SETUP_MODULE.assert_setup_state(' in cells['registry']
    assert 'planner_config = registry_manager.role' in cells['registry']
    assert 'planner_config.model_id' in cells['registry']
    assert 'planner_config.quantization' in cells['registry']
    assert 'planner_config.max_context' in cells['registry']
    assert 'planner_config.max_input_tokens' in cells['registry']
    assert 'planner_config.max_new_tokens' in cells['registry']
    assert '기획 native context:' in cells['registry']
    assert '기획 page input:' in cells['registry']
    assert '기획 page output:' in cells['registry']
    assert 'assert_current_colab_setup()' in cells['plan']
    assert 'assert_current_colab_setup()' in cells['build']

def test_qwen_fastpath_extra_is_linux_only_and_includes_fixed_fla() -> None:
    pyproject = Path('pyproject.toml').read_text(encoding='utf-8')
    assert 'qwen-fastpath = [' in pyproject
    assert 'flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6' in pyproject
    assert "sys_platform == 'linux'" in pyproject

def test_notebook_code_cells_compile_top_to_bottom() -> None:
    for cell in _load_notebook().cells:
        if cell['cell_type'] == 'code':
            assert cell.get('execution_count') is None
            assert not cell.get('outputs')
            compile(cell['source'], f"<colab:{cell['id']}>", 'exec')
