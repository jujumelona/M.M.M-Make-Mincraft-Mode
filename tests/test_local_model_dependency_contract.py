try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path


def test_local_model_pins_verified_qwen_runtime_and_published_fastpath() -> None:
    pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
    extras = pyproject['project']['optional-dependencies']
    local_model = extras['local-model']
    qwen_fastpath = extras['qwen-fastpath']
    bnb = extras['transformers-bnb']
    fla_requirement = "flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6; sys_platform == 'linux'"
    assert 'transformers>=4.52.0' in local_model
    assert all(not requirement.startswith('bitsandbytes') for requirement in local_model)
    assert bnb == ["bitsandbytes>=0.45,<1; sys_platform == 'linux'"]
    assert qwen_fastpath == [fla_requirement]
    assert 'training' not in extras
