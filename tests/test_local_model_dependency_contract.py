try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path


def test_local_model_keeps_one_native_runtime_without_retired_qwen_gpu_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    local_model = extras["local-model"]
    bnb = extras["transformers-bnb"]
    assert "transformers>=4.52.0" in local_model
    assert "huggingface-hub>=0.28,<2" in local_model
    assert all(not requirement.startswith("bitsandbytes") for requirement in local_model)
    assert bnb == ["bitsandbytes>=0.45,<1; sys_platform == 'linux'"]
    assert "qwen-fastpath" not in extras
    assert all(
        "flash-linear-attention" not in requirement
        for group in extras.values()
        for requirement in group
    )
    assert "training" not in extras
