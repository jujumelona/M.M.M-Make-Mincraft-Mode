try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from pathlib import Path


def test_local_model_pins_verified_qwen_runtime_and_published_fastpath() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    local_model = extras["local-model"]
    qwen_fastpath = extras["qwen-fastpath"]
    fla_requirement = (
        "flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6; "
        "sys_platform == 'linux'"
    )

    assert "transformers>=4.52.0" in local_model
    assert qwen_fastpath == [fla_requirement]


def test_colab_requirements_inherit_local_model_fastpath_contract() -> None:
    requirements = Path("requirements-colab.txt").read_text(encoding="utf-8")

    assert "ui,local-model,rag,image,speech,production-audio,training" in requirements
    assert "local-model extra includes the Linux/CUDA-only" in requirements
    assert ".[qwen-fastpath]" not in requirements
