from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dev extra supplies tomli.
    import tomli as tomllib

from minecraft_mod_ai import forced_tool_execution_contract
from tools import colab_runtime_setup


def _extras(target: str) -> set[str]:
    prefix, separator, suffix = target.partition("[")
    assert prefix == "." and separator and suffix.endswith("]")
    return {item.strip() for item in suffix[:-1].split(",") if item.strip()}


def test_forced_tool_finalization_does_not_depend_on_local_private_helpers() -> None:
    class RemoteAdapter:
        def generate_turn(self, request):
            return request

    remote_module = SimpleNamespace(OpenAICompatibleAdapter=RemoteAdapter)
    forced_tool_execution_contract.install(openai_compatible_module=remote_module)

    assert getattr(
        RemoteAdapter.generate_turn,
        "_mmm_forced_tool_execution_v1",
        False,
    ) is True


def test_colab_install_targets_reference_only_declared_project_extras() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["optional-dependencies"])

    for target in (
        colab_runtime_setup.REMOTE_PROJECT_INSTALL_TARGET,
        colab_runtime_setup.LOCAL_PROJECT_INSTALL_TARGET,
    ):
        unknown = _extras(target) - declared
        assert not unknown, f"Colab install target references undeclared extras: {sorted(unknown)}"
