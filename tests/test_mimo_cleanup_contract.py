"""Production MiMo migration guard: removed Qwen3.5 runtime cannot re-enter."""
from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "minecraft_mod_ai" / "config" / "model_registry.yaml"
COLAB = ROOT / "M.M.M_Make_Mincraft_Mode_Colab.ipynb"
TEMPLATE = ROOT / "minecraft_mod_ai" / "templates" / "mimo_v2_6.jinja"
OLD_RUNTIME_FILES = (
    "qwen35_mtp_hotpath_contract.py",
    "qwen35_request_policy.py",
    "qwen35_runtime_efficiency_contract.py",
)


def test_production_roles_use_only_selected_mimo_checkpoint() -> None:
    registry = ModelRegistry()
    for profile_name in ("MiMo-V2.6-9B_6GB", "t4_local", "t4_quality"):
        profile = registry.load_profile(profile_name)
        for role in ("planner", "researcher", "coder", "coder_safe", "visual_critic"):
            cfg = profile.roles[role]
            assert cfg.model_id == "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"
            assert cfg.extra["runtime_contract"] == "mimo"
            assert cfg.extra["chat_template_file"] == TEMPLATE.name
            assert cfg.extra["supports_mtp"] is False
            assert "native_mtp" not in cfg.extra
            assert "qwen_family" not in cfg.extra
            assert "qwen_tool_markup" not in cfg.extra
            assert "lora_adapters" not in cfg.extra

    assert registry.load_profile("Qwen3.8-27B_18GB").roles["planner"].extra["runtime_contract"] == "qwen"


def test_deleted_qwen35_runtime_is_not_importable_from_production() -> None:
    modules = ROOT / "minecraft_mod_ai"
    for name in OLD_RUNTIME_FILES:
        assert not (modules / name).exists()
    for path in (
        modules / "llama_tuning_pipeline.py",
        modules / "llama_server_runtime_tuning.py",
        modules / "config" / "model_registry.yaml",
        ROOT / "tools" / "colab_runtime_setup.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "MMM_QWEN35" not in source
        assert "Qwen3.5-9B-MTP-GGUF" not in source
        assert "Qwen3.5-9B-UD-Q4_K_XL.gguf" not in source


def test_colab_and_readme_do_not_offer_deleted_default() -> None:
    notebook = json.loads(COLAB.read_text(encoding="utf-8"))
    code = "\n".join(
        cell.get("source", "") if isinstance(cell.get("source"), str)
        else "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert 'MODEL_PROFILE = "MiMo-V2.6-9B_6GB"' in code
    assert "Qwen3.5-9B_6GB" not in code
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "- `MiMo-V2.6-9B_6GB`" in readme
    assert "Qwen3.5-9B_6GB" not in readme


def test_mimo_jinja_shipped_and_not_qwen_parser_signature() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "<function=" not in source
    assert "<parameter=" not in source
    assert "'<func' ~ 'tion='" in source
    assert "<think>" in source
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"templates/**/*.jinja"' in package


def test_default_runtime_bundles_match_pinned_mimo_server() -> None:
    setup = (ROOT / "tools" / "colab_runtime_setup.py").read_text(encoding="utf-8")
    bundle = (ROOT / "tools" / "native_llama_bundle.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "build-native-llama-cuda.yml").read_text(encoding="utf-8")
    assert "d81235049384534c167caea52b85a694f6103d14" in setup
    assert "d81235049384534c167caea52b85a694f6103d14" in workflow
    assert "native-llama-d812350-" in bundle
    assert "native-llama-1d2869c-" not in bundle


def test_live_capture_workflow_uses_model_native_tool_runner() -> None:
    workflow = (ROOT / ".github" / "workflows" / "model-live-integration.yml").read_text(encoding="utf-8")
    assert "tools.model_live_integration_capture" in workflow
    assert "qwen_live_integration_capture" not in workflow
    assert "--scenario" in workflow
    assert "--stop" not in workflow
    assert (ROOT / "tools" / "model_live_integration_capture.py").exists()
    assert not (ROOT / "tools" / "qwen_live_integration_capture.py").exists()
