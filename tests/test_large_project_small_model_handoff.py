from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from minecraft_mod_ai.model_context_budget import (
    fit_messages_to_context,
    request_message_budget,
)
from minecraft_mod_ai.project_index import ProjectIndex
from minecraft_mod_ai.repository_grounding import build_repository_observation_ledger


def _create_large_mock_project(root: Path, file_count: int = 40) -> Path:
    src_main_java = root / "src" / "main" / "java" / "com" / "example" / "mod"
    src_main_java.mkdir(parents=True, exist_ok=True)
    src_main_res = root / "src" / "main" / "resources" / "data" / "example" / "recipes"
    src_main_res.mkdir(parents=True, exist_ok=True)

    for i in range(file_count):
        java_file = src_main_java / f"Feature{i}Block.java"
        java_file.write_bytes(
            (
                f"package com.example.mod;\n\n"
                f"public class Feature{i}Block {{\n"
                f"    public static final String ID = \"feature_{i}\";\n"
                f"    public void register() {{\n"
                f"        // Registration logic for feature {i}\n"
                f"    }}\n"
                f"    public int getPowerLevel() {{\n"
                f"        return {i * 10};\n"
                f"    }}\n"
                f"}}\n"
            ).encode()
        )

        recipe_file = src_main_res / f"feature_{i}_recipe.json"
        recipe_file.write_bytes(
            json.dumps(
                {
                    "type": "minecraft:crafting_shaped",
                    "result": {"item": f"example:feature_{i}"},
                    "pattern": ["###", " # ", " # "],
                },
                indent=2,
            ).encode("utf-8")
        )

    return root


def test_large_project_planner_researcher_coder_handoff_bounded(tmp_path: Path) -> None:
    project_root = _create_large_mock_project(tmp_path / "large_mod")
    index = ProjectIndex(project_root)

    assert len(index.files) >= 80

    small_model_config = MagicMock()
    small_model_config.adapter = "llama_cpp"
    small_model_config.max_context = 32768
    small_model_config.max_input_tokens = 0
    small_model_config.max_new_tokens = 8192
    small_model_config.extra = {
        "gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
        "runtime_contract": "qwen",
        "qwen_family": "qwen3.5",
        "qwen_tool_markup": "qwen3_coder_xml",
        "qwen_action_thinking_control": "enable_thinking_false",
        "qwen_preserve_thinking": False,
        "qwen_reasoning_effort": False,
        "qwen_assistant_prefill": True,
    }

    budget = request_message_budget(small_model_config, tools=())
    assert 12 * 1024 <= budget <= 96 * 1024

    query = "Feature12Block registration and recipe"
    ledger = build_repository_observation_ledger(
        None,
        index,
        query=query,
        byte_budget=budget // 2,
    )

    ledger_json = json.dumps(ledger, ensure_ascii=False, separators=(",", ":"))
    assert len(ledger_json.encode("utf-8")) <= budget // 2
    assert "Feature12Block.java" in ledger_json
    assert "receipt" in ledger

    user_payload = {
        "phase": "implement_module",
        "task": "Add custom power scaling to Feature12Block",
        "initial_exact_source_context": ledger,
    }
    messages = [
        {"role": "system", "content": "You are an expert Minecraft Fabric coder."},
        {"role": "user", "content": json.dumps(user_payload)},
    ]

    fitted = fit_messages_to_context(messages, config=small_model_config, tools=())
    fitted_bytes = len(json.dumps(fitted).encode("utf-8"))
    assert fitted_bytes <= budget

    fitted_str = json.dumps(fitted)
    assert "Feature39Block" not in fitted_str or "omitted_record_count" in fitted_str
