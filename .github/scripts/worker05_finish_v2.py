from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    "import os\nimport threading\nimport time\n",
    "import os\n",
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    "values = tuple(sorted(set(int(value) for value in parallel_values)))",
    "values = tuple(sorted({int(value) for value in parallel_values}))",
)
replace_once(
    "minecraft_mod_ai/qwen35_mtp_hotpath_contract.py",
    '''        if value in {"--port", "-p"} and index + 1 < len(args):\n            if args[index + 1] == target:\n                return True\n''',
    '''        if (\n            value in {"--port", "-p"}\n            and index + 1 < len(args)\n            and args[index + 1] == target\n        ):\n            return True\n''',
)
replace_once(
    "tests/test_llama_semantic_progress_watchdog.py",
    '''from minecraft_mod_ai.llama_completion_liveness_contract import (\n    LlamaSemanticProgressTimeout,\n    _SemanticProgressWatchdog,\n    _semantic_progress_from_sse_line,\n)\n''',
    '''from minecraft_mod_ai.llama_completion_liveness_contract import (\n    LlamaSemanticProgressTimeout,\n    _semantic_progress_from_sse_line,\n    _SemanticProgressWatchdog,\n)\n''',
)

Path(__file__).unlink(missing_ok=True)
