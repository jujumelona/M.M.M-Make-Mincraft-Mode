from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "minecraft_mod_ai/progress_aware_tool_loop.py"
text = PATH.read_text(encoding="utf-8")


def replace_once_or_done(old: str, new: str) -> None:
    global text
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1:
        text = text.replace(old, new, 1)
        return
    if old_count == 0 and new_count == 1:
        return
    raise SystemExit(
        "progress_aware_tool_loop.py: ambiguous cleanup state "
        f"old={old_count} new={new_count}: {old[:80]!r}"
    )


replace_once_or_done(
    '    if clean.startswith("http://") or clean.startswith("https://") or "://" in clean:\n',
    '    if clean.startswith(("http://", "https://")) or "://" in clean:\n',
)
replace_once_or_done(
    '''    suffix = Path(clean).suffix.casefold()\n    if "/" not in clean and suffix not in {".java", ".json", ".toml", ".gradle", ".properties", ".txt", ".md", ".kt", ".groovy"}:\n        return False\n    return True\n''',
    '''    suffix = Path(clean).suffix.casefold()\n    workspace_suffixes = {\n        ".java",\n        ".json",\n        ".toml",\n        ".gradle",\n        ".properties",\n        ".txt",\n        ".md",\n        ".kt",\n        ".groovy",\n    }\n    return "/" in clean or suffix in workspace_suffixes\n''',
)
replace_once_or_done(
    '        def execute(call: Any) -> tuple[Any, Mapping[str, Any]]:\n',
    '''        def execute(\n            call: Any,\n            *,\n            allowed_tool_names: frozenset[str] = phase_tool_names,\n            localization_stage: LocalizationStage = current_localization_stage,\n        ) -> tuple[Any, Mapping[str, Any]]:\n''',
)
replace_once_or_done(
    "            if call.name not in phase_tool_names:\n",
    "            if call.name not in allowed_tool_names:\n",
)
replace_once_or_done(
    '                    f"(allowed tools: {sorted(phase_tool_names)})."\n',
    '                    f"(allowed tools: {sorted(allowed_tool_names)})."\n',
)
replace_once_or_done(
    '''                localization_attempt_stage = (\n                    current_localization_stage\n                    if implementation_requires_mutation and state.phase == LoopPhase.OBSERVE\n                    else None\n                )\n''',
    '''                localization_attempt_stage = (\n                    localization_stage\n                    if implementation_requires_mutation and state.phase == LoopPhase.OBSERVE\n                    else None\n                )\n''',
)
replace_once_or_done(
    "            except Exception as exc:\n",
    "            except Exception as exc:  # noqa: BLE001 - tool failures are model observations\n",
)
replace_once_or_done(
    '''                ctx_progress = False\n                if after_ctx is not None:\n                    if before_ctx is None or after_ctx.localization_stage != before_ctx.localization_stage or after_ctx.target_path != before_ctx.target_path or after_ctx.target_symbol != before_ctx.target_symbol or after_ctx.source_body != before_ctx.source_body:\n                        ctx_progress = True\n''',
    '''                ctx_progress = bool(\n                    after_ctx is not None\n                    and (\n                        before_ctx is None\n                        or after_ctx.localization_stage != before_ctx.localization_stage\n                        or after_ctx.target_path != before_ctx.target_path\n                        or after_ctx.target_symbol != before_ctx.target_symbol\n                        or after_ctx.source_body != before_ctx.source_body\n                    )\n                )\n''',
)
replace_once_or_done(
    '''                    if state.phase == LoopPhase.OBSERVE and implementation_requires_mutation:\n                        if is_mutation_ready(messages, state):\n                            state.phase = LoopPhase.ACT\n''',
    '''                    if (\n                        state.phase == LoopPhase.OBSERVE\n                        and implementation_requires_mutation\n                        and is_mutation_ready(messages, state)\n                    ):\n                        state.phase = LoopPhase.ACT\n''',
)

PATH.write_text(text, encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
