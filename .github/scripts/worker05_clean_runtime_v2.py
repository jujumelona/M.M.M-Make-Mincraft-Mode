from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once_or_done(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        return
    if old_count == 0 and new_count == 1:
        return
    raise SystemExit(
        f"{path}: ambiguous cleanup state old={old_count} new={new_count}"
    )


def remove_line_if_present(path: str, line: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(line)
    if count > 1:
        raise SystemExit(f"{path}: expected at most one line, found {count}: {line!r}")
    if count == 1:
        target.write_text(text.replace(line, "", 1), encoding="utf-8")


def replace_indented_block(
    path: str,
    *,
    header: str,
    replacement_body: tuple[str, ...],
) -> None:
    target = ROOT / path
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.strip() == header]
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected one block header {header!r}, found {len(matches)}")
    start = matches[0]
    prefix = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    indent = len(prefix)
    end = start + 1
    while end < len(lines):
        raw = lines[end]
        if not raw.strip():
            end += 1
            continue
        current_indent = len(raw) - len(raw.lstrip())
        if current_indent <= indent:
            break
        end += 1
    wanted = [lines[start]] + [prefix + "    " + body + "\n" for body in replacement_body]
    existing = "".join(lines[start:end])
    desired = "".join(wanted)
    if existing == desired:
        return
    lines[start:end] = wanted
    target.write_text("".join(lines), encoding="utf-8")


# 1. Own the finite host tool-round policy directly in model_router.
replace_once_or_done(
    "minecraft_mod_ai/model_router.py",
    '''def _agent_tool_round_limit() -> int | None:\n    raw = os.environ.get("MMM_AGENT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return None\n    try:\n        value = int(raw)\n    except ValueError:\n        return None\n    return value if value > 0 else None\n''',
    '''_DEFAULT_AGENT_TOOL_ROUNDS = 128\n_MIN_AGENT_TOOL_ROUNDS = 16\n_MAX_AGENT_TOOL_ROUNDS = 512\n\n\ndef _default_agent_tool_rounds() -> int:\n    raw = os.environ.get("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return _DEFAULT_AGENT_TOOL_ROUNDS\n    try:\n        value = int(raw)\n    except ValueError:\n        return _DEFAULT_AGENT_TOOL_ROUNDS\n    return max(_MIN_AGENT_TOOL_ROUNDS, min(_MAX_AGENT_TOOL_ROUNDS, value))\n\n\ndef _agent_tool_round_limit() -> int:\n    raw = os.environ.get("MMM_AGENT_TOOL_ROUNDS", "").strip()\n    if not raw:\n        return _default_agent_tool_rounds()\n    try:\n        value = int(raw)\n    except ValueError:\n        return _default_agent_tool_rounds()\n    return value if value > 0 else _default_agent_tool_rounds()\n''',
)
remove_line_if_present(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    from .llama_tool_round_safety_contract import install as install_tool_round_safety\n",
)
remove_line_if_present(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    install_tool_round_safety(model_router)\n",
)


# 2. Remove both lossy three-message context fallbacks. The canonical context owner
# must either produce a protocol-safe fit or fail explicitly; no hidden third decode.
replace_indented_block(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    header="if not _replace_live_messages(messages, emergency):",
    replacement_body=(
        "mark_context_recovery_exhausted(exc)",
        "raise",
    ),
)
replace_indented_block(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    header="if retry_kind == CONTEXT_PRESSURE:",
    replacement_body=(
        "mark_context_recovery_exhausted(exc)",
        "raise exc from retry_exc",
    ),
)
replace_once_or_done(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    "        if round_limit is not None and state.step_index > round_limit:\n",
    "        if state.step_index > round_limit:\n",
)
replace_once_or_done(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    "Agent reached the explicit tool-round limit before required "\n',
    '                    "Agent reached the host tool-round limit before required "\n',
)
replace_once_or_done(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    "Writable coder reached the explicit tool-round limit before a "\n',
    '                    "Writable coder reached the host tool-round limit before a "\n',
)
replace_once_or_done(
    "minecraft_mod_ai/progress_aware_tool_loop.py",
    '                    f"The explicitly configured host tool limit was reached after {round_limit} rounds. "\n',
    '                    f"The host tool-round limit was reached after {round_limit} rounds. "\n',
)


# 3. With the lossy fallback gone, the wrapper that intercepted that fallback is dead.
context_path = ROOT / "minecraft_mod_ai/llama_context_safety_contract.py"
context_text = context_path.read_text(encoding="utf-8")
marker = '_REPLACE_MARKER = "_mmm_protocol_safe_live_replace_v1"\n'
if marker in context_text:
    context_text = context_text.replace(marker, "", 1)
    start = context_text.find("\ndef _is_unsafe_three_message_fallback(")
    end = context_text.find("\ndef install(context_module: Any) -> None:")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit("llama_context_safety_contract.py: replacement guard block malformed")
    context_text = context_text[:start] + context_text[end:]
    context_text = context_text.replace("\n    _install_tool_loop_guards(context_module)\n", "\n")
    context_text = context_text.replace('    "_is_unsafe_three_message_fallback",\n', "")
elif "_is_unsafe_three_message_fallback" in context_text:
    raise SystemExit("llama_context_safety_contract.py: guard survived without marker")
context_path.write_text(context_text, encoding="utf-8")


# 4. Move tests from the deleted shim to the canonical owner.
old_test = ROOT / "tests/test_llama_tool_round_safety_contract.py"
if old_test.exists():
    old_test.unlink()
(ROOT / "tests/test_agent_tool_round_policy.py").write_text(
    '''from __future__ import annotations\n\nfrom minecraft_mod_ai import model_router\n\n\ndef test_default_tool_round_limit_is_finite(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)\n    monkeypatch.delenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", raising=False)\n    assert model_router._agent_tool_round_limit() == 128\n\n\ndef test_default_tool_round_limit_is_tunable_with_safe_bounds(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    assert model_router._agent_tool_round_limit() == 64\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "1")\n    assert model_router._agent_tool_round_limit() == 16\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "9999")\n    assert model_router._agent_tool_round_limit() == 512\n\n\ndef test_explicit_positive_tool_round_limit_wins(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "37")\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    assert model_router._agent_tool_round_limit() == 37\n\n\ndef test_invalid_or_nonpositive_explicit_limit_stays_finite(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_AGENT_DEFAULT_TOOL_ROUNDS", "64")\n    for raw in ("garbage", "0", "-7"):\n        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)\n        assert model_router._agent_tool_round_limit() == 64\n''',
    encoding="utf-8",
)

test_context = ROOT / "tests/test_llama_context_safety_contract.py"
test_text = test_context.read_text(encoding="utf-8")
legacy_start = test_text.find("\ndef test_ultra_context_fallback_never_silently_drops_authored_task()")
if legacy_start >= 0:
    legacy_end = test_text.find("\ndef test_runtime_context_safety_wrappers_are_installed()", legacy_start)
    if legacy_end < 0:
        raise SystemExit("context safety legacy test terminator not found")
    test_text = test_text[:legacy_start] + test_text[legacy_end:]
test_text = test_text.replace(
    '''    assert getattr(\n        tool_loop._replace_live_messages,\n        "_mmm_protocol_safe_live_replace_v1",\n        False,\n    )\n''',
    "",
)
test_text = test_text.replace(
    "from minecraft_mod_ai import progress_aware_tool_loop as tool_loop\n",
    "",
)
test_context.write_text(test_text, encoding="utf-8")

shim = ROOT / "minecraft_mod_ai/llama_tool_round_safety_contract.py"
shim.unlink(missing_ok=True)
for base in (ROOT / "minecraft_mod_ai", ROOT / "tests"):
    for path in base.rglob("*.py"):
        if "llama_tool_round_safety_contract" in path.read_text(encoding="utf-8"):
            raise SystemExit(f"stale tool-round shim reference: {path.relative_to(ROOT)}")

# One-shot scaffolding must not survive a successful cleanup commit.
(ROOT / ".github/scripts/worker05_clean_runtime_v1.py").unlink(missing_ok=True)
(ROOT / ".github/workflows/worker05-clean-runtime-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
