from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


# 1) Replace the fixed 40 KiB context-pressure cliff with live-token-guided recovery.
path = Path("minecraft_mod_ai/progress_aware_tool_loop.py")
text = path.read_text(encoding="utf-8")
anchor = "\ndef _generate_turn_with_context_recovery(\n"
helper = '''
def _exact_context_recovery_candidate(
    messages: Sequence[Mapping[str, Any]],
    *,
    turn_request: GenerationRequest,
    exact_accounting: Any,
    config: Any,
    tools: Sequence[Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, int]] | None:
    """Select the largest deterministic retry that leaves useful live output space."""

    base_budget = max(1, int(request_message_budget(config, tools)))
    budgets = tuple(
        dict.fromkeys(
            max(1, base_budget * numerator // 8)
            for numerator in (8, 7, 6, 5, 4, 3, 2, 1)
        )
    )
    original = tuple(messages)
    fallback: tuple[tuple[Mapping[str, Any], ...], dict[str, int]] | None = None

    for budget in budgets:
        candidate = tuple(emergency_fit_messages(original, budget_bytes=budget))
        if candidate == original:
            continue
        accounting = exact_accounting(replace(turn_request, messages=candidate))
        input_tokens = int(accounting.input_tokens)
        context_tokens = int(accounting.context_tokens)
        remaining_tokens = context_tokens - input_tokens
        if remaining_tokens <= 0:
            continue
        receipt = {
            "budget_bytes": budget,
            "input_tokens": input_tokens,
            "context_tokens": context_tokens,
            "remaining_tokens": remaining_tokens,
        }
        if fallback is None:
            fallback = (candidate, receipt)
        configured_output = max(1, int(getattr(config, "max_new_tokens", 0) or 1))
        desired_reserve = min(configured_output, max(1, context_tokens // 4))
        if remaining_tokens >= desired_reserve:
            return candidate, receipt
    return fallback

'''
if "def _exact_context_recovery_candidate(" not in text:
    if anchor not in text:
        raise SystemExit("missing context recovery insertion anchor")
    text = text.replace(anchor, "\n" + helper + "def _generate_turn_with_context_recovery(\n", 1)

old = '''        emergency_budget = min(
            40 * 1024,
            request_message_budget(config, request.tools),
        )
        emergency = emergency_fit_messages(
            messages,
            budget_bytes=emergency_budget,
        )
        if not _replace_live_messages(messages, emergency):
            mark_context_recovery_exhausted(exc)
            raise
        retry_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=media_paths,
        )
        print(
            "agent context: deterministic overflow recovery",
            f"messages={len(messages)}",
            f"budget_bytes={emergency_budget}",
            flush=True,
        )
'''
new = '''        recovery_receipt: dict[str, int] = {}
        if callable(exact_accounting):
            exact_recovery = _exact_context_recovery_candidate(
                messages,
                turn_request=turn_request,
                exact_accounting=exact_accounting,
                config=config,
                tools=request.tools,
            )
            if exact_recovery is None:
                mark_context_recovery_exhausted(exc)
                raise
            emergency, recovery_receipt = exact_recovery
        else:
            active_budget = max(1, request_message_budget(config, request.tools))
            emergency_budget = max(1, active_budget * 3 // 4)
            emergency = emergency_fit_messages(
                messages,
                budget_bytes=emergency_budget,
            )
            recovery_receipt = {"budget_bytes": emergency_budget}
        if not _replace_live_messages(messages, emergency):
            mark_context_recovery_exhausted(exc)
            raise
        retry_request = replace(
            turn_request,
            messages=tuple(messages),
            media_paths=media_paths,
        )
        print(
            "agent context: deterministic overflow recovery",
            f"messages={len(messages)}",
            *(f"{key}={value}" for key, value in recovery_receipt.items()),
            flush=True,
        )
'''
text = replace_once(text, old, new, label="40KiB context recovery")
path.write_text(text, encoding="utf-8")


# 2) Remove the hidden 40 KiB default from the canonical emergency packer.
path = Path("minecraft_mod_ai/model_context_budget.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "    budget_bytes: int = 40 * 1024,\n",
    "    budget_bytes: int | None = None,\n",
    label="model context emergency default",
)
text = replace_once(
    text,
    "    budget = max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, int(budget_bytes)))\n",
    "    requested_budget = _MAX_CONTEXT_BYTES if budget_bytes is None else int(budget_bytes)\n"
    "    budget = max(_MIN_CONTEXT_BYTES, min(_MAX_CONTEXT_BYTES, requested_budget))\n",
    label="model context emergency budget",
)
path.write_text(text, encoding="utf-8")


# 3) Keep the installed llama hard-cap guard aligned with the optional budget.
path = Path("minecraft_mod_ai/llama_context_safety_contract.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    "            budget_bytes: int = 40 * 1024,\n",
    "            budget_bytes: int | None = None,\n",
    label="llama safety emergency default",
)
old = '''            exact_budget = max(
                1,
                min(int(context_module._MAX_CONTEXT_BYTES), int(budget_bytes)),
            )
'''
new = '''            requested_budget = (
                int(context_module._MAX_CONTEXT_BYTES)
                if budget_bytes is None
                else int(budget_bytes)
            )
            exact_budget = max(
                1,
                min(int(context_module._MAX_CONTEXT_BYTES), requested_budget),
            )
'''
text = replace_once(text, old, new, label="llama safety emergency budget")
path.write_text(text, encoding="utf-8")


# 4) Preserve all host-approved authored research queries after dedupe.
path = Path("minecraft_mod_ai/authored_scope_research_contract.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    '"search_queries": queries[:5],',
    '"search_queries": queries,',
    label="search query cap",
)
text = replace_once(
    text,
    '"research_queries": queries[:5],',
    '"research_queries": queries,',
    label="research query cap",
)
path.write_text(text, encoding="utf-8")


# 5) Preserve every acceptance reference after canonical dedupe.
path = Path("minecraft_mod_ai/production_boundary_contract.py")
text = path.read_text(encoding="utf-8")
text = replace_once(
    text,
    'group["acceptance_refs"] = list(dict.fromkeys(refs))[:3]',
    'group["acceptance_refs"] = list(dict.fromkeys(refs))',
    label="acceptance ref cap",
)
path.write_text(text, encoding="utf-8")


# 6) Regression: exact recovery picks the largest live-safe candidate, not a fixed 40 KiB cliff.
path = Path("tests/test_agent_context_window_contract.py")
text = path.read_text(encoding="utf-8")
if "test_exact_context_recovery_uses_live_tokens_not_fixed_40k" not in text:
    text += '''


def test_exact_context_recovery_uses_live_tokens_not_fixed_40k(monkeypatch) -> None:
    original = (
        {"role": "system", "content": "s" * 100},
        {"role": "user", "content": "u" * 100},
        {"role": "assistant", "content": "a" * 100},
    )
    seen_budgets: list[int] = []

    def fake_emergency(messages, *, budget_bytes):
        seen_budgets.append(int(budget_bytes))
        marker = max(1, int(budget_bytes))
        return (*tuple(messages[:2]), {"role": "system", "content": f"budget={marker}"})

    class Accounting:
        def __init__(self, input_tokens: int):
            self.input_tokens = input_tokens
            self.context_tokens = 32_768

    def exact(request):
        marker = int(str(request.messages[-1]["content"]).split("=")[-1])
        return Accounting(20_000 if marker <= 72 * 1024 else 30_000)

    monkeypatch.setattr(tool_loop, "request_message_budget", lambda config, tools: 96 * 1024)
    monkeypatch.setattr(tool_loop, "emergency_fit_messages", fake_emergency)
    request = GenerationRequest(messages=original)
    recovered = tool_loop._exact_context_recovery_candidate(
        original,
        turn_request=request,
        exact_accounting=exact,
        config=_config(),
        tools=(),
    )
    assert recovered is not None
    candidate, receipt = recovered
    assert receipt["budget_bytes"] == 72 * 1024
    assert receipt["remaining_tokens"] == 12_768
    assert "budget=73728" in str(candidate[-1]["content"])
    assert seen_budgets[:3] == [96 * 1024, 84 * 1024, 72 * 1024]
'''
path.write_text(text, encoding="utf-8")
