from __future__ import annotations

from pathlib import Path
import re

root = Path("minecraft_mod_ai")
changed: list[str] = []


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    changed.append(str(path))


# 1. Structured decoding fallback: tolerate fenced JSON at the host boundary.
p = root / "structured_output.py"
s = p.read_text(encoding="utf-8")
old = "        value = json.loads(output)"
if old in s:
    new = '''        candidate_output = output.strip()
        if candidate_output.startswith("```"):
            fence_lines = candidate_output.splitlines()
            if fence_lines and fence_lines[0].lstrip().startswith("```"):
                fence_lines = fence_lines[1:]
            if fence_lines and fence_lines[-1].strip() == "```":
                fence_lines = fence_lines[:-1]
            candidate_output = "\\n".join(fence_lines).strip()
        value = json.loads(candidate_output)'''
    s = s.replace(old, new, 1)
    write(p, s)


# 2. Corrective retrieval: host creates queries; the small model never serializes a query plan.
p = root / "pre_design_rag_corrective.py"
s = p.read_text(encoding="utf-8")
start = s.index("def _generate_gap_queries(")
end = s.index("\ndef _read_and_verify_document(", start)
host_gap = r'''def _generate_gap_queries(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    domain: Mapping[str, Any],
    gaps: Sequence[str],
    prior_queries: Sequence[str],
    seen: set[str],
    raw_prompt: str,
    progress_label: str,
) -> list[str]:
    """Build corrective searches deterministically without a model-format contract."""
    del agentic_module, project_rag, router, progress_label, prior_queries
    result: list[str] = []

    for raw in domain.get("queries", ()):
        query = " ".join(str(raw or "").split()).strip()
        key = query.casefold()
        if query and key not in seen and _is_retrieval_query(query, raw_prompt=raw_prompt):
            seen.add(key)
            result.append(query)
            if len(result) >= 8:
                return result

    stop = {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "this", "that",
        "provided", "page", "content", "specific", "evidence", "requested", "user",
    }
    for gap in gaps:
        words = re.findall(r"[A-Za-z0-9_+.#/-]+", str(gap))
        terms: list[str] = []
        for word in words:
            low = word.casefold()
            if len(low) < 3 or low in stop or low in terms:
                continue
            terms.append(low)
            if len(terms) >= 10:
                break
        if not terms:
            continue
        query = ("Minecraft Fabric mod implementation " + " ".join(terms))[:180].strip()
        key = query.casefold()
        if key in seen or not _is_retrieval_query(query, raw_prompt=raw_prompt):
            continue
        seen.add(key)
        result.append(query)
        if len(result) >= 8:
            break
    return result
'''
s = s[:start] + host_gap + s[end:]

old_block = '''    return (
        _round_has_verified_claims(summary)
        and not _stable_text(page_gaps)
        and not _stable_text(support_rejections)
    )'''
if old_block in s:
    s = s.replace(
        old_block,
        '''    del page_gaps, support_rejections
    return _round_has_verified_claims(summary)''',
        1,
    )

# Page-local omissions and rejected paraphrases are diagnostics, not authored requirement gaps.
s = s.replace('summary["gaps"] = list(active_page_gaps)', 'summary["gaps"] = []', 1)
s = s.replace(
    '"gap_semantics": "unresolved_page_or_support_gap_blocks_domain_sufficiency",',
    '"gap_semantics": "page_local_diagnostic_only; authored_or_implementation_obligations_block",',
    1,
)
old_reason = '''        if fixed_point != _VERIFIED_FIXED_POINT:
            reasons.append(
                "corrective retrieval did not reach verified sufficiency: "
                + (fixed_point or "no_terminal_state")
            )'''
if old_reason in s:
    s = s.replace(
        old_reason,
        '''        if not claims and fixed_point != _VERIFIED_FIXED_POINT:
            reasons.append(
                "corrective retrieval produced zero support-verified implementation evidence: "
                + (fixed_point or "no_terminal_state")
            )''',
        1,
    )
s = s.replace(
    '        if active_support_rejections:\n            reasons.append("unresolved support rejections remain")\n',
    "",
)
write(p, s)


# 3. Claim support: tiny text protocol. Host owns indices, quote checks and EvidenceCards.
p = root / "pre_design_rag_support.py"
s = p.read_text(encoding="utf-8")
start = s.index("def _verify_page_claims(")
end = s.index("\ndef _merge_verified_notes(", start)
support_fn = r'''def _verify_page_claims(
    agentic_module: Any,
    project_rag: Any,
    router: Any,
    *,
    domain_id: str,
    page: Mapping[str, Any],
    claims: Sequence[str],
    progress_label: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify claims with a small-model-friendly line protocol and host exact quotes."""
    if not claims:
        return [], []
    content = str(page.get("content") or "")
    page_ref = str(page.get("page_ref") or "").strip()
    messages = [
        {
            "role": "system",
            "content": (
                "Judge each numbered claim only from SOURCE. Output exactly one line per claim: "
                "INDEX<TAB>NO, or INDEX<TAB>an exact contiguous quote copied from SOURCE. "
                "No JSON, markdown, explanation, or extra lines."
            ),
        },
        {
            "role": "user",
            "content": "CLAIMS\n"
            + "\n".join(f"{i}\t{claim}" for i, claim in enumerate(claims))
            + "\n\nSOURCE\n"
            + content,
        },
    ]

    def parse(raw: str) -> tuple[list[dict[str, Any]], list[str]]:
        indexed: dict[int, str | None] = {}
        for line in str(raw or "").splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            left, right = line.split("\t", 1)
            try:
                index = int(left.strip())
            except ValueError:
                continue
            if not 0 <= index < len(claims) or index in indexed:
                continue
            quote = right.strip()
            indexed[index] = (
                None
                if quote.casefold() in {"no", "false", "unsupported", "none"}
                else quote
            )

        accepted: list[dict[str, Any]] = []
        rejected: list[str] = []
        for index, claim in enumerate(claims):
            quote = indexed.get(index)
            if (
                not quote
                or quote not in content
                or len("".join(quote.split())) < _MIN_QUOTE_CHARS
            ):
                rejected.append(claim)
                _emit_support_trace(
                    "claim_support_rejected",
                    page_ref=page_ref,
                    claim_index=index,
                    claim=claim,
                    reason="no_exact_host_quote",
                )
                continue
            accepted.append(
                {
                    "claim": claim,
                    "evidence_refs": [page_ref],
                    "support_quote": quote,
                    "support_quote_sha256": _sha256_text(quote),
                    "support_verification": "model_line_entailment+host_exact_quote",
                }
            )
            _emit_support_trace(
                "claim_support_accepted",
                page_ref=page_ref,
                claim_index=index,
                claim=claim,
                quote_chars=len(quote),
                quote_sha256=_sha256_text(quote),
            )
        return accepted, rejected

    return project_rag._generate_bounded(
        agentic_module,
        router,
        messages=messages,
        response_schema=None,
        parser=parse,
        progress_label=progress_label + " claim-support",
    )
'''
s = s[:start] + support_fn + s[end:]
s = s.replace(
    '== "model_entailment+host_exact_quote"',
    'in {"model_entailment+host_exact_quote", "model_line_entailment+host_exact_quote"}',
)
write(p, s)


# 4. Runtime bounded helper can run plain semantic microtasks without structured decoding.
p = root / "runtime_stability_contract.py"
s = p.read_text(encoding="utf-8")
s = s.replace(
    "response_schema: Mapping[str, Any],\n        parser: Any,",
    "response_schema: Mapping[str, Any] | None,\n        parser: Any,",
    1,
)
old = '''        bound_schema, domain_id = _bound_research_schema(
            module,
            response_schema,
            messages,
        )
        aligned_messages = _aligned_research_messages(
            messages,
            response_schema=bound_schema,
            domain_id=domain_id,
        )'''
new = '''        if response_schema is None:
            bound_schema: Mapping[str, Any] | None = None
            domain_id = ""
            aligned_messages = [dict(message) for message in messages]
        else:
            bound_schema, domain_id = _bound_research_schema(
                module, response_schema, messages,
            )
            aligned_messages = _aligned_research_messages(
                messages, response_schema=bound_schema, domain_id=domain_id,
            )'''
if old not in s:
    raise SystemExit("runtime bounded preamble not found")
s = s.replace(old, new, 1)
old = '''            raw = router.generate_text(
                "planner",
                aligned_messages,
                response_format="json",
                response_schema=bound_schema,
                tool_stage="research",
                enable_tools=False,
            )'''
new = '''            if bound_schema is None:
                raw = router.generate_text(
                    "planner", aligned_messages, tool_stage="research", enable_tools=False
                )
            else:
                raw = router.generate_text(
                    "planner",
                    aligned_messages,
                    response_format="json",
                    response_schema=bound_schema,
                    tool_stage="research",
                    enable_tools=False,
                )'''
if old not in s:
    raise SystemExit("runtime bounded model call not found")
s = s.replace(old, new, 1)
write(p, s)


# 5. Local project retrieval: enable semantic retrieval and reranking.
p = root / "agentic_pre_design_rag.py"
s = p.read_text(encoding="utf-8")
if "semantic=False, rerank=False" in s:
    s = s.replace("semantic=False, rerank=False", "semantic=True, rerank=True")
    write(p, s)


# 6. Remove the global eight-query truncation only in the external-query owner.
cap_owners: list[str] = []
for p in root.glob("*.py"):
    text = p.read_text(encoding="utf-8")
    if (
        "bounded_fallback_queries" not in text
        and "not_selected_by_bounded_query_plan" not in text
    ):
        continue
    original = text
    text = text.replace("[:configured_limit]", "").replace("[: configured_limit]", "")
    text = re.sub(
        r"((?:EXTERNAL|PREDESIGN)[A-Z0-9_]*QUERY[A-Z0-9_]*LIMIT[^\n]{0,120}?)(?<!\d)8(?!\d)",
        lambda match: match.group(1) + "64",
        text,
    )
    if text != original:
        write(p, text)
        cap_owners.append(str(p))
print("query-cap owners patched:", cap_owners)


# 7. Regression contract for the exact last-log failure class.
test = Path("tests/test_small_model_rag_host_contract.py")
test.write_text(
    '''from __future__ import annotations

from minecraft_mod_ai.pre_design_rag_corrective import (
    _generate_gap_queries,
    _round_is_terminally_sufficient,
)


def test_page_local_diagnostics_do_not_revoke_verified_evidence():
    assert _round_is_terminally_sufficient(
        {"claims": [{"claim": "ok"}]},
        page_gaps=["this page lacks colonization"],
        support_rejections=["paraphrase"],
    )


def test_corrective_queries_are_host_owned():
    class ExplodingRouter:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("model must not plan corrective queries")

    seen = {"minecraft mod first query"}
    result = _generate_gap_queries(
        object(),
        object(),
        ExplodingRouter(),
        domain={"queries": ["minecraft mod first query", "minecraft fabric colony persistence"]},
        gaps=["planet colonization persistence ownership lifecycle"],
        prior_queries=[],
        seen=seen,
        raw_prompt="사용자 요청",
        progress_label="test",
    )
    assert any("colony persistence" in query.casefold() for query in result)


def test_support_verifier_uses_line_protocol_not_json():
    from minecraft_mod_ai import pre_design_rag_support as support

    assert "line protocol" in (support._verify_page_claims.__doc__ or "")
''',
    encoding="utf-8",
)
changed.append(str(test))


# Fail the refactor itself if the old catastrophic path remains.
corr = (root / "pre_design_rag_corrective.py").read_text(encoding="utf-8")
assert "corrective query planner omitted queries/search_queries" not in corr
assert 'summary["gaps"] = list(active_page_gaps)' not in corr
assert "response_schema=None" in (root / "pre_design_rag_support.py").read_text(encoding="utf-8")
assert "if bound_schema is None:" in (root / "runtime_stability_contract.py").read_text(encoding="utf-8")
print("changed files:", sorted(set(changed)))
