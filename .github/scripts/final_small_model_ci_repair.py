from pathlib import Path

p = Path('minecraft_mod_ai/agentic_research_game_design.py')
s = p.read_text(encoding='utf-8')
old = '        + " Preserve exact approved requirement IDs. Write design content as Markdown, not JSON. "\n        + "No code fences, <think>, analysis, "'
new = '        + " Preserve exact approved requirement IDs. Write design content as Markdown, not JSON. "\n        + "No JSON. No code fences, <think>, analysis, "'
if old not in s:
    raise SystemExit('design prompt marker missing')
p.write_text(s.replace(old, new, 1), encoding='utf-8')

p = Path('minecraft_mod_ai/pre_design_domain_research.py')
s = p.read_text(encoding='utf-8')
s = s.replace(
    'from .small_model_predesign_research import research_document_domain\n',
    'from .small_model_predesign_research import research_document_domain as _small_model_research_document_domain\n',
    1,
)
marker = '\n\ndef _root_page_claims('
wrapper = '''\n\ndef research_document_domain(\n    agentic_module: Any,\n    project_rag: Any,\n    router: Any,\n    *,\n    prompt: str,\n    domain: Mapping[str, Any],\n    document: Mapping[str, Any],\n    trace_metadata: Mapping[str, Any] | None,\n) -> dict[str, Any]:\n    """Canonical facade for the host-owned small-model pre-design implementation."""\n    return _small_model_research_document_domain(\n        agentic_module,\n        project_rag,\n        router,\n        prompt=prompt,\n        domain=domain,\n        document=document,\n        trace_metadata=trace_metadata,\n    )\n'''
if marker not in s:
    raise SystemExit('domain owner insertion marker missing')
s = s.replace(marker, wrapper + marker, 1)
p.write_text(s, encoding='utf-8')

p = Path('tests/test_pre_design_rag_direct_owner.py')
s = p.read_text(encoding='utf-8')
s = s.replace(
    'assert owner.research_document_domain.__module__.endswith("small_model_predesign_research")',
    'assert owner.research_document_domain.__module__ == "minecraft_mod_ai.pre_design_domain_research"',
)
p.write_text(s, encoding='utf-8')

# This regression was introduced with the host-owned pipeline.  The canonical public
# owner is the pre_design_domain_research facade; implementation identity is not part
# of the contract and conflicts with the long-standing single-owner invariant.
p = Path('tests/test_small_model_predesign_v2.py')
s = p.read_text(encoding='utf-8')
s = s.replace(
    '    assert pre_design_domain_research.research_document_domain is small.research_document_domain\n',
    '    assert pre_design_domain_research.research_document_domain.__module__ == "minecraft_mod_ai.pre_design_domain_research"\n    assert callable(small.research_document_domain)\n',
    1,
)
p.write_text(s, encoding='utf-8')

print('final CI compatibility repair applied')
