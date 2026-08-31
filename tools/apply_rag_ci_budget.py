from pathlib import Path

path = Path('.github/workflows/ci.yml')
text = path.read_text(encoding='utf-8')
old = '''# Reviewed surface: inherited main was already at 424; Worker 03 adds one\n# requirement-traceability owner, bringing the exact audited surface to 425.\n# Any new runtime rebinding is a hard CI failure until deliberately reviewed.\nbudget = 425'''
new = '''# Reviewed surface: inherited main was already at 424; Worker 03 adds one\n# requirement-traceability owner and pre-design external source acquisition adds\n# one explicit _forced_rag_bundle owner, bringing the audited surface to 426.\n# Any new runtime rebinding is a hard CI failure until deliberately reviewed.\nbudget = 426'''
if text.count(old) != 1:
    raise SystemExit(f'expected one reviewed mutation budget block, found {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('reviewed RAG mutation budget updated to 426')
