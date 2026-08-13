from pathlib import Path

path = Path(".github/scripts/optimize_production_rag_structure.py")
text = path.read_text(encoding="utf-8")
old = "\\n\\nimport pytest\\n\\nfrom minecraft_mod_ai import production_tools"
new = "\\n\\nfrom minecraft_mod_ai import production_tools"
if old not in text:
    raise SystemExit("generated pytest import anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
