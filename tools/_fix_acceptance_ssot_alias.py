from pathlib import Path

path = Path('minecraft_mod_ai/evidence_first_planning.py')
text = path.read_text(encoding='utf-8')
old = "from .acceptance_contracts import is_public_acceptance\n\n_is_public_acceptance = is_public_acceptance\n"
new = "from .acceptance_contracts import is_public_acceptance as _is_public_acceptance\n"
if old not in text:
    raise SystemExit('acceptance alias target not found')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
