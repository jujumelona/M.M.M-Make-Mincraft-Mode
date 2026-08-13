from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"unexpected source shape in {path}: {old!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/game_design.py",
    "from typing import Any, Sequence\n",
    "from typing import Any, Mapping, Sequence\n",
)
replace_once(
    "tests/test_game_design_router.py",
    "    _system_prompt,\n    _validate_design,\n",
    "    _system_prompt,\n",
)
print("game-design hardening lint follow-up applied")
