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
replace_once(
    "tests/test_large_request_ingestion.py",
    'def test_malformed_large_request_page_fails_closed_after_local_repair() -> None:\n'
    '    prompt = "first requirement " + ("bounded filler " * 2500) + "last requirement"\n'
    '    with pytest.raises(\n'
    '        SpecValidationError,\n'
    '        match=r"page 2/.*failed after one page-local repair",\n'
    '    ):\n'
    '        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan(prompt)\n',
    'def test_malformed_large_request_page_fails_closed_at_exact_no_progress() -> None:\n'
    '    prompt = "first requirement " + ("bounded filler " * 2500) + "last requirement"\n'
    '    with pytest.raises(\n'
    '        SpecValidationError,\n'
    '        match=r"exact no-progress cycle",\n'
    '    ):\n'
    '        GameDesignPlanner(_MalformedSecondDesignPageRouter()).plan(prompt)\n',
)

for name in (
    "tests/test_game_design_router.py",
    "tests/test_large_request_ingestion.py",
):
    target = Path(name)
    target.write_text(target.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")

print("game-design hardening follow-up applied")
