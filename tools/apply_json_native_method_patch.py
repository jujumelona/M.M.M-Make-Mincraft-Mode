from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "minecraft_mod_ai" / "mod_development_methods.py"


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}.")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from dataclasses import asdict, dataclass\n",
        "import json\nfrom dataclasses import asdict, dataclass\n",
    )
    text = replace_once(
        text,
        '        "methods": [asdict(method) for method in ordered],\n',
        '        "methods": _json_native_methods(ordered),\n',
    )
    text = replace_once(
        text,
        '        "methods": [asdict(methods[key]) for key in sorted(methods)],\n',
        '        "methods": _json_native_methods(\n'
        '            [methods[key] for key in sorted(methods)]\n'
        '        ),\n',
    )
    marker = "\ndef mod_development_method_catalog() -> dict[str, Any]:\n"
    helper = '''\ndef _json_native_methods(\n    methods: list[ModDevelopmentMethod] | tuple[ModDevelopmentMethod, ...],\n) -> list[dict[str, Any]]:\n    """Return stable JSON-native records; tuples must not leak into contracts."""\n\n    return json.loads(\n        json.dumps(\n            [asdict(method) for method in methods],\n            ensure_ascii=False,\n            allow_nan=False,\n        )\n    )\n\n\ndef mod_development_method_catalog() -> dict[str, Any]:\n'''
    if "def _json_native_methods(" not in text:
        if marker not in text:
            raise RuntimeError("Catalog insertion point was not found.")
        text = text.replace(marker, helper, 1)
    TARGET.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
