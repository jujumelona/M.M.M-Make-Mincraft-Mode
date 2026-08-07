from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "minecraft_mod_ai" / "mod_development_methods.py"

OLD = '''            "dungeon",
            "던전",
            "template pool",
'''
NEW = '''            "dungeon",
            "던전",
            "village",
            "마을",
            "town",
            "template pool",
'''


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if NEW in text:
        return
    if OLD not in text:
        raise RuntimeError("Expected worldgen alias block was not found.")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
