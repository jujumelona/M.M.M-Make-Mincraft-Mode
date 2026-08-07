from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "minecraft_mod_ai" / "geckolib_generator.py"
OLD = (
    "import software.bernie.geckolib.core.animation.*;\\n"
    "import software.bernie.geckolib.util.GeckoLibUtil;"
)
NEW = (
    "import software.bernie.geckolib.core.animation.*;\\n"
    "import software.bernie.geckolib.core.object.PlayState;\\n"
    "import software.bernie.geckolib.util.GeckoLibUtil;"
)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if NEW in text:
        return
    if text.count(OLD) != 1:
        raise RuntimeError("Expected GeckoLib import block was not found exactly once.")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")


if __name__ == "__main__":
    main()
