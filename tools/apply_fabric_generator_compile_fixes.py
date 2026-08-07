from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence in {path}, found {count}: {old!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "minecraft_mod_ai/geckolib_generator.py",
        ":state.stop()",
        ":PlayState.STOP",
    )
    replace_once(
        "minecraft_mod_ai/extended_content_generator.py",
        "return validateTicker(type, MACHINE_ENTITY_TYPE, GeneratedMachineBlockEntity::tick);",
        "return checkType(type, MACHINE_ENTITY_TYPE, GeneratedMachineBlockEntity::tick);",
    )
    replace_once(
        "minecraft_mod_ai/system_templates_quest.py",
        '''        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, blockEntity) ->
            progress(
                player,
                "break",
                Registries.BLOCK.getId(state.getBlock()).toString(),
                1
            )
        );''',
        '''        PlayerBlockBreakEvents.AFTER.register((world, player, pos, state, blockEntity) -> {{
            if (player instanceof ServerPlayerEntity serverPlayer) {{
                progress(
                    serverPlayer,
                    "break",
                    Registries.BLOCK.getId(state.getBlock()).toString(),
                    1
                );
            }}
        }});''',
    )


if __name__ == "__main__":
    main()
