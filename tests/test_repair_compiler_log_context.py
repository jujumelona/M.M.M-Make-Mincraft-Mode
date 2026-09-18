from __future__ import annotations

from minecraft_mod_ai import repair_engine


def test_repair_log_keeps_javac_error_ahead_of_long_stacktrace(tmp_path) -> None:
    log = tmp_path / "gradle-build.log"
    source = tmp_path / "project/src/main/java/demo/DebugToken.java"
    log.write_text(
        "\n".join(
            [
                "> Task :compileJava",
                f"  {source}:4: error: package net.minecraft.item does not exist",
                "import net.minecraft.item.Item;",
                "                         ^",
                "1 error",
                *(f"at org.gradle.synthetic.Frame{i}(Frame.java:{i})" for i in range(5000)),
            ]
        ),
        encoding="utf-8",
    )

    excerpt = repair_engine._read_bounded_build_log(log)

    assert "DebugToken.java:4: error: package net.minecraft.item does not exist" in excerpt
    assert "import net.minecraft.item.Item;" in excerpt
    assert len(excerpt) <= repair_engine._REPAIR_LOG_SNIPPET_CHARS
