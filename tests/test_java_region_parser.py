from __future__ import annotations

import pytest

from minecraft_mod_ai.java_region_parser import (
    JavaRegionParseError,
    admit_initialize_region,
    admit_member_region,
    class_body_chunks,
    strict_member_chunks,
)


def test_tree_sitter_accepts_direct_java_members() -> None:
    source = (
        "private static final int LIMIT = 30;\n"
        "private static boolean valid(int value) { return value >= 0; }"
    )
    admitted = admit_member_region(source)
    assert "private static final int LIMIT = 30;" in admitted
    assert "private static boolean valid(int value)" in admitted


def test_tree_sitter_unwraps_accidental_outer_class_and_drops_host_lifecycle() -> None:
    source = """
package accidental.wrapper;
import java.util.List;

public final class AccidentalOuter {
    private AccidentalOuter() {}

    private static final int LIMIT = 30;

    static {
        throw new AssertionError("host-owned lifecycle must be dropped");
    }

    public static void initialize() {
        throw new AssertionError("host-owned initialize must be dropped");
    }

    private static boolean valid(int value) {
        return value >= 0;
    }
}
"""
    admitted = admit_member_region(source)
    assert "package accidental.wrapper" not in admitted
    assert "import java.util.List" not in admitted
    assert "AccidentalOuter()" not in admitted
    assert "static {" not in admitted
    assert "initialize()" not in admitted
    assert "private static final int LIMIT = 30;" in admitted
    assert "private static boolean valid(int value)" in admitted


def test_tree_sitter_keeps_private_nested_runtime_type() -> None:
    admitted = admit_member_region(
        "private static final class ActorState { private int value; }"
    )
    assert "private static final class ActorState" in admitted


def test_tree_sitter_rejects_non_private_nested_type_when_it_has_no_outer_payload() -> None:
    with pytest.raises(JavaRegionParseError):
        admit_member_region("public class Escape {}")


def test_tree_sitter_rejects_malformed_java_with_precise_parse_failure() -> None:
    with pytest.raises(JavaRegionParseError, match="syntax|missing|admission|class-body"):
        admit_member_region("private static int broken = ;")


def test_tree_sitter_parses_initialize_statements_inside_host_method() -> None:
    admitted = admit_initialize_region(
        "value = 1;\nif (ready) { start(); }"
    )
    assert "value = 1;" in admitted
    assert "if (ready) { start(); }" in admitted


def test_tree_sitter_member_chunks_ignore_comments_without_losing_members() -> None:
    chunks = strict_member_chunks(
        "// comment\nprivate static int value = 1;\n/* comment */\n"
        "private static int read() { return value; }"
    )
    assert len(chunks) == 2
    assert chunks[0].startswith("private static int value")
    assert chunks[1].startswith("private static int read")


def test_parser_only_split_keeps_host_static_initializer_for_host_code() -> None:
    chunks = class_body_chunks(
        "private static int value;\nstatic { value = 1; }"
    )
    assert len(chunks) == 2
    assert chunks[0].startswith("private static int value")
    assert chunks[1].startswith("static {")


def test_model_admission_still_rejects_static_initializer() -> None:
    with pytest.raises(JavaRegionParseError, match="static_initializer"):
        strict_member_chunks("static { value = 1; }")


def test_markdown_it_selects_last_admissible_java_fence_from_reasoning_output() -> None:
    output = """
The user is asking me to implement entry_conditions. Let me analyze the requirements.

```java
public static void initialize() {
    throw new AssertionError("draft");
}
```

I need to reconsider the design and produce the final member region.

```java
private static final java.util.concurrent.locks.ReentrantLock entryConditionLock =
        new java.util.concurrent.locks.ReentrantLock();
private static boolean firstSmelterCompleted;

public static void initialize() {
    firstSmelterCompleted = false;
}

public static boolean isEntryConditionsMet() {
    return firstSmelterCompleted;
}
```
"""
    admitted = admit_member_region(output)
    assert "entryConditionLock" in admitted
    assert "firstSmelterCompleted" in admitted
    assert "isEntryConditionsMet" in admitted
    assert "initialize()" not in admitted
    assert "The user is asking" not in admitted


def test_markdown_it_prefers_last_valid_java_draft() -> None:
    output = """
```java
private static int draftValue = 1;
```

revising...

```java
private static int finalValue = 2;
```
"""
    admitted = admit_member_region(output)
    assert "finalValue = 2" in admitted
    assert "draftValue" not in admitted
