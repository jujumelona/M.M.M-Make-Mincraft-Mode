from __future__ import annotations

import pytest

from minecraft_mod_ai.java_region_parser import (
    JavaRegionParseError,
    admit_initialize_region,
    admit_member_region,
    class_body_chunks,
    class_body_direct_return_calls,
    class_body_member_contracts,
    class_body_method_invocation_details,
    class_body_object_creations,
    public_source_member_contracts,
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



def test_tree_sitter_canonicalizes_leading_jdk_imports_in_member_region() -> None:
    source = (
        "import java.util.Map;\n"
        "import java.util.HashMap;\n"
        "import java.util.List;\n"
        "import java.util.ArrayList;\n"
        "import java.util.concurrent.atomic.AtomicReference;\n"
        "import java.util.concurrent.locks.ReentrantLock;\n\n"
        "private static final AtomicReference<String> RETAINED_STATE = new AtomicReference<>();\n"
        "private static final ReentrantLock LOCK = new ReentrantLock();\n"
        "private static final Map<String, Object> CHECKPOINT_DATA = new HashMap<>();\n"
        "public static List<String> snapshot(Map<String, Object> context) {\n"
        "    return new ArrayList<>();\n"
        "}"
    )

    admitted = admit_member_region(source)

    assert "import java." not in admitted
    assert "java.util.concurrent.atomic.AtomicReference<String>" in admitted
    assert "new java.util.concurrent.atomic.AtomicReference<>()" in admitted
    assert "java.util.concurrent.locks.ReentrantLock" in admitted
    assert "java.util.Map<String, Object>" in admitted
    assert "new java.util.HashMap<>()" in admitted
    assert "java.util.List<String> snapshot(java.util.Map<String, Object> context)" in admitted
    assert "new java.util.ArrayList<>()" in admitted


def test_java_region_failure_categories_separate_syntax_from_scope() -> None:
    with pytest.raises(JavaRegionParseError) as syntax:
        admit_member_region(
            "private static final java.util.Map<String, Object> CACHE;\n"
            "private static { CACHE = new java.util.HashMap<>(); }"
        )
    assert syntax.value.category == "syntax"

    with pytest.raises(JavaRegionParseError) as scope:
        admit_member_region("package escaped;")
    assert scope.value.category == "scope"


def test_tree_sitter_rejects_raw_package_declaration_as_member_structure() -> None:
    with pytest.raises(JavaRegionParseError, match="package declaration"):
        admit_member_region("package escaped;")


def test_tree_sitter_recovers_package_outer_class_and_jdk_imports() -> None:
    admitted = admit_member_region(
        """
package accidental;
import java.util.concurrent.atomic.AtomicLong;

public class WrongOuter {
    private static final AtomicLong VALUE = new AtomicLong(0L);
    public static long next() { return VALUE.incrementAndGet(); }
}
"""
    )

    assert "package accidental" not in admitted
    assert "class WrongOuter" not in admitted
    assert "java.util.concurrent.atomic.AtomicLong VALUE" in admitted
    assert "VALUE.incrementAndGet()" in admitted


def test_tree_sitter_canonicalizes_jdk_import_static_receiver() -> None:
    admitted = admit_member_region(
        "import java.util.Objects;\n"
        "private static boolean same(Object left, Object right) { "
        "return Objects.equals(left, right); }"
    )

    assert "import java.util.Objects" not in admitted
    assert "java.util.Objects.equals(left, right)" in admitted


def test_tree_sitter_normalizes_logged_fenced_import_static_receiver_shape() -> None:
    admitted = admit_member_region(
        """```java
import java.util.Map;
import java.util.concurrent.TimeUnit;

private static final long SAVE_INTERVAL_TICKS = TimeUnit.SECONDS.toSeconds(60L);
private static final String SAVE_TRIGGER = "WorldSave";
```
"""
    )

    assert "```" not in admitted
    assert "import java." not in admitted
    assert "java.util.concurrent.TimeUnit.SECONDS.toSeconds(60L)" in admitted


@pytest.mark.parametrize(
    "source",
    [
        "import java.util.*;\nprivate static Map<String, Object> cache;",
        "import static java.util.Collections.emptyList;\n"
        "private static Object value() { return emptyList(); }",
        "import com.example.Widget;\nprivate static Widget widget;",
    ],
)
def test_tree_sitter_rejects_ambiguous_or_non_jdk_member_imports(source: str) -> None:
    with pytest.raises(JavaRegionParseError):
        admit_member_region(source)


def test_tree_sitter_can_treat_host_initialize_only_as_empty_when_explicitly_allowed() -> None:
    source = """```java
public static void initialize() {
    AuthoredResourcesUi.initializeState("entry_points", java.util.Map.of());
}
```"""
    assert admit_member_region(
        source,
        allow_host_initialize_only_empty=True,
    ) == ""
    with pytest.raises(JavaRegionParseError):
        admit_member_region(source)


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


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            "public static void initialize() { value = 1; }",
            "value = 1;",
        ),
        (
            "void initialize() { value = 2; }",
            "value = 2;",
        ),
        (
            (
                "java.util.Map<String, Object> state = new java.util.HashMap<>();\n"
                "public static void initialize() { state.clear(); }"
            ),
            "state.clear();",
        ),
        (
            (
                "private static final java.util.Map<String, Object> state = "
                "new java.util.HashMap<>();\n"
                "public static void initialize() { state.clear(); }"
            ),
            "state.clear();",
        ),
    ],
)
def test_initialize_recovery_accepts_predicted_wrapper_shapes(
    source: str,
    expected: str,
) -> None:
    admitted = admit_initialize_region(source)
    assert expected in admitted
    assert "void initialize()" not in admitted


def test_initialize_recovery_accepts_logged_comment_only_wrapper_as_noop() -> None:
    admitted = admit_initialize_region(
        """```java
public static void initialize() {
    // Responsibilities: Client renders UI, Server validates economic rules.
    // No initialization required for this concern.
}
```
"""
    )

    assert admitted == ""


def test_initialize_recovery_handles_logged_field_plus_wrapper_shape() -> None:
    admitted = admit_initialize_region(
        """
java.util.concurrent.locks.Lock lock =
    new java.util.concurrent.locks.ReentrantLock();
java.util.Map<String, Object> uiState = new java.util.HashMap<>();
java.util.Map<String, Object> dataState = new java.util.HashMap<>();

public static void initialize() {
    lock.lock();
    try {
        uiState.clear();
        dataState.clear();
    } finally {
        lock.unlock();
    }
}
"""
    )

    assert "java.util.concurrent.locks.Lock lock =" in admitted
    assert "java.util.Map<String, Object> uiState =" in admitted
    assert "public static void initialize()" not in admitted
    assert "lock.lock();" in admitted
    assert "lock.unlock();" in admitted


def test_initialize_recovery_localizes_field_only_modifiers() -> None:
    admitted = admit_initialize_region(
        (
            "private static final java.util.Map<String, Object> state = "
            "new java.util.HashMap<>();\n"
            "public static void initialize() { state.clear(); }"
        )
    )

    assert "private " not in admitted
    assert "static " not in admitted
    assert "final java.util.Map<String, Object> state" in admitted
    assert "state.clear();" in admitted


def test_initialize_recovery_handles_fenced_jdk_imports_and_static_receiver() -> None:
    admitted = admit_initialize_region(
        """```java
import java.util.Objects;
Object value = new Object();
public static void initialize() {
    Objects.requireNonNull(value);
}
```
"""
    )

    assert "```" not in admitted
    assert "import java.util.Objects" not in admitted
    assert "java.util.Objects.requireNonNull(value);" in admitted


def test_initialize_recovery_handles_outer_class_envelope() -> None:
    admitted = admit_initialize_region(
        """
package accidental;
import java.util.Objects;

public final class WrongOuter {
    private static Object value = new Object();

    public static void initialize() {
        Objects.requireNonNull(value);
    }
}
"""
    )

    assert "class WrongOuter" not in admitted
    assert "package accidental" not in admitted
    assert "java.util.Objects.requireNonNull(value);" in admitted


def test_initialize_recovery_rejects_unrelated_helper_method() -> None:
    with pytest.raises(JavaRegionParseError, match="structurally inadmissible"):
        admit_initialize_region(
            (
                "private static void helper() {}\n"
                "public static void initialize() { helper(); }"
            )
        )


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


def test_markdown_it_handles_empty_fence_info_without_index_error() -> None:
    output = """
reasoning before code

```
private static int bareFenceValue = 7;
```
"""
    admitted = admit_member_region(output)
    assert "bareFenceValue = 7" in admitted


def test_initialize_admission_uses_same_bare_fence_envelope_boundary() -> None:
    output = """
initialization follows

```
value = 1;
if (ready) { start(); }
```
"""
    admitted = admit_initialize_region(output)
    assert "value = 1;" in admitted
    assert "if (ready) { start(); }" in admitted
    assert "initialization follows" not in admitted


def test_markdown_it_ignores_explicit_non_java_fence_and_uses_java_candidate() -> None:
    output = """
```text
not java
```

```java
private static int javaValue = 9;
```
"""
    admitted = admit_member_region(output)
    assert "javaValue = 9" in admitted
    assert "not java" not in admitted


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


def test_tree_sitter_class_body_contracts_preserve_types_and_mutability() -> None:
    contracts = class_body_member_contracts(
        "private static final java.util.Map<String, Object> CACHE = "
        "new java.util.HashMap<>();\n"
        "public static java.util.Map<String, Object> snapshot(String key, int limit) "
        "{ return CACHE; }"
    )

    field = next(item for item in contracts if item["kind"] == "field")
    method = next(item for item in contracts if item["kind"] == "method")

    assert field["symbol"] == "CACHE"
    assert field["declared_type"] == "java.util.Map<String, Object>"
    assert field["static"] is True
    assert field["final"] is True
    assert field["mutable"] is False

    assert method["symbol"] == "snapshot"
    assert method["return_type"] == "java.util.Map<String, Object>"
    assert method["static"] is True
    assert method["parameters"] == [
        {"type": "String", "name": "key"},
        {"type": "int", "name": "limit"},
    ]


def test_tree_sitter_public_source_contracts_ignore_private_helpers() -> None:
    contracts = public_source_member_contracts(
        """
package example;

public final class Dependency {
    private static final Object INTERNAL = new Object();
    public static Object getState(String key) { return INTERNAL; }
    protected static java.util.Map<String, Object> copy(
            java.util.Map<String, Object> input) { return input; }
    private static void helper() {}
}
"""
    )

    assert [(item["kind"], item["symbol"]) for item in contracts] == [
        ("method", "getState"),
        ("method", "copy"),
    ]
    get_state = contracts[0]
    copy = contracts[1]
    assert get_state["return_type"] == "Object"
    assert get_state["parameters"] == [{"type": "String", "name": "key"}]
    assert copy["return_type"] == "java.util.Map<String, Object>"
    assert copy["parameters"] == [
        {"type": "java.util.Map<String, Object>", "name": "input"}
    ]


def test_tree_sitter_extracts_unqualified_direct_return_call() -> None:
    calls = class_body_direct_return_calls(
        "private static Object readState() { return null; }\n"
        "private static java.util.Map<String, Object> shipConfig() { "
        "return readState(); }"
    )

    assert calls == (
        {
            "method": "shipConfig",
            "declared_return_type": "java.util.Map<String, Object>",
            "receiver": "",
            "symbol": "readState",
            "argument_count": 0,
        },
    )


def test_tree_sitter_extracts_parenthesized_direct_return_call() -> None:
    calls = class_body_direct_return_calls(
        "private static Object readState() { return null; }\n"
        "private static String value() { return (readState()); }"
    )

    assert calls[0]["method"] == "value"
    assert calls[0]["symbol"] == "readState"


def test_tree_sitter_method_invocation_details_preserve_arguments_and_name_span() -> None:
    source = (
        'private static void edge(java.util.Map<String, Object> context) { '
        'AuthoredStateModel.applyUpdate("drift_action", "left", context); }'
    )
    calls = class_body_method_invocation_details(source)
    target = next(call for call in calls if call["symbol"] == "applyUpdate")

    assert target["receiver"] == "AuthoredStateModel"
    assert target["arguments"] == ('"drift_action"', '"left"', "context")
    encoded = source.encode("utf-8")
    assert encoded[target["name_start_byte"]:target["name_end_byte"]].decode("utf-8") == "applyUpdate"


def test_tree_sitter_extracts_object_creation_type_and_arity() -> None:
    creations = class_body_object_creations(
        "private static Object a = new Object();\n"
        "private static String b = new String(new char[]{'x'});"
    )

    assert creations == (
        {"type": "Object", "argument_count": 0},
        {"type": "String", "argument_count": 1},
    )
