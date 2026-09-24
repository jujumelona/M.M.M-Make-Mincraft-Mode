"""An initialization latch without behavior must not satisfy an authored unit."""

import pytest

from minecraft_mod_ai.authored_feature_source import authored_feature_source_diagnostics
from minecraft_mod_ai.final_artifact import _authored_feature_semantic_findings

TARGET = "src/main/java/demo/AuthoredFeature002.java"


def source_for(body, *, fields="private static boolean inited = false;", members=""):
    return f"""package demo;
public final class AuthoredFeature002 {{
    {fields}
    private AuthoredFeature002() {{}}
    public static void initialize() {{ {body} }}
    {members}
}}
"""


@pytest.mark.parametrize("body, fields", [
    ("if (inited) return; inited = true; // No external dependencies\n",
     "private static boolean inited = false;"),
    ("if (!inited) { /* registration will go here */ inited = true; }",
     "private static boolean inited;"),
    ("synchronized (lock) { if (!inited) { inited = true; } }",
     "private static boolean inited = false; private static final Object lock = new Object();"),
])
def test_guard_only_unit_fails_generation_and_release(tmp_path, body, fields):
    source = source_for(body, fields=fields)
    diagnostics = authored_feature_source_diagnostics(
        source, path=TARGET, symbol="AuthoredFeature002"
    )
    assert any(item["code"] == "host:authored-empty" for item in diagnostics)
    path = tmp_path / TARGET
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    findings = _authored_feature_semantic_findings(tmp_path, [
        {"module_id": "authored_feature_002", "path": TARGET, "symbol": "AuthoredFeature002"}
    ])
    assert findings


@pytest.mark.parametrize("body, fields, members", [
    ("if (inited) return; inited = true; register();",
     "private static boolean inited;", "private static void register() {}"),
    ("inited = true;", "private static boolean inited;",
     "public static boolean ready() { return inited; }"),
    ("ready = true;", "public static boolean ready;", ""),
    ("miningRange = 5;", "private static int miningRange;",
     "public static int range() { return miningRange; }"),
    ("if (inited) return; inited = true; System.out.println(\"ready\");",
     "private static boolean inited;", ""),
    ("if (inited) return; inited = true;", "private static volatile boolean inited;", ""),
    ("if (inited) return; inited = true;", "volatile private static boolean inited;", ""),
])
def test_behavior_or_observable_state_is_not_guard_only(body, fields, members):
    assert not authored_feature_source_diagnostics(
        source_for(body, fields=fields, members=members),
        path=TARGET, symbol="AuthoredFeature002",
    )
