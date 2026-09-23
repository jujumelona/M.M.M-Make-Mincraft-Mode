from __future__ import annotations

from minecraft_mod_ai import repair_engine


def test_runtime_linkage_diagnostics_survive_gradle_tail_and_reach_repair_context(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from minecraft_mod_ai import repair_approved_reuse_context as reuse
    from minecraft_mod_ai import repository_grounding
    from minecraft_mod_ai import research_evidence_handoff_contract as handoff

    source = tmp_path / "src/main/java/demo/Feature.java"
    source.parent.mkdir(parents=True)
    source.write_text("package demo; public class Feature {}", encoding="utf-8")
    log = tmp_path / "build.log"
    cause = "Caused by: java.lang.NoSuchMethodError: 'void demo.Feature.initialize()'\n\tat demo.Main.onInitialize(Main.java:9)\n"
    log.write_text(cause + "\tat org.gradle.synthetic.Frame.run(Frame.java:1)\n" * 1500 + "BUILD FAILED in 29s\n", encoding="utf-8")
    evidence = {"passed": False, "diagnostics": {"diagnostics": []}, "build": {"status": "FAIL", "commands": [{"exit_code": 1, "log_path": str(log)}]}}
    first_signature = repair_engine.RepairEngine._signature(evidence)
    log.write_text(log.read_text(encoding="utf-8").replace("29s", "31s"), encoding="utf-8")
    assert repair_engine.RepairEngine._signature(evidence) == first_signature
    observed = {}

    def grounding(_router, _index, **kwargs):
        observed.update(kwargs)
        return {"manifest": {}, "relevant": {"files": []}}

    monkeypatch.setattr(repository_grounding, "build_repair_repository_context", grounding)
    monkeypatch.setattr(handoff, "_fresh_official_repair_evidence", lambda *_: {})
    monkeypatch.setattr(reuse, "build_approved_repair_reuse_context", lambda *_: None)
    engine = repair_engine.RepairEngine(router=SimpleNamespace(), gradle_cache=tmp_path / "cache")
    context = engine._context(tmp_path, evidence)
    assert "src/main/java/demo/Feature.java" in observed["diagnostic_paths"]
    assert "NoSuchMethodError" in observed["query"]
    assert context["source_diagnostics"][0]["path"] == "src/main/java/demo/Feature.java"


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
