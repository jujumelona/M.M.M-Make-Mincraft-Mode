"""Focused compile repair changes exactly one located method, not a class graph."""
from minecraft_mod_ai import focused_method_compile_repair as focused

TARGET = "src/main/java/demo/Alien.java"
SOURCE = """package demo;
public class Alien {
    public void attack() {
        MissingAPI.call();
    }

    public int getHealth() {
        return 10;
    }
}
"""


def test_focused_repair_edits_only_one_method(tmp_path, monkeypatch):
    root = tmp_path
    target = root / TARGET
    target.parent.mkdir(parents=True)
    target.write_text(SOURCE, encoding="utf-8")
    invoked = []

    def fake_compile(root, *, target_path, edit, max_repairs):
        assert target_path == TARGET and max_repairs == 2
        updated = edit(SOURCE, ({"path": TARGET, "line": 4,
                                 "code": "cant.resolve", "message": "MissingAPI"},), 1)
        assert "MissingAPI" not in updated
        assert "getHealth()" in updated and "return 10;" in updated
        return {"status": "COMPILED", "source": updated}

    monkeypatch.setattr(focused, "compile_and_repair_feature", fake_compile)

    class Router:
        def generate_java_source(self, role, messages, *, source_frame, **kwargs):
            invoked.append((role, messages))
            assert role == "coder"
            assert "void attack()" in source_frame["prefix"]
            return (source_frame["prefix"]
                    + "System.out.println(1);"
                    + source_frame["suffix"])

    result = focused.repair_staged_java_with_model(Router(), root,
                                                   target_path=TARGET)
    assert result["status"] == "COMPILED"
    assert len(invoked) == 1


def test_unsupported_error_without_method_does_not_summon_model(tmp_path, monkeypatch):
    file = tmp_path / TARGET
    file.parent.mkdir(parents=True)
    file.write_text(SOURCE, encoding="utf-8")

    def fake_compile(root, *, target_path, edit, max_repairs):
        assert edit(SOURCE, ({"line": 99, "message": "outside"},), 1) == SOURCE
        return {"status": "NO_PROGRESS"}

    monkeypatch.setattr(focused, "compile_and_repair_feature", fake_compile)

    class Router:
        def generate_java_source(self, *_args, **_kwargs):
            raise AssertionError("Model may not repair unrelated source")

    assert focused.repair_staged_java_with_model(
        Router(), tmp_path, target_path=TARGET)["status"] == "NO_PROGRESS"
