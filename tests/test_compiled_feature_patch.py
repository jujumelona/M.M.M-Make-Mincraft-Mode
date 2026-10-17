"""A repaired feature resumes without replaying stale or cross-version patches."""
from minecraft_mod_ai.compiled_feature_patch import (
    persist_compiled_repair, replay_compiled_repair,
)
import hashlib

TARGET = "src/main/java/demo/Alien.java"
BAD = "package demo; public class Alien { public void attack(){ Missing(); } }"
GOOD = "package demo; public class Alien { public void attack(){ System.out.println(1); } }"


def _setup(tmp_path):
    for relative, text in {
        "build.gradle": "plugins { id 'fabric-loom' }",
        "gradle.properties": "minecraft_version=26.2",
        "settings.gradle": "rootProject.name='alien'",
        "src/main/resources/fabric.mod.json": '{"id":"alienmod"}',
        TARGET: GOOD,
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_exact_compiled_source_can_replay_after_restart(tmp_path):
    _setup(tmp_path)
    receipt = persist_compiled_repair(
        tmp_path, TARGET, original_sha256=hashlib.sha256(BAD.encode()).hexdigest(),
        compiled_source=GOOD)
    assert receipt["runtime_verified"] is False
    assert replay_compiled_repair(tmp_path, TARGET, BAD) == GOOD
    assert replay_compiled_repair(tmp_path, TARGET, GOOD) == GOOD
    assert replay_compiled_repair(tmp_path, TARGET, BAD + " // changed") != GOOD


def test_changed_project_version_cannot_replay_patch(tmp_path):
    _setup(tmp_path)
    persist_compiled_repair(
        tmp_path, TARGET, original_sha256=hashlib.sha256(BAD.encode()).hexdigest(),
        compiled_source=GOOD)
    (tmp_path/"gradle.properties").write_text("minecraft_version=1.21.8")
    assert replay_compiled_repair(tmp_path, TARGET, BAD) == BAD
