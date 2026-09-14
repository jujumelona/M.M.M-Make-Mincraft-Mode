import json

import pytest

from minecraft_mod_ai.project_edit import (
    FabricProjectInfo,
    ProjectEditError,
    ensure_client_entrypoint,
    ensure_dependency,
    inspect_fabric_project,
    write_text_files,
)


@pytest.fixture
def project(tmp_path):
    metadata = tmp_path / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        '{"id":"example","entrypoints":{"main":["example.Main"]}}',
        encoding="utf-8",
    )
    return FabricProjectInfo(
        tmp_path, "example", "example.Main", "example", "Main",
        tmp_path / "src/main/java/example/Main.java", metadata,
    )


@pytest.mark.parametrize("content", [
    '{"id":"example","id":"other","entrypoints":{"main":["example.Main"]}}',
    '{"id":"example","entrypoints":{"main":["example.Main"]},"custom":{"n":NaN}}',
])
def test_inspection_rejects_ambiguous_metadata(project, content):
    project.fabric_mod_json.write_text(content, encoding="utf-8")
    with pytest.raises(ProjectEditError):
        inspect_fabric_project(project.root)


@pytest.mark.parametrize("relative,content", [
    ("config/example.json", '{"a":1,"a":2}'),
    ("config/example.json", '{"nested":{"a":1,"a":2}}'),
    ("config/example.json", '{"a":Infinity}'),
    ("config/example.json", '{"a":1e999}'),
    ("src/main/resources/assets/example/lang/en_us.json", '{"key":4}'),
    ("src/main/resources/fabric.mod.json", '{"entrypoints":{"client":[4]}}'),
])
def test_invalid_resource_aborts_whole_write(project, relative, content):
    before = project.fabric_mod_json.read_bytes()
    with pytest.raises(ProjectEditError):
        write_text_files(project, {"other.txt": "not committed", relative: content}, replace_existing=True)
    assert not (project.root / "other.txt").exists()
    assert project.fabric_mod_json.read_bytes() == before


def test_json_writer_serializes_deterministically_and_is_idempotent(project):
    relative = "config/example.json"
    write_text_files(project, {relative: '{"z":2,"a":{"y":true,"b":"한글"}}'})
    assert (project.root / relative).read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": "한글",\n    "y": true\n  },\n  "z": 2\n}\n'
    )
    receipt = write_text_files(project, {relative: '{"a":{"b":"한글","y":true},"z":2}'})
    assert receipt["status"] == "UNCHANGED"


def test_client_edit_preserves_custom_fields_and_adapter_entrypoints(project):
    original = {
        "id": "example", "entrypoints": {"main": ["example.Main"],
        "client": [{"adapter": "custom", "value": "example.Client"}]},
        "custom": {"vendor": {"unknown": [True, None, 3]}},
    }
    project.fabric_mod_json.write_text(json.dumps(original), encoding="utf-8")
    ensure_client_entrypoint(project, entrypoint="example.NewClient")
    result = json.loads(project.fabric_mod_json.read_text(encoding="utf-8"))
    assert result["custom"] == original["custom"]
    assert result["entrypoints"]["client"] == [
        {"adapter": "custom", "value": "example.Client"}, "example.NewClient",
    ]
    assert ensure_client_entrypoint(project, entrypoint="example.NewClient")["status"] == "UNCHANGED"


def test_client_edit_rejects_duplicate_fields_before_mutation(project):
    content = '{"id":"example","custom":{"x":1,"x":2}}'
    project.fabric_mod_json.write_text(content, encoding="utf-8")
    with pytest.raises(ProjectEditError):
        ensure_client_entrypoint(project, entrypoint="example.Client")
    assert project.fabric_mod_json.read_text(encoding="utf-8") == content


REPOSITORY = "maven {\n name = 'GeckoLib'\n url = 'https://example.org/maven/'\n}"
DEPENDENCY = 'modImplementation("software.bernie.geckolib:geckolib-fabric-1.20.1:4.0.0")'


def test_gradle_dependency_uses_separate_fragment_not_deceptive_blocks(project):
    original = ('// dependencies { deceptive comment\n'
                'buildscript { repositories { mavenCentral() } }\n'
                'def description = "dependencies { quoted }"\n'
                'dependencies { implementation("example:existing:1") }\n')
    build = project.root / "build.gradle"
    build.write_text(original, encoding="utf-8")
    ensure_dependency(project, repository_block=REPOSITORY, dependency_line=DEPENDENCY, marker="geo/../lib")
    updated = build.read_text(encoding="utf-8")
    assert updated.startswith(original)
    fragments = list((project.root / "gradle/mmm-dependencies").glob("*.gradle"))
    assert len(fragments) == 1
    fragment = fragments[0].read_text(encoding="utf-8")
    assert 'modImplementation("software.bernie.geckolib:geckolib-fabric-1.20.1:4.0.0")' in fragment
    assert "https://example.org/maven/" in fragment
    assert fragments[0].relative_to(project.root).as_posix() in updated
    assert ensure_dependency(project, repository_block=REPOSITORY, dependency_line=DEPENDENCY,
                             marker="geo/../lib")["status"] == "UNCHANGED"
    assert build.read_text(encoding="utf-8") == updated


@pytest.mark.parametrize("dependency", [
    'modImplementation("g:a:1"); println("unexpected")',
    'modImplementation("g:a:${version}")',
    'modImplementation(files("local.jar"))',
])
def test_gradle_dependency_rejects_untyped_expressions_without_writes(project, dependency):
    build = project.root / "build.gradle"
    original = "dependencies {}\n"
    build.write_text(original, encoding="utf-8")
    with pytest.raises(ProjectEditError):
        ensure_dependency(project, repository_block=REPOSITORY, dependency_line=dependency, marker="geo")
    assert build.read_text(encoding="utf-8") == original
    assert not (project.root / "gradle/mmm-dependencies").exists()


@pytest.mark.parametrize("original", ["dependencies {\n", "/* unterminated", 'def a = "unterminated'])
def test_gradle_fragment_cannot_be_appended_inside_unclosed_syntax(project, original):
    build = project.root / "build.gradle"
    build.write_text(original, encoding="utf-8")
    with pytest.raises(ProjectEditError):
        ensure_dependency(project, repository_block=REPOSITORY, dependency_line=DEPENDENCY, marker="geo")
    assert build.read_text(encoding="utf-8") == original
