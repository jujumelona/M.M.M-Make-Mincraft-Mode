import copy

import pytest


def model(root):
    return {'project_root': str(root), 'gradle_version': '8.6', 'source_sets': [
        {'id': ':main', 'project_path': ':', 'name': 'main',
         'source_roots': [str(root / 'src/main/java')], 'classpath': [],
         'output_dirs': [str(root / 'build/classes/java/main')],
         'java_home': str(root), 'source_compatibility': '17', 'target_compatibility': '17',
         'release': 17, 'compiler_args': [], 'annotation_processor_path': []}], 'mappings': {}}


def test_resolved_model_is_immutable_and_content_bound(tmp_path):
    from minecraft_mod_ai.project_model import ResolvedBuildModel

    raw = model(tmp_path)
    resolved = ResolvedBuildModel.from_dict(raw)
    original = resolved.model_id
    raw['source_sets'][0]['classpath'].append('untrusted')
    assert resolved.source_sets[0].classpath == ()
    assert resolved.model_id == original
    changed = model(tmp_path)
    changed['source_sets'][0]['release'] = 11
    assert ResolvedBuildModel.from_dict(changed).model_id != original


def test_missing_or_duplicate_source_sets_fail_closed(tmp_path):
    from minecraft_mod_ai.project_model import ResolvedBuildModel

    raw = model(tmp_path)
    raw['source_sets'].append(copy.deepcopy(raw['source_sets'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        ResolvedBuildModel.from_dict(raw)
    raw = model(tmp_path)
    del raw['source_sets'][0]['java_home']
    with pytest.raises(ValueError, match='java_home'):
        ResolvedBuildModel.from_dict(raw)


def test_model_inputs_ignore_java_but_detect_external_gradle_changes(tmp_path):
    from minecraft_mod_ai.project_model import ProjectModelInputs

    build = tmp_path / 'build.gradle'
    build.write_text("plugins { id 'java' }")
    inputs = ProjectModelInputs(tmp_path)
    first = inputs.revision()
    (tmp_path / 'A.java').write_text('class A {}')
    assert inputs.revision() == first
    build.write_text("plugins { id 'java-library' }")
    assert inputs.revision() != first


def test_model_inputs_cover_nested_build_logic_and_wrapper(tmp_path):
    from minecraft_mod_ai.project_model import ProjectModelInputs

    inputs = ProjectModelInputs(tmp_path)
    previous = inputs.revision()
    for relative in ('sub/build.gradle.kts', 'buildSrc/src/main/java/Plugin.java',
                     'gradle/wrapper/gradle-wrapper.properties', 'gradle/libs.versions.toml'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('changed')
        current = inputs.revision()
        assert current != previous
        previous = current
