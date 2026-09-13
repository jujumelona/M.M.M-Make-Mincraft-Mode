from tools.model_realistic_replay_matrix import discover


def test_every_realistic_replay_behavior_is_an_independent_matrix_cell():
    files = discover()
    assert len(files) >= 17
    assert len(files) == len(set(files))
    assert all(path.startswith("tests/model_realistic_replay/test_") for path in files)


def test_replay_tests_are_not_accidentally_invisible_to_recursive_pytest():
    files = discover()
    assert all(path.endswith(".py") for path in files)
