from __future__ import annotations

import pytest

from minecraft_mod_ai.target_profile_semantics import (
    mappings_applicable,
    minecraft_version_tuple,
    minimum_java_major,
    uses_native_names,
)


def test_native_name_boundary_is_owned_once() -> None:
    assert mappings_applicable("1.21.4") is True
    assert uses_native_names("26.0") is False
    assert uses_native_names("26.1") is True
    assert uses_native_names("26.2") is True


def test_java_major_is_derived_from_same_target_semantics() -> None:
    assert minimum_java_major("1.20.4") is None
    assert minimum_java_major("1.20.5") == 21
    assert minimum_java_major("1.21.4") == 21
    assert minimum_java_major("26.1") == 25
    assert minimum_java_major("26.2") == 25


def test_version_parser_rejects_unparseable_target_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="TARGET_MINECRAFT_VERSION"):
        minecraft_version_tuple("latest")
