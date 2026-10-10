"""Protect generated initializer integration from duplicate registration calls."""
from __future__ import annotations

from minecraft_mod_ai.project_edit import _insert_initializer_call


SOURCE = """package example;
public class ExampleMod {
    public void onInitialize() {
        ModBlocks.initialize();
    }
}
"""


def test_initializer_does_not_duplicate_existing_call_under_new_marker() -> None:
    updated, inserted = _insert_initializer_call(
        SOURCE, "ModBlocks.initialize();", "another-stage",
    )
    assert inserted is True
    assert updated == SOURCE
    assert updated.count("ModBlocks.initialize();") == 1


def test_different_initializers_are_retained_once_each() -> None:
    first, inserted = _insert_initializer_call(
        SOURCE, "ModItems.initialize();", "items",
    )
    assert inserted is True
    assert first.count("ModItems.initialize();") == 1
    second, inserted = _insert_initializer_call(
        first, "ModItems.initialize();", "resources-items",
    )
    assert inserted is True
    assert second.count("ModItems.initialize();") == 1
    assert second.count("ModBlocks.initialize();") == 1


def test_call_in_another_method_does_not_hide_missing_main_initializer() -> None:
    source = """package example;
public class ExampleMod {
    void helper() { ModItems.initialize(); }
    public void onInitialize() {
        ModBlocks.initialize();
    }
}
"""
    updated, inserted = _insert_initializer_call(
        source, "ModItems.initialize();", "items",
    )
    assert inserted is True
    assert updated.count("ModItems.initialize();") == 2
