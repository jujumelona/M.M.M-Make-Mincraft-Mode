from __future__ import annotations

from minecraft_mod_ai.internal_package_preflight import (
    validate_internal_package_integrity,
)


def test_current_package_has_no_broken_internal_module_imports() -> None:
    receipt = validate_internal_package_integrity()

    assert receipt["status"] == "PASS"
    assert receipt["python_file_count"] > 0
    assert receipt["internal_import_count"] > 0
