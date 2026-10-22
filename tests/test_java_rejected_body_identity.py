"""A repeated rejected body is not a repair."""
import json

from minecraft_mod_ai.construction_source_repair import repeated_invalid_java_implementation


def test_invalid_multi_method_body_repeated_with_only_transport_newlines():
    broken = (
        "private void handleA() {}\n"
        "private void registerA() {}\n"
        "private void handleB() {}\n"
        "private void registerB() {}"
    )
    first = json.dumps({"implementation": broken + "\n"})
    second = json.dumps({"implementation": broken})
    assert repeated_invalid_java_implementation(first, second)


def test_semantically_different_revision_is_not_a_duplicate():
    first = json.dumps({"implementation": "return inventedVariable;"})
    second = json.dumps({"implementation": "return actualValue;"})
    assert not repeated_invalid_java_implementation(first, second)


def test_invalid_transport_still_preserved_for_diagnostic():
    assert repeated_invalid_java_implementation(' {"implementation": "return;"} ', '{"implementation":"return;"}')
