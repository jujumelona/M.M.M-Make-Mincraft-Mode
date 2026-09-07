"""Detailed planning operates on dependency-scoped JSON sections."""
import pytest
from minecraft_mod_ai import planning_state_implementation as implementation
from minecraft_mod_ai.planning_detail_template import CORE_WORKSHEET_SECTIONS


def test_structured_section_transport_failure_is_not_retried():
    calls = []
    class Router:
        def generate_text(self, *args, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("transport failure")
    with pytest.raises(RuntimeError, match="transport failure"):
        implementation._compile_worksheet_section(
            Router(), requirement={"requirement_id": "r", "statement": "Gather resources."},
            selected_sections=CORE_WORKSHEET_SECTIONS, section="behavior_contract",
            evidence=[], allowed=set(), completed={},
        )
    assert len(calls) == 1


def test_old_free_form_section_parser_does_not_exist():
    assert not hasattr(implementation, "_compile_requirement_specifications")
    assert not hasattr(implementation, "_normalize_section_text")
