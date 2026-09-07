"""Structured design fixtures shared by planning integration tests."""
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def specification(section):
    return {
        **{
            concern: [{field: f"{section} {concern} {field}: server owns the observable outcome."
                       for field in columns.split()}]
            for concern, columns in DETAIL_RECORDS[section].items()
        },
        "inapplicable_concerns": [],
    }


def row(section, refs=()):
    return {"specification": specification(section), "constraint_evidence_refs": list(refs)}
