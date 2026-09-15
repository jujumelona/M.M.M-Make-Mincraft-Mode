from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from minecraft_mod_ai.task_template_catalog import load_record_template


_IDENTIFIER = "feature/algorithm/atomic_mutations"


def test_atomic_mutations_schema_rejects_logged_context_echo_shape() -> None:
    schema = load_record_template(_IDENTIFIER)["record_schema"]
    echoed_context = json.dumps(
        {
            "requirement_id": "req_004",
            "requirement": (
                "Players can upgrade spacecraft performance and acquire new weapons "
                "or crew through purchase or trade."
            ),
            "criterion": (
                "Players can improve spacecraft stats or add new modules via "
                "in-game transactions."
            ),
        },
        ensure_ascii=False,
    )

    for mutations in (echoed_context, "Context copy: " + echoed_context):
        candidate = {
            "mutations": mutations,
            "commit": "Apply the validated purchase atomically after payment succeeds.",
            "rollback": "Restore the pre-transaction spacecraft state if commit fails.",
        }
        errors = list(Draft202012Validator(schema).iter_errors(candidate))
        assert errors
        assert tuple(errors[0].absolute_path) == ("mutations",)
