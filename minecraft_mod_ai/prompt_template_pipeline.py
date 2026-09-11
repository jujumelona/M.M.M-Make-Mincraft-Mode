"""Prompt-only semantic tasks, assembled by the host into the planning boundary."""
from .bounded_record_template import run_record_template
from .planner_operation import planner_operation
from .task_template_catalog import load_template
from .task_value_runner import run_value_template


def extract_prompt_records(router, prompt, *, progress=None, checkpoint=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("PROMPT_INPUT: original prompt must be a non-empty string")

    context = {"original_prompt": prompt}
    results = {}
    for identifier in load_template("prompt/workflow")["steps"]:
        template = load_template(identifier)
        with planner_operation(identifier):
            if template["execution"] == "records":
                result = run_record_template(
                    router,
                    identifier,
                    context=context,
                    allowed_refs=set(),
                    progress=progress,
                    checkpoint=checkpoint,
                )
                results[identifier] = result["records"]
            else:
                results[identifier] = run_value_template(
                    router,
                    identifier,
                    context=context,
                    progress=progress,
                    checkpoint=checkpoint,
                )

    known = []
    for record in (
        results["prompt/parse"]
        + results["prompt/constraints"]
        + results["prompt/output_requirements"]
    ):
        if record not in known:
            known.append(record)
    return {
        "goal": results["prompt/intent"],
        "known": known,
        "references": results["prompt/entity_resolution"],
        "scope_status": results["prompt/scope"]["scope_status"],
        "unresolved": results["prompt/ambiguities"],
    }
