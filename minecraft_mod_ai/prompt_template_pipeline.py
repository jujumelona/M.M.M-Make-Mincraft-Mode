"""Prompt-only semantic tasks, assembled by the host into the planning boundary."""

from .bounded_record_template import run_record_template
from .parallel_model_tasks import deterministic_model_map, serialized_callback
from .planner_operation import planner_operation
from .task_template_catalog import load_template
from .task_value_runner import run_value_template


def extract_prompt_records(router, prompt, *, progress=None, checkpoint=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("PROMPT_INPUT: original prompt must be a non-empty string")

    context = {"original_prompt": prompt}
    identifiers = tuple(load_template("prompt/workflow")["steps"])
    safe_checkpoint = serialized_callback(checkpoint)

    def run_identifier(identifier):
        template = load_template(identifier)
        with planner_operation(identifier):
            if template["execution"] == "records":
                result = run_record_template(
                    router,
                    identifier,
                    context=context,
                    allowed_refs=set(),
                    progress=progress,
                    checkpoint=safe_checkpoint,
                )
                return result["records"]
            return run_value_template(
                router,
                identifier,
                context=context,
                progress=progress,
                checkpoint=safe_checkpoint,
            )

    values = deterministic_model_map(
        router,
        identifiers,
        run_identifier,
        role="planner",
        thread_name_prefix="prompt-template",
    )
    results = dict(zip(identifiers, values))

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
