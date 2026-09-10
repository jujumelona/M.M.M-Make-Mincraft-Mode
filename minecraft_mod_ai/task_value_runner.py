"""Execute a single host-owned value task with its exact input/output schemas."""
import json
from copy import deepcopy

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_template
from .task_template_input import task_binding, task_context


def run_value_template(router, identifier, *, context, progress=None, checkpoint=None):
    template = load_template(identifier)
    context = task_context(template, context)
    binding = task_binding(template, context)
    saved = (progress or {}).get(binding)
    if saved is not None:
        if not isinstance(saved, list) or len(saved) != 1:
            raise ValueError("TEMPLATE_PROGRESS: single-value task requires exactly one result")
        value = deepcopy(saved[0])
    elif template["execution"] == "host" and template.get("operation") == "identity":
        value = context
    elif template["execution"] == "value":
        value = generate_fixed_template_value(
            router, "planner",
            [{"role": "system", "content": template["task"] + "\n" + "\n".join(template["rules"])},
             {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
            response_schema=template["output_schema"], enable_tools=False,
            tool_name="submit_" + identifier.replace("/", "_"),
        )
    else:
        raise ValueError(f"TEMPLATE_EXECUTION: unsupported single-value task {identifier}")
    Draft202012Validator(template["output_schema"]).validate(value)
    if template["execution"] == "host" and template.get("operation") == "identity" and value != context:
        raise ValueError("TEMPLATE_PROGRESS: host identity result differs from its input")
    if any(isinstance(item, str) and not item.strip()
           and template["output_schema"]["properties"][key].get("minLength", 0) > 0
           for key, item in value.items()):
        raise ValueError(f"TEMPLATE_VALUE: blank required field in {identifier}")
    if saved is None and checkpoint is not None:
        checkpoint(binding, [deepcopy(value)])
    return value
