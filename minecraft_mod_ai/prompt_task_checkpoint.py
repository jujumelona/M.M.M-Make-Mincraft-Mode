"""Prompt-task progress is never a researched or implementation-ready planning state."""
from copy import deepcopy


def is_prompt_checkpoint(value):
    return isinstance(value, dict) and value.get("checkpoint_kind") == "prompt_tasks"


def restore_prompt_progress(value, prompt):
    from .planning_state_contract import _hash_without

    if value is None:
        return {}
    if not is_prompt_checkpoint(value) or set(value) != {
        "checkpoint_kind", "original_prompt", "template_progress", "state_sha256"
    }:
        raise ValueError("PROMPT_TASK_CHECKPOINT: invalid progress envelope")
    if value["original_prompt"] != prompt:
        raise ValueError("PROMPT_TASK_CHECKPOINT: original prompt changed")
    if value["state_sha256"] != _hash_without(value, "state_sha256"):
        raise ValueError("PROMPT_TASK_CHECKPOINT: checkpoint hash mismatch")
    if not isinstance(value["template_progress"], dict):
        raise ValueError("PROMPT_TASK_CHECKPOINT: expected progress mapping")
    return deepcopy(value["template_progress"])


def prompt_checkpoint(prompt, progress):
    from .planning_state_contract import _hash_without

    result = {"checkpoint_kind": "prompt_tasks", "original_prompt": prompt,
              "template_progress": deepcopy(progress), "state_sha256": ""}
    result["state_sha256"] = _hash_without(result, "state_sha256")
    return result
