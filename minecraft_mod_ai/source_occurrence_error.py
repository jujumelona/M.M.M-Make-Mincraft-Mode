"""Shared occurrence-planning error without dependencies on authoring engines."""
import json


class SourceHookUnavailable(ValueError):
    def __init__(self, behavior_id: str, *, reason: str, inspected: int):
        super().__init__("SOURCE_HOOK_PRODUCER_REQUIRED: " + json.dumps({
            "behavior_id": behavior_id, "reason": reason,
            "inspected_callback_count": inspected,
            "required_action": "materialize a custom event/mixin source owner "
                               "or bind a real observable Fabric callback before effects",
        }, ensure_ascii=False))
