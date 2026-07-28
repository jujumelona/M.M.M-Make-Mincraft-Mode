"""Compatibility exports for the approval-gated local tool broker.

The former prototype accepted arbitrary string actions and reported simulated
success.  The real broker uses a closed enum, an approved proposal hash, and a
workspace boundary for every mutating request.
"""

from minecraft_mod_ai.broker import (
    LocalPolicyBroker,
    PolicyDenied,
    ToolAction,
    ToolRequest,
)
from minecraft_mod_ai.capabilities import (
    capability_manifest,
    capability_manifest_hash,
)

__all__ = [
    "LocalPolicyBroker",
    "PolicyDenied",
    "ToolAction",
    "ToolRequest",
    "capability_manifest",
    "capability_manifest_hash",
]
