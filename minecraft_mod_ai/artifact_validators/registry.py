from __future__ import annotations

"""Registry identifier and namespace validation."""

import re

_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")


class RegistryValidationError(ValueError):
    pass


def validate_registry_identifier(identifier: str) -> dict[str, str]:
    """Ensure identifier matches Minecraft's strict lowercase namespace:path syntax."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise RegistryValidationError("REGISTRY_ID_EMPTY: Identifier cannot be empty")

    clean = identifier.strip()
    if not _IDENTIFIER_PATTERN.match(clean):
        raise RegistryValidationError(
            f"REGISTRY_ID_INVALID: Identifier {identifier!r} must match format 'namespace:path' "
            "with lowercase alphanumeric characters, underscores, dashes, or dots"
        )

    namespace, path = clean.split(":", 1)
    return {
        "status": "PASS",
        "identifier": clean,
        "namespace": namespace,
        "path": path,
    }
