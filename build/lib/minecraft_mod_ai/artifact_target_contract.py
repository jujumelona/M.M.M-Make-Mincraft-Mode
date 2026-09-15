"""Validate declared template schema targets without inventing a Minecraft version."""

from packaging.version import InvalidVersion, Version


def validate_artifact_target(template, version):
    contract = template.get("target_contract")
    if not contract:
        return
    try:
        current = Version(version)
        minimum = Version(contract["minimum_minecraft"])
        maximum = Version(contract["maximum_minecraft"])
    except (InvalidVersion, TypeError, KeyError) as exc:
        raise ValueError(
            "ARTIFACT_TARGET_REQUIRED: explicit supported Minecraft version required"
        ) from exc
    if not minimum <= current <= maximum:
        raise ValueError(f"ARTIFACT_TARGET_UNSUPPORTED: {version}")
