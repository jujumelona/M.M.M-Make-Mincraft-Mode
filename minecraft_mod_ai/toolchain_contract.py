from __future__ import annotations

from typing import Any


LOOM_VERSION = "1.10.5"
GRADLE_VERSION = "8.12"
GRADLE_SHA256 = "7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03"


def fabric_dependency_predicates(platform: Any) -> dict[str, str]:
    """Return the runtime predicates covered by the immutable platform lock.

    Fabric treats a standalone version as an exact version predicate.  Keeping
    this policy in one code-owned function prevents generated metadata from
    quietly advertising every future Loader or Fabric API release via ``>=``.
    The build metadata in the Fabric API version is additionally bound by the
    exact Gradle dependency in ``gradle.properties`` and by release provenance.
    """

    platform.validate()
    values = {
        "fabricloader": platform.fabric_loader,
        "minecraft": platform.minecraft_version,
        "java": platform.java_version,
        "fabric-api": platform.fabric_api,
    }
    for dependency_id, value in values.items():
        if (
            type(value) is not str
            or not value.strip()
            or value != value.strip()
            or any(ord(character) < 0x20 for character in value)
        ):
            raise ValueError(
                f"Invalid locked version for Fabric dependency {dependency_id!r}."
            )
    return values


def install(spec_module: Any, runner_module: Any) -> None:
    """Install one immutable Fabric 1.20.1 toolchain contract.

    GeckoLib 4.8.2 is published with Loom 1.10.5 metadata. Loom 1.5.4 refuses
    that dependency before Java compilation. Loom 1.10 targets Gradle 8.12,
    so generation, validation, wrapper creation and downloads must use this
    exact pair and the official binary distribution checksum.
    """

    def platform_init(
        self: Any,
        edition: str = "java",
        loader: str = "fabric",
        minecraft_version: str = "1.20.1",
        java_version: str = "17",
        yarn_mappings: str = "1.20.1+build.1",
        fabric_loader: str = "0.16.10",
        fabric_api: str = "0.92.11+1.20.1",
        fabric_loom: str = LOOM_VERSION,
        gradle: str = GRADLE_VERSION,
    ) -> None:
        for field_name, value in {
            "edition": edition,
            "loader": loader,
            "minecraft_version": minecraft_version,
            "java_version": java_version,
            "yarn_mappings": yarn_mappings,
            "fabric_loader": fabric_loader,
            "fabric_api": fabric_api,
            "fabric_loom": fabric_loom,
            "gradle": gradle,
        }.items():
            object.__setattr__(self, field_name, value)

    def platform_validate(self: Any) -> None:
        expected = {
            "edition": "java",
            "loader": "fabric",
            "minecraft_version": "1.20.1",
            "java_version": "17",
            "yarn_mappings": "1.20.1+build.1",
            "fabric_loader": "0.16.10",
            "fabric_api": "0.92.11+1.20.1",
            "fabric_loom": LOOM_VERSION,
            "gradle": GRADLE_VERSION,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise spec_module.SpecValidationError(
                    f"Unsupported platform adapter: {field_name}={actual!r}; "
                    f"the verified Fabric 1.20.1 toolchain is pinned to {expected_value!r}."
                )

    spec_module.PlatformLock.__init__ = platform_init
    spec_module.PlatformLock.validate = platform_validate
    runner_module.GRADLE_VERSION = GRADLE_VERSION
    runner_module.GRADLE_URL = (
        f"https://services.gradle.org/distributions/gradle-{GRADLE_VERSION}-bin.zip"
    )
    runner_module.GRADLE_SHA256 = GRADLE_SHA256
