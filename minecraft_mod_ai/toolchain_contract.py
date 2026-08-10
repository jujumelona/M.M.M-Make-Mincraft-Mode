from __future__ import annotations

from typing import Any


LOOM_VERSION = "1.10.5"
GRADLE_VERSION = "8.12"
GRADLE_SHA256 = "7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03"


def fabric_dependency_predicates(platform: Any) -> dict[str, str]:
    """Return runtime predicates covered by the verified platform contract.

    The build remains pinned to the verified Fabric Loader baseline in
    ``gradle.properties``. Runtime metadata advertises that baseline as a minimum,
    because Loom/Fabric dependency resolution may legitimately select a newer
    compatible Loader. An exact predicate here makes an otherwise successful build
    fail at launch when the resolved Loader is newer than the compile-time baseline.

    Minecraft, Java and Fabric API stay exact because they are part of the generated
    version adapter and resource/API surface. The Loader predicate alone is a lower
    bound over the verified baseline.
    """

    platform.validate()
    values = {
        "fabricloader": f">={platform.fabric_loader}",
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
    """Install one verified Fabric 1.20.1 toolchain contract.

    GeckoLib 4.8.2 is published with Loom 1.10.5 metadata. Loom 1.5.4 refuses
    that dependency before Java compilation. Loom 1.10 targets Gradle 8.12,
    so generation, validation, wrapper creation and downloads use this exact pair
    and the official binary distribution checksum. The generated build still pins
    Fabric Loader 0.16.10 as its verified baseline, while runtime metadata accepts
    compatible newer Loader releases.
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
