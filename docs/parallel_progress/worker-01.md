# Worker 01 Progress

WORKER: 01
ROLE: Minecraft platform target + build scaffold authority
STATUS: VERIFYING
BASE_MAIN_SHA: 32b6b2ca92ce51fe060c23fb7a2c672c0e49f422

## Root cause

`platform_catalog.py` already owns executable platform support through validated, frozen `PlatformAdapter` receipts, but `verified_scaffold_registry.py` independently owned a static `SUPPORTED_TARGET_SPECS` Minecraft/loader matrix. That second authority could advertise Forge/NeoForge combinations without executable providers, freeze stale Gradle/Loom/API coordinates, and re-resolve a target after an exact adapter had already been selected.

## Fix

- Removed the static Minecraft/loader support matrix from scaffold authority.
- Scaffold acceptance is now backed by the executable provider registry and exact `PlatformAdapter.validate()` receipt.
- Added adapter-native template/materialization entry points; an already-resolved adapter is reused without catalog/network re-resolution.
- Compatibility `(loader, minecraft_version)` entry points resolve exactly once and delegate.
- Embedded adapters in target context are reused only when loader/version identity agrees; mismatches fail closed.
- Unsupported loaders fail closed until they have a real executable provider and scaffold implementation.
- Gradle distribution SHA-256 is taken from the exact adapter receipt, not a Minecraft-version table.
- Historical wrapper checksums remain only as artifact-integrity compatibility pins; unknown/current Gradle wrappers are grounded in the current official Fabric template and verified by Git blob identity plus JAR structure.
- Fabric Loom plugin selection follows the current template boundary: unobfuscated 26.x+ uses `net.fabricmc.fabric-loom`; older remapped releases use `net.fabricmc.fabric-loom-remap` and Mojang mappings when applicable.

## Current external evidence checked (2026-08-31)

- Fabric Meta API: https://meta.fabricmc.net/
- Fabric official template repository: https://github.com/FabricMC/fabricmc.net
- Observed template commit: `34b9cbd2119564d1b2d56d6877e814cac9ad1f81`
- Observed official template Gradle: `9.5.1`
- Observed official template Loom: `1.17-SNAPSHOT`

The removed static scaffold matrix topped out at Gradle 8.10.2, demonstrating that it had already diverged from the live provider/template ecosystem.

## Regression coverage

`tests/test_worker01_platform_scaffold_authority.py` covers:

1. exact-adapter reuse with zero catalog re-resolution;
2. compatibility wrapper resolves exactly once;
3. invalid provider metadata rejection before scaffold acceptance;
4. unsupported loader cannot bypass executable-provider support;
5. provider/offline resolution failure is fail-closed;
6. 26.x vs older Loom plugin/remapping boundary;
7. Gradle integrity ownership by adapter rather than static target table;
8. embedded adapter reuse without re-resolution;
9. embedded adapter identity mismatch rejection;
10. structural absence of the stale static Minecraft target matrix.

## Production support policy

Executable production support remains Fabric-only because Fabric is the only registered executable provider. NeoForge/Forge must not be advertised until equivalent live metadata, generation, buildability, and validation providers exist.

## Residual limitations

- First-time wrapper materialization requires network access unless a verified wrapper is already cached; failure is explicit and fail-closed.
- Adding another loader requires a real executable provider plus equivalent scaffold/buildability validation, not static compatibility rows.

FINAL_VERIFICATION_COMMIT: pending
FINAL_VERIFICATION_RUN: pending
UNRESOLVED: final CI verification pending
