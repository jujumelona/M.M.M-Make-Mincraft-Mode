WORKER: 08
ROLE: Minecraft Domain Correctness
STATUS: IN_PROGRESS
LAST_UPDATED_MAIN_SHA: 1927e0a74d24b003e0a0c079ec0cf12e4b48365c

COMPLETED:
- Audited deterministic Minecraft content generation against the live platform capability contract.
- Confirmed the live Fabric adapter advertises deterministic_module_kinds=frozenset(), while the legacy generator still contains concrete version-specific Fabric/Minecraft API templates.
- Added a fail-closed domain guard that consults platform_catalog.adapter_from_project as the single target authority before deterministic generation can mutate files.
- Added target-specific allowlist handling so a future adapter may enable only reviewed module kinds instead of implicitly enabling every legacy template.
- Installed the guard after runtime bootstrap and before runtime finalization.
- Added regression tests for empty capability, partial capability, allowed capability, and pre-generator mutation blocking.

IN_PROGRESS:
- Audit remaining Minecraft-domain generators for version-specific registry/mapping/resource/client-server assumptions outside apply_minecraft_content_spec.
- Identify whether legacy resource/data paths and Java templates are reachable on live targets through any non-deterministic path.

ROOT_CAUSES_CONFIRMED:
- Platform capability and deterministic generator support were disconnected: live targets explicitly advertised no reviewed deterministic module kinds, but apply_minecraft_content_spec still reached the legacy generator.
- The legacy generator contains concrete API-era assumptions, so treating its internal _SUPPORTED set as target support is unsound.

DECISIONS_AND_EVIDENCE:
- Do not create a second Minecraft version table in worker 08. platform_catalog.adapter_from_project and PlatformAdapter.deterministic_module_kinds remain authoritative.
- Official Fabric 26.2 documentation uses current ResourceKey/Item.Properties/BlockBehaviour.Properties registration patterns, which differ materially from the legacy FabricItemSettings/FabricBlockSettings/Yarn-era templates in extended_content_generator.py.
- Fail closed before mutation when target support is absent or a requested kind is not advertised.

COMMITS_ALREADY_PUSHED:
- 341e882636ba9b68d13c5860eeb984a2a9e1ae0c add target-aware Minecraft generation guard
- 48fb141b956bc0676824af372c8d8b4c40c3d378 add target capability regression tests
- 1927e0a74d24b003e0a0c079ec0cf12e4b48365c install Minecraft domain correctness guard

TESTS_ALREADY_PASSING:
- python -m py_compile for the new guard and regression test modules in an isolated local harness
- isolated guard smoke: empty live capability rejects before mutation; reviewed capability passes
- no repository GitHub Actions workflow was attached to integration commit 1927e0a at check time

NEXT_EXACT_ACTIONS:
1. Audit other Java/resource generation functions and call sites for live-target reachability.
2. Add fail-closed/version-grounded contracts for any reachable unverified path rather than patching individual old API names.
3. Add regression tests for resource paths and client/server separation where current code has deterministic assumptions.
4. Re-check latest origin/main, run available repository CI/status checks, then mark this worker complete only when unresolved domain-correctness paths are accounted for.

FILES_CURRENTLY_RELEVANT:
- minecraft_mod_ai/minecraft_domain_correctness_contract.py
- minecraft_mod_ai/deterministic_minecraft_content_contract.py
- minecraft_mod_ai/extended_content_generator.py
- minecraft_mod_ai/platform_catalog.py
- minecraft_mod_ai/platform_generation_contract.py
- minecraft_mod_ai/__init__.py
- tests/test_minecraft_domain_correctness_contract.py

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Worker 01 owns platform discovery and decides which PlatformAdapter capabilities are verified.
- Worker 07 owns repository localization/source-edit mechanics; worker 08 only defines which Minecraft implementation is valid.
- Worker 12 owns shared bootstrap/core; worker 08 made only the minimal __init__.py install hook required to enforce domain correctness.

UNRESOLVED:
- Other generation paths may still encode stale Minecraft/Fabric APIs or legacy resource paths; reachability audit is not complete.
