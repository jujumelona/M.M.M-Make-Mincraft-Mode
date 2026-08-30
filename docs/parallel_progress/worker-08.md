WORKER: 08
ROLE: Minecraft Domain Correctness
STATUS: COMPLETE
LAST_UPDATED_MAIN_SHA: 7b6703c73f1a7db93b49b03eccfd0d9e954e1db4

COMPLETED:
- Audited the live Fabric deterministic generation surface for Minecraft API/mappings correctness, registry/resource assumptions, specialized networking/templates, client/server separation, and public reachability.
- Confirmed `platform_catalog.PlatformAdapter.deterministic_module_kinds` remains the single target capability authority; Worker 08 owns no separate Minecraft-version table.
- Confirmed the live Fabric adapter currently advertises `deterministic_module_kinds=frozenset()`, so unverified legacy deterministic templates remain intentionally fail-closed.
- Added and retained the central fail-closed Minecraft domain guard around the shared extended-content mutation primitive and both legacy project-generator entry points.
- Retargeted package-local import-by-value aliases of `generate_extended_content` so stale aliases cannot bypass the central guard.
- Verified `MinecraftModPipeline`, `ScalableMinecraftModPipeline`, and the complete orchestrator all reach guarded generator classes/shared primitives.
- Verified runtime bootstrap installs `install_minecraft_domain_correctness()` in the platform stage before specialized deterministic generator guards.
- Audited built-in system-pack generation. Its Java templates include target-sensitive command/persistence/networking/resource assumptions and therefore remain capability-gated before mutation.
- Audited GeckoLib generation. Client model/renderer sources are emitted under the client package/entrypoint, but GeckoLib API/artifact compatibility is target- and dependency-version-sensitive.
- Hardened specialized capability semantics so ordinary semantic support cannot accidentally unlock a larger static implementation:
  - built-in system packs now require all semantic kinds plus `system-pack:<pack-id>`;
  - GeckoLib now requires `entity`, `geckolib:entity`, and exact `geckolib:version:<version>` evidence.
- Added regression tests proving semantic-only capabilities do not unlock system/GeckoLib templates and that GeckoLib approval is bound to the requested dependency version.
- Audited local AI sidecar generation and resource-asset production; these do not constitute version-specific Minecraft/Fabric Java API template paths and do not need the deterministic Minecraft capability guard.
- Audited the official Fabric template provider; it delegates scaffold creation to Fabric's maintained CLI and verifies the resulting target/toolchain receipt rather than supplying another legacy Minecraft Java API template.
- Recorded cross-owner re-enable criteria and CI ownership in `docs/parallel_handoff/worker-08.md`.

ROOT_CAUSES_CONFIRMED:
- Platform capability and legacy deterministic generation were originally disconnected: live targets could advertise no reviewed deterministic kinds while legacy writers were still callable.
- Legacy deterministic templates contain concrete API-era assumptions, so a generator's internal `_SUPPORTED` set is not evidence that the selected Minecraft/Fabric target supports that template.
- A semantic capability is narrower than a specialized implementation capability. `entity` does not prove GeckoLib compatibility; `quest` does not prove an entire quest system template including persistence/commands/resources.
- Third-party integration compatibility must be bound to the exact dependency version, not merely the Minecraft semantic feature.

DECISIONS_AND_EVIDENCE:
- Fail closed before the first filesystem mutation when the authoritative target does not explicitly advertise every required deterministic capability.
- Keep dynamic target-grounded custom Java/RAG editing available; Worker 08 blocks only unverified static deterministic writers.
- Do not patch old API names one-by-one and then declare a target supported. Re-enable only after exact-target compile/resource/runtime evidence.
- Preserve the current bootstrap ordering: central Minecraft domain correctness guard first, specialized generator guard second.
- Treat `verified_scaffold_registry.py` as a Worker 01 reachability/authority audit item because it contains a historical hardcoded target matrix; Worker 08 did not introduce or rely on it.

COMMITS_ALREADY_PUSHED:
- 341e882636ba9b68d13c5860eeb984a2a9e1ae0c add target-aware Minecraft generation guard
- 48fb141b956bc0676824af372c8d8b4c40c3d378 add target capability regression tests
- 1927e0a74d24b003e0a0c079ec0cf12e4b48365c install Minecraft domain correctness guard
- 6908085520c96032150633240e8b482f6ae6eb67 checkpoint Worker 08 progress
- 74bc890cb190662c21d6317e818e239ccd808cfe guard legacy project generators by target capability
- 266e90b33fe361ade66d21df3cb2ce9a993d1cb8 prevent legacy project generation on unreviewed targets
- 8457b43d497e279aa14799d1b7ce2422318b110c verify guard coverage for public generators
- d057659bd51b04bba4098ecd4dc77312c074dcbf isolate specialized Minecraft capabilities
- 2a306a9fd308cff5116a0457e896053382faba2c require specialized capability evidence in regression tests
- 7b6703c73f1a7db93b49b03eccfd0d9e954e1db4 hand off Minecraft capability verification and shared-CI follow-up

TESTS_AND_CI:
- Earlier isolated local harness: Python compile of Worker 08 guard/test modules passed; empty live capability rejected before mutation and reviewed capability passed.
- GitHub CI for the latest Worker 08 code/test commits successfully installed dependencies and passed the repository static internal-import/bootstrap audit (`STATIC DEBUG AUDIT OK: 380 package Python files and 16 workflows checked`).
- Full repository compile/pytest did NOT execute on those CI runs because the shared `Audit runtime mutation budget` gate failed first: reviewed behavioral budget=413, measured behavioral surface=443. Later jobs/tests were skipped or cancelled.
- Therefore Worker 08 does not claim a full repository pytest pass. The shared runtime-mutation gate is handed to Worker 12 with exact acceptance criteria.

FILES_AUDITED_OR_CHANGED:
- minecraft_mod_ai/minecraft_domain_correctness_contract.py
- minecraft_mod_ai/deterministic_minecraft_content_contract.py
- minecraft_mod_ai/extended_content_generator.py
- minecraft_mod_ai/generator.py
- minecraft_mod_ai/scalable_generator.py
- minecraft_mod_ai/pipeline.py
- minecraft_mod_ai/scalable_pipeline.py
- minecraft_mod_ai/platform_catalog.py
- minecraft_mod_ai/platform_generation_contract.py
- minecraft_mod_ai/platform_specialized_generator_contract.py
- minecraft_mod_ai/system_pack_generator.py
- minecraft_mod_ai/system_templates_common.py
- minecraft_mod_ai/system_templates_class_skill.py
- minecraft_mod_ai/system_templates_economy.py
- minecraft_mod_ai/system_templates_groups.py
- minecraft_mod_ai/system_templates_quest.py
- minecraft_mod_ai/system_templates_social.py
- minecraft_mod_ai/geckolib_generator.py
- minecraft_mod_ai/production_tools.py
- minecraft_mod_ai/mod_generation_mcp_server.py
- minecraft_mod_ai/complete_orchestrator.py
- minecraft_mod_ai/runtime_bootstrap.py
- minecraft_mod_ai/runtime_wrapper_integrity.py
- minecraft_mod_ai/local_ai_sidecar_generator.py
- minecraft_mod_ai/resource_asset_production.py
- minecraft_mod_ai/fabric_official_template_provider.py
- minecraft_mod_ai/verified_scaffold_registry.py
- tests/test_minecraft_domain_correctness_contract.py
- tests/test_minecraft_specialized_generation_guard.py
- docs/parallel_handoff/worker-08.md

KNOWN_CROSS_ROLE_DEPENDENCIES:
- Worker 01: verifies exact target/mappings/loader/API/dependency combinations and is the only owner that should advertise deterministic capability tokens after evidence. Handoff recorded.
- Worker 07: owns repository localization/source-edit mechanics; dynamic target-grounded source editing remains the supported path while static templates are unverified.
- Worker 12: owns shared runtime composition/mutation-budget integrity. Handoff recorded for the current 443>413 CI blocker.

NEXT_EXACT_ACTIONS:
- No further Worker 08-owned safety-boundary change is required on the audited current surface.
- Worker 01 must modernize/verify any static template before advertising its exact capability tokens.
- Worker 12 must resolve the shared runtime mutation budget gate and rerun the Worker 08 tests so repository CI can execute them.

UNRESOLVED:
- None within Worker 08's owned fail-closed domain-correctness boundary.
- Static template modernization/re-enablement remains intentionally delegated to Worker 01 because it changes provider-advertised support.
- Repository-wide CI remains blocked before pytest by the shared runtime mutation budget and is delegated to Worker 12.
