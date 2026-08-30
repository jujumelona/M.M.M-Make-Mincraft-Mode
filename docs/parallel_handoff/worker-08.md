FROM_WORKER: 08
ROLE: Minecraft Domain Correctness
STATUS: OPEN

TARGET_OWNER: 01
REASON:
- The live Fabric provider correctly advertises no reviewed deterministic module kinds today, so legacy version-specific templates remain fail-closed.
- Legacy static generators still contain concrete Yarn/Fabric-era Java, resource-path, GameTest, networking, and registry assumptions. They must not be re-enabled merely because a semantic kind such as `entity` or `quest` becomes supported.
- Worker 08 hardened specialized capability boundaries: system packs require their semantic kinds plus `system-pack:<pack-id>`; GeckoLib requires `entity`, `geckolib:entity`, and the exact `geckolib:version:<version>` token.

FILES:
- minecraft_mod_ai/platform_catalog.py
- minecraft_mod_ai/generator.py
- minecraft_mod_ai/scalable_generator.py
- minecraft_mod_ai/extended_content_generator.py
- minecraft_mod_ai/system_pack_generator.py
- minecraft_mod_ai/system_templates_common.py
- minecraft_mod_ai/system_templates_class_skill.py
- minecraft_mod_ai/system_templates_economy.py
- minecraft_mod_ai/system_templates_groups.py
- minecraft_mod_ai/system_templates_quest.py
- minecraft_mod_ai/system_templates_social.py
- minecraft_mod_ai/geckolib_generator.py
- minecraft_mod_ai/verified_scaffold_registry.py

EXACT_HANDOFF:
1. Keep `PlatformAdapter.deterministic_module_kinds` as the single capability authority; do not create a second target/version allowlist.
2. Do not advertise a deterministic or specialized capability until the exact discovered target, mappings family, loader/API coordinates, resource layout, and any third-party dependency version have evidence.
3. Modernize and verify the legacy static templates against the selected target before enabling them. Known review points include old `FabricItemSettings`/`FabricBlockSettings`, `Identifier` construction, GameTest APIs, recipe/loot resource paths, registry calls, Fabric networking APIs, and client/server-only classes.
4. For a built-in system pack, advertise all semantic kinds plus `system-pack:<pack-id>` only after the complete generated system template (commands/persistence/networking/resources as applicable) is verified.
5. For GeckoLib, advertise `entity`, `geckolib:entity`, and the exact `geckolib:version:<version>` only after that exact GeckoLib artifact/API combination compiles and runs on the discovered target.
6. Audit `verified_scaffold_registry.py` for live reachability. If it is reachable from target selection/generation, remove or quarantine that historical hardcoded matrix from the live-provider authority path rather than letting it compete with `platform_catalog`.

ACCEPTANCE_CRITERIA:
- Generated source compiles against the exact provider-discovered mappings/API target.
- Registry and networking bindings use the target-correct APIs and server authority model.
- Resource paths and pack metadata load on the exact target.
- Client-only renderer/UI classes are never loaded by a dedicated server.
- GameTest/runtime evidence covers the enabled deterministic template.
- The provider advertises only the exact reviewed capability tokens, with no semantic-token shortcut to a larger specialized generator.

---

FROM_WORKER: 08
ROLE: Minecraft Domain Correctness
STATUS: OPEN

TARGET_OWNER: 12
REASON:
- Worker 08 code/tests reached CI, but the repository-wide runtime mutation audit fails before Python compile/pytest can run: reviewed behavioral budget is 413 while the measured surface is 443.
- The bootstrap currently installs Worker 08's Minecraft guard in the platform stage and then the specialized generator guard; preserve that ordering while reducing/reviewing the global wrapper surface.

FILES:
- .github/scripts/audit_runtime_mutations.py
- .github/workflows/ci.yml
- minecraft_mod_ai/runtime_bootstrap.py
- minecraft_mod_ai/runtime_wrapper_integrity.py

EXACT_HANDOFF:
1. Reconcile the 443 behavioral runtime mutations with the canonical contract owners and pruning work; do not blindly raise the reviewed budget.
2. Preserve `install_minecraft_domain_correctness()` before `install_specialized_generator_guards(...)` in the platform bootstrap stage.
3. Once the mutation audit is clean, rerun runtime composition plus Worker 08's Minecraft correctness tests so the tests actually execute instead of being cancelled upstream.

ACCEPTANCE_CRITERIA:
- Runtime mutation audit passes through reviewed ownership/pruning rather than an unexplained budget increase.
- Package bootstrap and runtime wrapper integrity pass.
- `tests/test_minecraft_domain_correctness_contract.py` executes and passes.
- `tests/test_minecraft_specialized_generation_guard.py` executes and passes.
