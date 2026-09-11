# HOST Integrity Remaining Tasks

## CRITICAL P0 (Must complete for production)

### P0-REGISTRY: Complete ImplementationRegistry fail-closed enforcement
**Status**: IN PROGRESS
- [x] Remove fallback hashes in make_implementation()
- [x] Remove fallback hashes in template_hashes()
- [ ] Register all validators in registry (semantic_contract, java_syntax, json_schema)
- [ ] Register all schemas in TypeRegistry for each canonical leaf
- [ ] Test that catalog generation fails without complete registrations

### P0-API-EPOCH: Redesign API epoch catalog based on actual target inspection
**Status**: NOT STARTED
**Problem**: Current api_epoch_catalog.py mixes Fabric/NeoForge APIs and uses guesswork
**Solution**:
1. Remove hardcoded DeferredRegister (NeoForge) from Fabric project
2. Extract actual API contracts from:
   - Minecraft JAR (per version)
   - Fabric Loader/API JAR
   - Yarn/Intermediary mappings
3. Generate epoch profiles from actual availability:
   - Registry.register() vs RegistryKey + FabricDefaultAttributeRegistry
   - Item.Settings() constructor changes
   - Block.Settings() API changes
   - Component DataComponentType availability (1.21+)
4. Create actual template files for each epoch:
   - fabric/item/register_v1_19.yaml (direct Registry.register)
   - fabric/item/register_v1_20.yaml (RegistryKey pattern)
   - fabric/item/register_v1_21.yaml (modern + components)

### P0-SYMBOL-RESOLVER: Connect real Java symbol resolution
**Status**: NOT STARTED
**Problem**: Current java_symbol_resolver.py uses regex, can't resolve descriptors/overloads
**Solution**:
1. Choose resolver: jdt.ls (LSP) OR javac attribution OR JavaParser symbol solver
2. Implement:
   - `resolve_with_classpath(java_source, exact_classpath) -> [(owner, name, descriptor, static, side)]`
   - Full descriptor including parameter types
   - Overload disambiguation
   - Static vs instance
3. Use exact target's minecraft + fabric + mappings JARs as classpath
4. Reject admission if symbols can't be resolved

### P0-API-EXTRACTOR: Extract real API symbols from JARs
**Status**: NOT STARTED  
**Problem**: Current api_symbol_extractor.py hardcodes representative set, doesn't parse classes
**Solution**:
1. Download actual JARs:
   - `maven_download(minecraft_version) -> minecraft-{version}.jar`
   - `maven_download(fabric_api_version) -> fabric-api-{version}.jar`
2. Parse class files:
   - Use jawa or similar to read constant pool
   - Extract methods: (owner, name, descriptor, static, side)
   - Extract fields: (owner, name, descriptor, static, side)
3. Apply mappings:
   - Load yarn/intermediary mappings for target
   - Apply class/method/field remapping
   - Keep official, intermediary, named namespaces separate
4. Store per-target API catalog:
   - `data/api_symbols/{minecraft_version}_{fabric_version}.json`
   - Used for admission validation

### P0-MAPPINGS: Implement real mappings application
**Status**: NOT STARTED
**Problem**: apply_mappings() is stub
**Solution**:
1. Parse yarn mappings format (tiny v2)
2. Implement class/method/field remapping
3. Handle namespaces: official → intermediary → named
4. Validate HOST symbols match target namespace

### P0-EVIDENCE: Generate actual compile/test evidence
**Status**: NOT STARTED
**Problem**: Production gate exists but no evidence files
**Solution**:
1. For each (minecraft_version, required_leaf):
   - Run actual Gradle compile with exact dependencies
   - Capture compile output, exit code, duration
   - Store CompileResult in EvidenceStore
2. For runtime-sensitive leaves:
   - Run actual GameTest or runtime validation
   - Store GameTestResult/RuntimeTestResult
3. Evidence file per leaf:
   - `data/evidence/{evidence_id}.json`
4. production_readiness_audit() queries EvidenceStore
5. No evidence = FAIL

### P0-SIDE-ENFORCEMENT: Connect to actual execution
**Status**: NOT STARTED
**Problem**: side_enforcement.py exists but not enforced
**Solution**:
1. In artifact_job executor dispatcher:
   - Before rendering template, call `validate_side_constraints(leaf_side, symbols, target_source_set)`
   - If validation fails, abort job with clear error
2. In validator:
   - Check generated code doesn't reference wrong-side symbols
3. Test:
   - Try to use ClientLevel in common leaf → FAIL
   - Try to use MinecraftServer in client leaf → FAIL

## P1 (High Priority)

### P1-AUDIT: Complete leaf reachability audit
**Status**: NOT STARTED
**Problem**: audit_all_leaves.py only counts states, doesn't verify reachability
**Solution**:
1. For each required_leaf in each supported_version:
   - Try to resolve: canonical_leaf → implementation → executor → validator → schemas
   - Try to find evidence in EvidenceStore
   - Check all references are valid (no dangling IDs)
2. Report:
   - PASS: Full reachability chain exists
   - FAIL: Missing link in chain
   - Save detailed report

### P1-TEMPLATES: Create epoch-specific executable templates
**Status**: NOT STARTED
**Depends on**: P0-API-EPOCH
**Solution**:
1. Create templates/fabric/item/register_v1_19.yaml
2. Create templates/fabric/item/register_v1_20.yaml
3. Create templates/fabric/item/register_v1_21.yaml
4. Similar for block, entity, networking, worldgen
5. Each template uses era-appropriate APIs

### P1-GENERATORS: Implement Python generators for complex leaves
**Status**: NOT STARTED
**Problem**: Entity/GUI/network/worldgen have no templates yet
**Solution**:
1. Create Python generator functions:
   - `generate_entity_registration(inputs) -> [(file, content)]`
   - `generate_gui_screen(inputs) -> [(file, content)]`
   - etc.
2. Register in ImplementationRegistry as ExecutorType.PYTHON_GENERATOR
3. Test generators produce valid code
4. Add evidence for generated outputs

## P2 (Future)

### P2-OPTIMIZE: Cache and incremental builds
- Cache JAR downloads
- Cache API symbol extraction
- Incremental evidence generation
- Parallel compilation

### P2-OBSERVABILITY: Better diagnostics
- Detailed trace of admission decisions
- Symbol resolution debug output
- Evidence generation logs
- Failure attribution

## Current Blockers

1. **TypeRegistry empty**: No schemas registered yet
   - Need to create schema definitions for each canonical leaf
   - Register in get_global_type_registry()

2. **Validator registry empty**: No validators registered
   - Need to implement actual validator functions
   - Register with specific ValidatorType

3. **Evidence store empty**: No actual evidence files
   - Need to run real compilations
   - Generate and store evidence records

4. **Symbol resolution incomplete**: Can't validate API usage
   - Need real classpath-based resolution
   - Need descriptor extraction

5. **Mappings not applied**: Can't match symbols across namespaces
   - Need to parse and apply yarn mappings
   - Need namespace-aware comparison

## Success Criteria

Production-ready when:
- [ ] All SUPPORTED_MINECRAFT_VERSIONS × REQUIRED_CANONICAL_LEAVES have evidence
- [ ] validate_support_matrix() passes for all versions
- [ ] No fallback/fake hashes anywhere
- [ ] All symbols resolved with descriptors
- [ ] Side constraints enforced at runtime
- [ ] Audit shows 100% reachability for required leaves
- [ ] Integration test: Generate full mod, compile, run GameTest → PASS
