from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .project_edit import (
    ensure_main_initializer_call,
    inspect_fabric_project,
    write_text_files,
)
from .scale_policy import ScalePolicy


class WorldRuntimeGenerationError(RuntimeError):
    pass


def generate_world_runtime_bridge(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    """Install the fixed-size Fabric bridge for sharded world contracts.

    Compiled world resources must already be merged into ``src/main/resources``.
    The bridge discovers per-structure contracts at runtime, scans chunk candidates
    with a bounded tick budget, and persists every reserved anchor and shard cursor.
    """

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise WorldRuntimeGenerationError(
            "World runtime target does not match the Fabric project."
        )
    contract_root = (
        info.root
        / "src"
        / "main"
        / "resources"
        / "data"
        / mod_id
        / "mmm_world"
        / "contracts"
    )
    if not contract_root.is_dir() or contract_root.is_symlink():
        raise WorldRuntimeGenerationError(
            "Compiled world contracts are missing from project resources."
    )
    contract_count = 0
    piece_contract_count = 0
    runtime_structure_count = 0
    for path in sorted(contract_root.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
            raise WorldRuntimeGenerationError(
                f"World contract is not a regular file: {path}"
            )
        if path.stat().st_size > policy.max_single_file_bytes:
            raise WorldRuntimeGenerationError(
                f"World contract exceeds the per-file policy: {path}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorldRuntimeGenerationError(
                f"World contract is not valid JSON: {path}"
            ) from exc
        if not isinstance(value, dict) or not str(
            value.get("schema_version", "")
        ).startswith("mmm/world-"):
            raise WorldRuntimeGenerationError(
                f"World contract schema is invalid: {path}"
            )
        if "structure_pieces" in path.relative_to(
            contract_root
        ).parts:
            piece_contract_count += 1
            continue
        contract_count += 1
        if (
            value.get("schema_version")
            == "mmm/world-structure-runtime-v1"
            and value.get("placement")
            == "runtime_function_shards"
        ):
            runtime_structure_count += 1
    if contract_count == 0:
        raise WorldRuntimeGenerationError(
            "Compiled world contract directory is empty."
        )

    runtime_package = package_name + ".world"
    java_root = (
        "src/main/java/"
        + runtime_package.replace(".", "/")
        + "/"
    )
    replacements = {
        "__PACKAGE__": runtime_package,
        "__MOD_ID__": mod_id,
    }
    runtime_source = _replace_all(_RUNTIME_JAVA, replacements)
    state_source = _replace_all(_STATE_JAVA, replacements)
    files = {
        java_root + "GeneratedWorldRuntime.java": runtime_source,
        java_root + "GeneratedWorldState.java": state_source,
    }
    write_receipt = write_text_files(
        info,
        files,
        replace_existing=True,
    )
    binding = ensure_main_initializer_call(
        info,
        import_line=(
            f"import {runtime_package}.GeneratedWorldRuntime"
        ),
        call_line="GeneratedWorldRuntime.register()",
        marker="world:generated-runtime",
    )
    return {
        "schema_version": "mmm/world-runtime-generation-v1",
        "status": "GENERATED",
        "contract_count": contract_count,
        "piece_contract_count": piece_contract_count,
        "runtime_structure_count": runtime_structure_count,
        "java_files": sorted(files),
        "write_receipt": write_receipt,
        "initializer_binding": binding,
        "persistent_state": f"{mod_id}_generated_world_runtime",
        "runtime_verification": "required",
    }


def _replace_all(source: str, replacements: dict[str, str]) -> str:
    result = source
    for marker, value in replacements.items():
        result = result.replace(marker, value)
    return result


_RUNTIME_JAVA = r'''package __PACKAGE__;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerChunkEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.registry.RegistryKey;
import net.minecraft.resource.Resource;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.function.CommandFunction;
import net.minecraft.server.function.CommandFunctionManager;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.util.Identifier;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.Heightmap;
import net.minecraft.world.World;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public final class GeneratedWorldRuntime {
    private static final String MOD_ID = "__MOD_ID__";
    private static final String CONTRACT_PREFIX = "mmm_world/contracts";
    private static final int CONTRACT_CHECKS_PER_TICK = 64;
    private static final Logger LOGGER =
        LoggerFactory.getLogger(MOD_ID + "/generated-world-runtime");
    private static final List<StructureContract> STRUCTURES = new ArrayList<>();
    private static final Map<String, JsonObject> REGIONS = new HashMap<>();
    private static final Map<String, JsonObject> ROUTES = new HashMap<>();
    private static final Map<String, JsonObject> QUESTS = new HashMap<>();
    private static final List<JsonObject> CONSTRAINTS = new ArrayList<>();
    private static final ArrayDeque<ChunkScan> SCANS = new ArrayDeque<>();
    private static final Set<String> QUEUED_SCANS = new HashSet<>();
    private static boolean registered;

    private GeneratedWorldRuntime() {
    }

    public static synchronized void register() {
        if (registered) {
            return;
        }
        registered = true;
        ServerLifecycleEvents.SERVER_STARTING.register(
            GeneratedWorldRuntime::loadContracts
        );
        ServerLifecycleEvents.END_DATA_PACK_RELOAD.register(
            (server, resources, success) -> {
                if (success) {
                    loadContracts(server);
                }
            }
        );
        ServerLifecycleEvents.SERVER_STOPPED.register(server -> clearRuntime());
        ServerChunkEvents.CHUNK_LOAD.register(
            (world, chunk) -> enqueueScan(world, chunk.getPos())
        );
        ServerTickEvents.END_SERVER_TICK.register(
            GeneratedWorldRuntime::tick
        );
    }

    private static void loadContracts(MinecraftServer server) {
        clearRuntime();
        Map<Identifier, Resource> resources =
            server.getResourceManager().findResources(
                CONTRACT_PREFIX,
                id -> id.getNamespace().equals(MOD_ID)
                    && id.getPath().endsWith(".json")
                    && !id.getPath().contains("/structure_pieces/")
            );
        resources.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> loadContract(entry.getKey(), entry.getValue()));
        STRUCTURES.sort(Comparator.comparing(StructureContract::id));
        LOGGER.info(
            "Loaded {} generated world structures and {} other contracts",
            STRUCTURES.size(),
            REGIONS.size() + ROUTES.size() + QUESTS.size()
                + CONSTRAINTS.size()
        );
    }

    private static void loadContract(Identifier resourceId, Resource resource) {
        try (
            InputStreamReader reader = new InputStreamReader(
                resource.getInputStream(),
                StandardCharsets.UTF_8
            )
        ) {
            JsonObject root = JsonParser.parseReader(reader).getAsJsonObject();
            String schema = string(root, "schema_version");
            if (schema.equals("mmm/world-structure-runtime-v1")) {
                StructureContract contract = StructureContract.parse(root);
                if (contract.runtimePlacement()) {
                    STRUCTURES.add(contract);
                }
            } else if (schema.equals("mmm/world-region-runtime-v1")) {
                REGIONS.put(string(root, "id"), root);
            } else if (schema.equals("mmm/world-route-runtime-v1")) {
                ROUTES.put(string(root, "id"), root);
            } else if (schema.equals("mmm/world-quest-runtime-v1")) {
                QUESTS.put(string(root, "id"), root);
            } else if (schema.equals("mmm/world-constraint-runtime-v1")) {
                CONSTRAINTS.add(root);
            }
        } catch (Exception error) {
            LOGGER.error("Rejected generated world contract {}", resourceId, error);
        }
    }

    private static void clearRuntime() {
        STRUCTURES.clear();
        REGIONS.clear();
        ROUTES.clear();
        QUESTS.clear();
        CONSTRAINTS.clear();
        SCANS.clear();
        QUEUED_SCANS.clear();
    }

    private static void enqueueScan(ServerWorld world, ChunkPos position) {
        if (STRUCTURES.isEmpty()) {
            return;
        }
        String key = world.getRegistryKey().getValue() + ":"
            + position.x + ":" + position.z;
        if (QUEUED_SCANS.add(key)) {
            SCANS.addLast(
                new ChunkScan(
                    key,
                    world.getRegistryKey(),
                    position.x,
                    position.z,
                    0
                )
            );
        }
    }

    private static void tick(MinecraftServer server) {
        scanContracts(server);
        for (ServerWorld world : server.getWorlds()) {
            executeOnePlacementShard(server, world);
        }
    }

    private static void scanContracts(MinecraftServer server) {
        int budget = CONTRACT_CHECKS_PER_TICK;
        while (budget > 0 && !SCANS.isEmpty()) {
            ChunkScan scan = SCANS.removeFirst();
            ServerWorld world = server.getWorld(scan.world());
            if (world == null) {
                QUEUED_SCANS.remove(scan.key());
                continue;
            }
            int cursor = scan.cursor();
            while (budget > 0 && cursor < STRUCTURES.size()) {
                inspectCandidate(
                    world,
                    scan.chunkX(),
                    scan.chunkZ(),
                    STRUCTURES.get(cursor)
                );
                cursor++;
                budget--;
            }
            if (cursor < STRUCTURES.size()) {
                SCANS.addLast(scan.withCursor(cursor));
            } else {
                QUEUED_SCANS.remove(scan.key());
            }
        }
    }

    private static void inspectCandidate(
        ServerWorld world,
        int chunkX,
        int chunkZ,
        StructureContract contract
    ) {
        if (!contract.dimension().equals(
            world.getRegistryKey().getValue().toString()
        )) {
            return;
        }
        if (!isCandidateChunk(chunkX, chunkZ, contract)) {
            return;
        }
        int x = chunkX * 16 + 8;
        int z = chunkZ * 16 + 8;
        int y = contract.anchorY() == null
            ? world.getTopY(Heightmap.Type.WORLD_SURFACE, x, z)
            : contract.anchorY();
        BlockPos anchor = new BlockPos(x, y, z);
        Optional<RegistryKey<net.minecraft.world.biome.Biome>> biomeKey =
            world.getBiome(anchor).getKey();
        String biome = biomeKey
            .map(key -> key.getValue().toString())
            .orElse("");
        if (!contract.biomes().contains(biome)) {
            return;
        }
        if (!constraintsAllow(
            world,
            anchor,
            biome,
            contract.id(),
            contract.regionId()
        )) {
            return;
        }
        GeneratedWorldState state = state(world);
        state.reserve(
            contract.id(),
            anchor,
            contract.shardCount()
        );
    }

    private static boolean isCandidateChunk(
        int chunkX,
        int chunkZ,
        StructureContract contract
    ) {
        int spacing = contract.spacing();
        int spread = spacing - contract.separation();
        int gridX = Math.floorDiv(chunkX, spacing);
        int gridZ = Math.floorDiv(chunkZ, spacing);
        long mixed = mix(gridX, gridZ, contract.salt());
        int offsetX = Math.floorMod((int) mixed, spread);
        int offsetZ = Math.floorMod((int) (mixed >>> 32), spread);
        return chunkX == gridX * spacing + offsetX
            && chunkZ == gridZ * spacing + offsetZ;
    }

    private static long mix(int x, int z, int salt) {
        long value = ((long) x * 341873128712L)
            ^ ((long) z * 132897987541L)
            ^ ((long) salt * 42317861L);
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        value *= 0xc4ceb9fe1a85ec53L;
        return value ^ (value >>> 33);
    }

    private static boolean constraintsAllow(
        ServerWorld world,
        BlockPos anchor,
        String biome,
        String structureId,
        String regionId
    ) {
        JsonObject region = payload(REGIONS.get(regionId));
        if (region != null && !payloadAllows(region, world, anchor, biome)) {
            return false;
        }
        for (JsonObject wrapper : CONSTRAINTS) {
            JsonObject constraint = payload(wrapper);
            if (constraint != null
                && appliesTo(constraint, structureId, regionId)
                && !payloadAllows(constraint, world, anchor, biome)) {
                return false;
            }
        }
        return true;
    }

    private static boolean appliesTo(
        JsonObject constraint,
        String structureId,
        String regionId
    ) {
        if (constraint.has("structure_id")
            && !constraint.get("structure_id").getAsString()
                .equals(structureId)) {
            return false;
        }
        if (constraint.has("structure_ids")
            && !arrayContains(
                constraint.getAsJsonArray("structure_ids"),
                structureId
            )) {
            return false;
        }
        if (constraint.has("region_id")
            && !constraint.get("region_id").getAsString()
                .equals(regionId)) {
            return false;
        }
        return !constraint.has("region_ids")
            || arrayContains(
                constraint.getAsJsonArray("region_ids"),
                regionId
            );
    }

    private static boolean payloadAllows(
        JsonObject payload,
        ServerWorld world,
        BlockPos anchor,
        String biome
    ) {
        if (payload.has("min_y") && anchor.getY() < payload.get("min_y").getAsInt()) {
            return false;
        }
        if (payload.has("max_y") && anchor.getY() > payload.get("max_y").getAsInt()) {
            return false;
        }
        String dimension = world.getRegistryKey().getValue().toString();
        if (payload.has("dimensions")
            && !arrayContains(payload.getAsJsonArray("dimensions"), dimension)) {
            return false;
        }
        return !payload.has("excluded_biomes")
            || !arrayContains(payload.getAsJsonArray("excluded_biomes"), biome);
    }

    private static boolean arrayContains(JsonArray values, String target) {
        for (JsonElement value : values) {
            if (value.isJsonPrimitive()
                && value.getAsString().equals(target)) {
                return true;
            }
        }
        return false;
    }

    private static void executeOnePlacementShard(
        MinecraftServer server,
        ServerWorld world
    ) {
        GeneratedWorldState state = state(world);
        GeneratedWorldState.Job job = state.currentJob();
        if (job == null) {
            return;
        }
        Identifier functionId = new Identifier(
            MOD_ID,
            "generated/" + job.structureId() + "/part_"
                + String.format(Locale.ROOT, "%04d", job.nextShard())
        );
        CommandFunctionManager manager = server.getCommandFunctionManager();
        Optional<CommandFunction> function = manager.getFunction(functionId);
        if (function.isEmpty()) {
            LOGGER.error(
                "Missing generated world function {} for anchor {}",
                functionId,
                job.anchor()
            );
            state.deferCurrentJob();
            return;
        }
        manager.execute(
            function.get(),
            server.getCommandSource()
                .withWorld(world)
                .withPosition(Vec3d.of(job.anchor()))
                .withLevel(2)
                .withSilent()
        );
        state.completeCurrentShard();
    }

    private static GeneratedWorldState state(ServerWorld world) {
        return world.getPersistentStateManager().getOrCreate(
            GeneratedWorldState::fromNbt,
            GeneratedWorldState::new,
            MOD_ID + "_generated_world_runtime"
        );
    }

    public static Optional<JsonObject> region(String id) {
        return Optional.ofNullable(payload(REGIONS.get(id)));
    }

    public static Collection<JsonObject> routes() {
        return ROUTES.values().stream()
            .map(GeneratedWorldRuntime::payload)
            .filter(value -> value != null)
            .toList();
    }

    public static Optional<JsonObject> quest(String id) {
        return Optional.ofNullable(payload(QUESTS.get(id)));
    }

    private static JsonObject payload(JsonObject wrapper) {
        if (wrapper == null || !wrapper.has("payload")
            || !wrapper.get("payload").isJsonObject()) {
            return null;
        }
        return wrapper.getAsJsonObject("payload");
    }

    private static String string(JsonObject value, String key) {
        if (!value.has(key) || !value.get(key).isJsonPrimitive()) {
            throw new IllegalArgumentException("Missing string field " + key);
        }
        return value.get(key).getAsString();
    }

    private record ChunkScan(
        String key,
        RegistryKey<World> world,
        int chunkX,
        int chunkZ,
        int cursor
    ) {
        ChunkScan withCursor(int nextCursor) {
            return new ChunkScan(key, world, chunkX, chunkZ, nextCursor);
        }
    }

    private record StructureContract(
        String id,
        String regionId,
        String dimension,
        int spacing,
        int separation,
        int salt,
        Set<String> biomes,
        Integer anchorY,
        long shardCount,
        boolean runtimePlacement
    ) {
        static StructureContract parse(JsonObject value) {
            int spacing = value.get("spacing").getAsInt();
            int separation = value.get("separation").getAsInt();
            long shardCount = value.get("shard_count").getAsLong();
            if (spacing < 3 || separation < 2 || separation >= spacing
                || shardCount < 0) {
                throw new IllegalArgumentException(
                    "Invalid generated structure placement contract"
                );
            }
            Set<String> biomes = new HashSet<>();
            for (JsonElement item : value.getAsJsonArray("biomes")) {
                biomes.add(item.getAsString());
            }
            Integer anchorY = value.has("anchor_y")
                && !value.get("anchor_y").isJsonNull()
                ? value.get("anchor_y").getAsInt()
                : null;
            return new StructureContract(
                string(value, "id"),
                string(value, "region_id"),
                string(value, "dimension"),
                spacing,
                separation,
                value.get("salt").getAsInt(),
                Set.copyOf(biomes),
                anchorY,
                shardCount,
                string(value, "placement").equals("runtime_function_shards")
            );
        }
    }
}
'''


_STATE_JAVA = r'''package __PACKAGE__;

import net.minecraft.nbt.NbtCompound;
import net.minecraft.nbt.NbtElement;
import net.minecraft.nbt.NbtList;
import net.minecraft.nbt.NbtString;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.PersistentState;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class GeneratedWorldState extends PersistentState {
    private final Set<String> generatedAnchors = new HashSet<>();
    private final List<Job> jobs = new ArrayList<>();
    private int cursor;

    public GeneratedWorldState() {
    }

    public static GeneratedWorldState fromNbt(NbtCompound root) {
        GeneratedWorldState state = new GeneratedWorldState();
        NbtList anchors = root.getList("GeneratedAnchors", NbtElement.STRING_TYPE);
        for (int index = 0; index < anchors.size(); index++) {
            state.generatedAnchors.add(anchors.getString(index));
        }
        NbtList savedJobs = root.getList("Jobs", NbtElement.COMPOUND_TYPE);
        for (int index = 0; index < savedJobs.size(); index++) {
            NbtCompound saved = savedJobs.getCompound(index);
            Job job = new Job(
                saved.getString("Structure"),
                new BlockPos(
                    saved.getInt("X"),
                    saved.getInt("Y"),
                    saved.getInt("Z")
                ),
                saved.getLong("NextShard"),
                saved.getLong("ShardCount")
            );
            if (job.valid()) {
                state.jobs.add(job);
            }
        }
        state.cursor = Math.max(0, root.getInt("Cursor"));
        return state;
    }

    public boolean reserve(
        String structureId,
        BlockPos anchor,
        long shardCount
    ) {
        if (shardCount < 1) {
            return false;
        }
        String key = anchorKey(structureId, anchor);
        if (!generatedAnchors.add(key)) {
            return false;
        }
        jobs.add(new Job(structureId, anchor.toImmutable(), 0, shardCount));
        markDirty();
        return true;
    }

    public Job currentJob() {
        if (jobs.isEmpty()) {
            return null;
        }
        cursor = Math.floorMod(cursor, jobs.size());
        return jobs.get(cursor);
    }

    public void completeCurrentShard() {
        if (jobs.isEmpty()) {
            return;
        }
        cursor = Math.floorMod(cursor, jobs.size());
        Job current = jobs.get(cursor);
        long next = current.nextShard() + 1L;
        if (next >= current.shardCount()) {
            jobs.remove(cursor);
            if (!jobs.isEmpty()) {
                cursor = Math.floorMod(cursor, jobs.size());
            } else {
                cursor = 0;
            }
        } else {
            jobs.set(
                cursor,
                new Job(
                    current.structureId(),
                    current.anchor(),
                    next,
                    current.shardCount()
                )
            );
            cursor = (cursor + 1) % jobs.size();
        }
        markDirty();
    }

    public void deferCurrentJob() {
        if (jobs.size() > 1) {
            cursor = (cursor + 1) % jobs.size();
            markDirty();
        }
    }

    @Override
    public NbtCompound writeNbt(NbtCompound root) {
        NbtList anchors = new NbtList();
        generatedAnchors.stream()
            .sorted()
            .forEach(value -> anchors.add(NbtString.of(value)));
        root.put("GeneratedAnchors", anchors);

        NbtList savedJobs = new NbtList();
        for (Job job : jobs) {
            NbtCompound saved = new NbtCompound();
            saved.putString("Structure", job.structureId());
            saved.putInt("X", job.anchor().getX());
            saved.putInt("Y", job.anchor().getY());
            saved.putInt("Z", job.anchor().getZ());
            saved.putLong("NextShard", job.nextShard());
            saved.putLong("ShardCount", job.shardCount());
            savedJobs.add(saved);
        }
        root.put("Jobs", savedJobs);
        root.putInt("Cursor", cursor);
        return root;
    }

    private static String anchorKey(String structureId, BlockPos anchor) {
        return structureId + ":" + anchor.getX() + ":"
            + anchor.getY() + ":" + anchor.getZ();
    }

    public record Job(
        String structureId,
        BlockPos anchor,
        long nextShard,
        long shardCount
    ) {
        boolean valid() {
            return !structureId.isBlank()
                && nextShard >= 0
                && shardCount > 0
                && nextShard < shardCount;
        }
    }
}
'''
