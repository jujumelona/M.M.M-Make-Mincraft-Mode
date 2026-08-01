from __future__ import annotations


def _persistent_store_java(package_name: str, mod_id: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.WorldSavePath;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

public final class MmmPersistentStore {{
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final Type TYPE = new TypeToken<Map<String, Map<String, Object>>>() {{}}.getType();
    private static final Map<String, Map<String, Object>> DATA = new LinkedHashMap<>();
    private static boolean lifecycleRegistered;

    private MmmPersistentStore() {{}}

    public static synchronized void registerLifecycle() {{
        if (lifecycleRegistered) return;
        lifecycleRegistered = true;
        ServerLifecycleEvents.SERVER_STARTED.register(MmmPersistentStore::load);
        ServerLifecycleEvents.SERVER_STOPPING.register(MmmPersistentStore::save);
    }}

    public static synchronized Map<String, Object> namespace(String id) {{
        return DATA.computeIfAbsent(id, ignored -> new LinkedHashMap<>());
    }}

    public static synchronized void load(MinecraftServer server) {{
        Path file = dataFile(server);
        DATA.clear();
        if (!Files.isRegularFile(file)) return;
        try (Reader reader = Files.newBufferedReader(file)) {{
            Map<String, Map<String, Object>> loaded = GSON.fromJson(reader, TYPE);
            if (loaded != null) DATA.putAll(loaded);
        }} catch (IOException exception) {{
            throw new IllegalStateException("Could not load M.M.M persistent state", exception);
        }}
    }}

    public static synchronized void save(MinecraftServer server) {{
        Path file = dataFile(server);
        Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
        try {{
            Files.createDirectories(file.getParent());
            try (Writer writer = Files.newBufferedWriter(temporary)) {{
                GSON.toJson(DATA, TYPE, writer);
            }}
            try {{
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            }} catch (IOException atomicUnsupported) {{
                Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING);
            }}
        }} catch (IOException exception) {{
            throw new IllegalStateException("Could not save M.M.M persistent state", exception);
        }}
    }}

    private static Path dataFile(MinecraftServer server) {{
        return server.getSavePath(WorldSavePath.ROOT).resolve("data/{mod_id}_mmm_systems.json");
    }}
}}
'''


def _config_loader_java(package_name: str, mod_id: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.fabricmc.loader.api.FabricLoader;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Consumer;

public final class MmmSystemConfig {{
    private static final Gson GSON = new Gson();
    private MmmSystemConfig() {{}}

    public static JsonObject load(String absoluteResourcePath) {{
        try (InputStream stream = MmmSystemConfig.class.getResourceAsStream(absoluteResourcePath)) {{
            if (stream == null) throw new IllegalStateException("Missing system resource: " + absoluteResourcePath);
            return GSON.fromJson(new InputStreamReader(stream, StandardCharsets.UTF_8), JsonObject.class);
        }} catch (Exception exception) {{
            throw new IllegalStateException("Could not load system resource: " + absoluteResourcePath, exception);
        }}
    }}

    public static void forEachModule(
        String absoluteResourcePath,
        Consumer<JsonObject> visitor
    ) {{
        JsonObject catalog = load(absoluteResourcePath);
        if (catalog.has("modules")) {{
            visitModules(catalog.getAsJsonArray("modules"), visitor);
            return;
        }}
        String storage = catalog.get("storage_schema_version").getAsString();
        int expected = catalog.get("module_count").getAsInt();
        if ("mmm/system-pack-directory-v1".equals(storage)) {{
            visitDirectory(
                resourcePath(catalog.get("directory")),
                expected,
                visitor
            );
            return;
        }}
        if (!"mmm/system-pack-index-v1".equals(storage)) {{
            throw new IllegalStateException(
                "Unsupported system catalog: " + absoluteResourcePath
            );
        }}
        int visitedModules = 0;
        ArrayDeque<String> pending = new ArrayDeque<>();
        Set<String> visitedPaths = new HashSet<>();
        pending.add(resourcePath(catalog.get("root")));
        while (!pending.isEmpty()) {{
            String path = pending.removeFirst();
            if (!visitedPaths.add(path)) {{
                throw new IllegalStateException("System catalog cycle: " + path);
            }}
            JsonObject node = load(path);
            String schema = node.get("schema_version").getAsString();
            if ("mmm/system-module-shard-v1".equals(schema)) {{
                JsonArray modules = node.getAsJsonArray("modules");
                visitedModules += modules.size();
                visitModules(modules, visitor);
                continue;
            }}
            if (!"mmm/system-module-index-node-v1".equals(schema)) {{
                throw new IllegalStateException(
                    "Unsupported system catalog node: " + path
                );
            }}
            JsonArray children = node.getAsJsonArray("children");
            if (children.size() == 0) {{
                throw new IllegalStateException(
                    "Empty system catalog index: " + path
                );
            }}
            for (int index = children.size() - 1; index >= 0; index--) {{
                pending.addFirst(resourcePath(children.get(index)));
            }}
        }}
        if (visitedModules != expected) {{
            throw new IllegalStateException(
                "System module count mismatch: expected "
                    + expected + ", loaded " + visitedModules
            );
        }}
    }}

    private static void visitDirectory(
        String absoluteDirectory,
        int expected,
        Consumer<JsonObject> visitor
    ) {{
        Map<String, Path> records = new TreeMap<>();
        String relative = absoluteDirectory.substring(1);
        FabricLoader.getInstance()
            .getModContainer("{mod_id}")
            .orElseThrow()
            .getRootPaths()
            .forEach(root -> collectRecords(root.resolve(relative), records));
        if (records.size() != expected) {{
            throw new IllegalStateException(
                "System module count mismatch: expected "
                    + expected + ", found " + records.size()
            );
        }}
        for (Map.Entry<String, Path> entry : records.entrySet()) {{
            JsonObject record = loadFile(entry.getValue());
            if (!"mmm/system-module-record-v1".equals(
                record.get("schema_version").getAsString()
            )) {{
                throw new IllegalStateException(
                    "Unsupported system module record: " + entry.getKey()
                );
            }}
            JsonObject module = record.getAsJsonObject("module");
            String moduleId = module.get("module_id").getAsString();
            if (!entry.getKey().equals(moduleId + ".json")) {{
                throw new IllegalStateException(
                    "System module record filename mismatch: " + entry.getKey()
                );
            }}
            visitor.accept(module);
        }}
    }}

    private static void collectRecords(
        Path directory,
        Map<String, Path> records
    ) {{
        if (!Files.isDirectory(directory) || Files.isSymbolicLink(directory)) {{
            return;
        }}
        try (var paths = Files.list(directory)) {{
            paths.filter(path ->
                Files.isRegularFile(path)
                    && !Files.isSymbolicLink(path)
                    && path.getFileName().toString().endsWith(".json")
            ).forEach(path -> {{
                String name = path.getFileName().toString();
                Path previous = records.putIfAbsent(name, path);
                if (previous != null && !previous.equals(path)) {{
                    throw new IllegalStateException(
                        "Duplicate system module record: " + name
                    );
                }}
            }});
        }} catch (java.io.IOException error) {{
            throw new IllegalStateException(
                "Could not enumerate system module records",
                error
            );
        }}
    }}

    private static JsonObject loadFile(Path path) {{
        try (var reader = Files.newBufferedReader(
            path,
            StandardCharsets.UTF_8
        )) {{
            return GSON.fromJson(reader, JsonObject.class);
        }} catch (Exception error) {{
            throw new IllegalStateException(
                "Could not load system module record: " + path,
                error
            );
        }}
    }}

    private static void visitModules(
        JsonArray modules,
        Consumer<JsonObject> visitor
    ) {{
        for (JsonElement element : modules) {{
            visitor.accept(element.getAsJsonObject());
        }}
    }}

    private static String resourcePath(JsonElement raw) {{
        String path = raw.getAsString();
        if (
            !path.startsWith("/data/")
                || path.contains("..")
                || path.contains("\\\\")
        ) {{
            throw new IllegalStateException(
                "Unsafe system catalog resource path: " + path
            );
        }}
        return path;
    }}
}}
'''
