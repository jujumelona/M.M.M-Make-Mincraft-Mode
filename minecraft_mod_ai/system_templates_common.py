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


def _config_loader_java(package_name: str) -> str:
    return f'''package {package_name}.system;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

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
}}
'''
