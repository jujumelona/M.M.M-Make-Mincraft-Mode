from __future__ import annotations

import re


def modern_fabric_gametest_api(minecraft_version: str) -> bool:
    """Return whether Fabric's v2 GameTest annotation API is required."""

    numbers = [int(part) for part in re.findall(r"\d+", str(minecraft_version))]
    if not numbers:
        return False
    if numbers[0] >= 2:
        return True
    major = numbers[0]
    minor = numbers[1] if len(numbers) > 1 else 0
    patch = numbers[2] if len(numbers) > 2 else 0
    return (major, minor, patch) >= (1, 21, 5)


def discovered_gametest_root_java(
    *,
    package_name: str,
    mod_id: str,
    root_class_name: str,
    unit_class_prefix: str,
    mappings_kind: str = "yarn",
    minecraft_version: str = "",
) -> str:
    """Render one fixed-size GameTest entrypoint for the locked platform."""

    package_path = package_name.replace(".", "/")
    normalized = str(mappings_kind or "").strip().casefold()
    mojang = normalized in {"mojang", "official", "official_mojang"}
    modern = modern_fabric_gametest_api(minecraft_version)

    if mojang:
        context_import = "import net.minecraft.gametest.framework.GameTestHelper;"
        context_type = "GameTestHelper"
        complete = "context.succeed();"
    else:
        context_import = "import net.minecraft.test.TestContext;"
        context_type = "TestContext"
        complete = "context.complete();"

    if modern:
        api_imports = "import net.fabricmc.fabric.api.gametest.v1.GameTest;"
        annotation = "@GameTest"
    elif mojang:
        api_imports = """import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.gametest.framework.GameTest;"""
        annotation = "@GameTest(template = FabricGameTest.EMPTY_STRUCTURE)"
    else:
        api_imports = """import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.minecraft.test.GameTest;"""
        annotation = "@GameTest(templateName = FabricGameTest.EMPTY_STRUCTURE)"

    return f'''package {package_name};

{api_imports}
import net.fabricmc.loader.api.FabricLoader;
{context_import}

import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.TreeSet;

public final class {root_class_name} {{
    {annotation}
    public void generatedRegistriesAreLive({context_type} context) {{
        for (String className : generatedUnitClasses()) {{
            invokeUnit(className, context);
        }}
        {complete}
    }}

    private static Set<String> generatedUnitClasses() {{
        Set<String> classes = new TreeSet<>();
        String relative = "{package_path}";
        FabricLoader.getInstance()
            .getModContainer("{mod_id}_gametest")
            .orElseThrow()
            .getRootPaths()
            .forEach(root -> collectUnits(root.resolve(relative), classes));
        return classes;
    }}

    private static void collectUnits(Path directory, Set<String> classes) {{
        if (!Files.isDirectory(directory)) return;
        try (var paths = Files.list(directory)) {{
            paths.filter(path -> {{
                String name = path.getFileName().toString();
                return name.startsWith("{unit_class_prefix}")
                    && name.endsWith(".class");
            }}).forEach(path -> {{
                String name = path.getFileName().toString();
                classes.add(
                    "{package_name}." + name.substring(0, name.length() - 6)
                );
            }});
        }} catch (java.io.IOException error) {{
            throw new IllegalStateException(
                "Could not enumerate generated GameTest units",
                error
            );
        }}
    }}

    private static void invokeUnit(String className, {context_type} context) {{
        try {{
            Class<?> unit = Class.forName(
                className,
                true,
                {root_class_name}.class.getClassLoader()
            );
            unit.getMethod("run", {context_type}.class).invoke(null, context);
        }} catch (InvocationTargetException error) {{
            Throwable cause = error.getCause();
            if (cause instanceof RuntimeException runtime) throw runtime;
            if (cause instanceof Error fatal) throw fatal;
            throw new IllegalStateException(
                "Generated GameTest unit failed: " + className,
                cause
            );
        }} catch (ReflectiveOperationException error) {{
            throw new IllegalStateException(
                "Could not run generated GameTest unit " + className,
                error
            );
        }}
    }}
}}
'''
