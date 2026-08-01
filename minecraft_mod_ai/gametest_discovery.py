from __future__ import annotations


def discovered_gametest_root_java(
    *,
    package_name: str,
    mod_id: str,
    root_class_name: str,
    unit_class_prefix: str,
) -> str:
    """Render one fixed-size GameTest entrypoint that discovers bounded units."""

    package_path = package_name.replace(".", "/")
    return f'''package {package_name};

import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.test.GameTest;
import net.minecraft.test.TestContext;

import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.TreeSet;

public final class {root_class_name} {{
    @GameTest(templateName = FabricGameTest.EMPTY_STRUCTURE)
    public void generatedRegistriesAreLive(TestContext context) {{
        for (String className : generatedUnitClasses()) {{
            invokeUnit(className, context);
        }}
        context.complete();
    }}

    private static Set<String> generatedUnitClasses() {{
        Set<String> classes = new TreeSet<>();
        String relative = "{package_path}";
        FabricLoader.getInstance()
            .getModContainer("{mod_id}")
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

    private static void invokeUnit(String className, TestContext context) {{
        try {{
            Class<?> unit = Class.forName(
                className,
                true,
                {root_class_name}.class.getClassLoader()
            );
            unit.getMethod("run", TestContext.class).invoke(null, context);
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
