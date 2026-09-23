"""Opt-in real Fabric bytecode stripping regression; no Minecraft/LLM substitute claim.

Set MMM_FABRIC_RUNTIME_TEST_CLASSPATH to Fabric Loader and ASM jars (path separator).
"""

import os
import shutil
import subprocess

import pytest

from minecraft_mod_ai.authored_feature_source import authored_feature_source_diagnostics


def test_actual_fabric_stripping_reproduces_and_repairs_missing_initialize(tmp_path):
    dependencies = os.environ.get("MMM_FABRIC_RUNTIME_TEST_CLASSPATH", "")
    javac, java = shutil.which("javac"), shutil.which("java")
    if not dependencies or not javac or not java:
        pytest.skip("requires Java and explicit Fabric Loader/ASM test jars")
    source = tmp_path / "AuthoredFeature001.java"
    broken = """package probe;
import net.fabricmc.api.Environment;
import net.fabricmc.api.EnvType;
public final class AuthoredFeature001 {
    @Environment(EnvType.CLIENT)
    public static void initialize() { System.setProperty("probe.initialized", "yes"); }
}
"""
    entry = tmp_path / "Entry.java"
    entry.write_text(
        "package probe; public class Entry { public static void run() { AuthoredFeature001.initialize(); } }",
        encoding="utf-8",
    )
    harness = tmp_path / "Harness.java"
    harness.write_text(
        """import java.nio.file.*;
import java.lang.reflect.*;
import org.objectweb.asm.*;
import net.fabricmc.loader.impl.transformer.*;
public class Harness extends ClassLoader {
    Class<?> define(String name, byte[] code) { return defineClass(name, code, 0, code.length); }
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]);
        ClassReader reader = new ClassReader(Files.readAllBytes(root.resolve("probe/AuthoredFeature001.class")));
        EnvironmentStrippingData environment = new EnvironmentStrippingData(Opcodes.ASM9, "SERVER");
        reader.accept(environment, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG);
        ClassWriter writer = new ClassWriter(0);
        reader.accept(new ClassStripper(Opcodes.ASM9, writer, environment.getStripInterfaces(), environment.getStripFields(), environment.getStripMethods()), 0);
        Harness loader = new Harness();
        loader.define("probe.AuthoredFeature001", writer.toByteArray());
        Class<?> entry = loader.define("probe.Entry", Files.readAllBytes(root.resolve("probe/Entry.class")));
        try {
            entry.getMethod("run").invoke(null);
            if (!"yes".equals(System.getProperty("probe.initialized"))) throw new AssertionError("initialize did not execute");
            System.out.println("INITIALIZE_EXECUTED");
        } catch (InvocationTargetException exception) {
            if (!(exception.getCause() instanceof NoSuchMethodError)) throw exception;
            System.out.println(exception.getCause());
            System.exit(17);
        }
    }
}
""",
        encoding="utf-8",
    )
    outputs = []
    for text, expected in (
        (broken, 17),
        (broken.replace("@Environment(EnvType.CLIENT)", ""), 0),
    ):
        source.write_text(text, encoding="utf-8")
        diagnostics = authored_feature_source_diagnostics(
            text,
            path="src/main/java/probe/AuthoredFeature001.java",
            symbol="AuthoredFeature001",
        )
        assert bool(diagnostics) == (expected != 0)
        compiled = subprocess.run(
            [
                javac,
                "-cp",
                dependencies,
                "-d",
                str(tmp_path),
                str(source),
                str(entry),
                str(harness),
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert compiled.returncode == 0, compiled.stderr
        ran = subprocess.run(
            [
                java,
                "-cp",
                str(tmp_path) + os.pathsep + dependencies,
                "Harness",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert ran.returncode == expected, ran.stdout + ran.stderr
        outputs.append(ran.stdout)
    assert "NoSuchMethodError" in outputs[0]
    assert "INITIALIZE_EXECUTED" in outputs[1]
