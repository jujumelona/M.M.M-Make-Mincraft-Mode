"""Read exported APIs from the exact installed JDK's modular runtime image.

JMOD archives are optional distribution artifacts. ModuleFinder/ModuleReader
provide the actual class bytes without an additional jimage executable or SDK.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import RLock

from .jar_api_extractor import ApiClass, inspect_jar


_READER_SOURCE = r'''
import java.io.InputStream;
import java.lang.module.ModuleDescriptor;
import java.lang.module.ModuleFinder;
import java.lang.module.ModuleReader;
import java.lang.module.ModuleReference;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

public final class JdkRuntimeInventory {
    public static void main(String[] args) throws Exception {
        Path expected = Paths.get(args[0]).toRealPath();
        Path actual = Paths.get(System.getProperty("java.home"), "lib", "modules").toRealPath();
        if (!expected.equals(actual)) {
            throw new IllegalStateException("JDK_RUNTIME_IMAGE_MISMATCH: " + actual);
        }
        if (Runtime.version().major() != Integer.parseInt(args[2])) {
            throw new IllegalStateException("JDK_RUNTIME_VERSION_MISMATCH: " + Runtime.version());
        }
        TreeMap<String, ModuleReference> modules = new TreeMap<>();
        for (ModuleReference reference : ModuleFinder.ofSystem().findAll()) {
            String name = reference.descriptor().name();
            // Inventory the same default module graph as unnamed-module javac.
            // Installed incubator modules require explicit compiler opt-in.
            if (ModuleLayer.boot().findModule(name).isPresent()) modules.put(name, reference);
        }
        Set<String> written = new HashSet<>();
        try (ZipOutputStream archive = new ZipOutputStream(Files.newOutputStream(Paths.get(args[1])))) {
            for (ModuleReference reference : modules.values()) {
                Set<String> exported = new HashSet<>();
                for (ModuleDescriptor.Exports entry : reference.descriptor().exports()) {
                    if (!entry.isQualified()) exported.add(entry.source());
                }
                try (ModuleReader reader = reference.open()) {
                    List<String> names;
                    try (Stream<String> resources = reader.list()) {
                        names = resources.filter(name -> name.endsWith(".class")
                            && !name.equals("module-info.class")).sorted().collect(Collectors.toList());
                    }
                    for (String name : names) {
                        int slash = name.lastIndexOf('/');
                        if (slash < 0 || !exported.contains(name.substring(0, slash).replace('/', '.'))) continue;
                        if (!written.add(name)) throw new IllegalStateException("JDK_RUNTIME_CLASS_DUPLICATE: " + name);
                        ZipEntry entry = new ZipEntry(name);
                        entry.setTime(0L);
                        archive.putNextEntry(entry);
                        try (InputStream source = reader.open(name).orElseThrow(
                                () -> new IllegalStateException("JDK_RUNTIME_CLASS_MISSING: " + name))) {
                            source.transferTo(archive);
                        }
                        archive.closeEntry();
                    }
                }
            }
        }
        if (!written.contains("java/lang/Object.class")) {
            throw new IllegalStateException("JDK_RUNTIME_BASE_API_MISSING");
        }
    }
}
'''

_INSPECTION_LOCK = RLock()


def _run_reader(command: list[str], *, cwd: Path, stage: str, timeout_seconds: int) -> None:
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"CONSTRUCTION_JDK_IMAGE_{stage}_TIMEOUT") from exc
    if result.returncode:
        raise ValueError(
            f"CONSTRUCTION_JDK_IMAGE_{stage}_FAILED: " + (result.stderr or result.stdout)[-4000:]
        )


@lru_cache(maxsize=2)
def _inspect_image(home_text: str, image_text: str, java_version: int, side: str,
                   signature: tuple[int, int]) -> tuple[ApiClass, ...]:
    image = Path(image_text)
    home = Path(home_text)
    suffix = ".exe" if os.name == "nt" else ""
    java, javac = home / "bin" / ("java" + suffix), home / "bin" / ("javac" + suffix)
    if not java.is_file() or not javac.is_file():
        raise ValueError("CONSTRUCTION_JDK_IMAGE_TOOLCHAIN_MISSING: " + str(home))
    with TemporaryDirectory(prefix="mmm-jdk-inventory-") as directory:
        root = Path(directory)
        source, archive = root / "JdkRuntimeInventory.java", root / "jdk-api.jar"
        source.write_text(_READER_SOURCE, encoding="utf-8")
        _run_reader(
            [str(javac), "-encoding", "UTF-8", "-classpath", str(root), "-d", str(root), str(source)],
            cwd=root, stage="COMPILER", timeout_seconds=60,
        )
        _run_reader(
            [str(java), "-cp", str(root), "JdkRuntimeInventory", str(image), str(archive), str(java_version)],
            cwd=root, stage="READER", timeout_seconds=120,
        )
        inspected = inspect_jar(archive, java_version=java_version, default_side=side)
    # An exported package can also contain private implementation classes.
    # Admit only public classes whose enclosing classes are public too.
    def public_type(name: str) -> bool:
        while name in inspected:
            if not inspected[name].access & 1:
                return False
            if "$" not in name:
                break
            name = name.rsplit("$", 1)[0]
        return True

    state = image.stat()
    if (state.st_size, state.st_mtime_ns) != signature:
        raise ValueError("CONSTRUCTION_JDK_IMAGE_CHANGED_DURING_INSPECTION")
    return tuple(item for name, item in sorted(inspected.items()) if public_type(name))


def inspect_jdk_runtime_image(image: Path, *, java_version: int,
                              default_side: str = "COMMON") -> tuple[ApiClass, ...]:
    image = Path(image).expanduser()
    if image.name != "modules" or image.parent.name != "lib" or not image.is_file():
        raise ValueError("CONSTRUCTION_JDK_RUNTIME_IMAGE_REQUIRED: " + str(image))
    # A module image may be linked outside its JDK home. Preserve the selected
    # executable home while canonicalizing only the image identity/provenance.
    home = image.parent.parent.resolve()
    image = image.resolve()
    state = image.stat()
    with _INSPECTION_LOCK:
        return _inspect_image(
            str(home), str(image), java_version, default_side, (state.st_size, state.st_mtime_ns),
        )
