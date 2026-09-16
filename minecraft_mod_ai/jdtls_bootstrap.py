from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

JDTLS_VERSION = "1.60.0"
JDTLS_BUILD = "202606262232"
JDTLS_ARCHIVE = f"jdt-language-server-{JDTLS_VERSION}-{JDTLS_BUILD}.tar.gz"
JDTLS_BASE_URL = f"https://download.eclipse.org/jdtls/milestones/{JDTLS_VERSION}"
ADOPTIUM_BASE_URL = "https://api.adoptium.net/v3"
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_JDK_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_JDK_METADATA_BYTES = 1024 * 1024
_JAVA_VERSION = re.compile(r'version\s+"(?P<major>\d+)')
_JAVA_MAJOR_SETTING = re.compile(r"^(?:JavaSE-)?(?P<major>\d+)$", re.IGNORECASE)
_SHA256 = re.compile(r"\b(?P<digest>[0-9a-fA-F]{64})\b")


class JDTLSBootstrapError(RuntimeError):
    pass


def _cache_root() -> Path:
    configured = os.environ.get("MMM_JDTLS_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / ".cache"
        / "mmm"
        / "jdtls"
        / f"{JDTLS_VERSION}-{JDTLS_BUILD}"
    ).resolve()


def _java_major(java: str) -> int | None:
    try:
        output = subprocess.check_output(
            [java, "-version"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _JAVA_VERSION.search(output)
    return int(match.group("major")) if match else None


def _requested_project_java_major() -> int | None:
    raw = os.environ.get("MMM_JAVA_VERSION", "").strip()
    if not raw:
        return None
    match = _JAVA_MAJOR_SETTING.fullmatch(raw)
    if match is None:
        raise JDTLSBootstrapError(
            "MMM_JAVA_VERSION must be a Java major such as '21', '25', or 'JavaSE-25'; "
            f"got {raw!r}."
        )
    major = int(match.group("major"))
    if major < 1:
        raise JDTLSBootstrapError(f"MMM_JAVA_VERSION must be positive; got {raw!r}.")
    return major


def _jdk_java(home: Path) -> Path:
    executable = "java.exe" if os.name == "nt" else "java"
    return home / "bin" / executable


def _candidate_jdk_homes() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("MMM_PROJECT_JAVA_HOME", "JAVA_HOME"):
        configured = os.environ.get(variable, "").strip()
        if configured:
            candidates.append(Path(configured).expanduser())

    if sys.platform.startswith("linux"):
        for root in (Path.home() / ".jdks", Path("/usr/lib/jvm")):
            if root.is_dir():
                candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    elif sys.platform == "darwin":
        java_vm_root = Path("/Library/Java/JavaVirtualMachines")
        if java_vm_root.is_dir():
            candidates.extend(
                sorted(path / "Contents" / "Home" for path in java_vm_root.iterdir() if path.is_dir())
            )

    current = shutil.which("java")
    if current:
        resolved = Path(current).resolve()
        if len(resolved.parents) >= 2:
            candidates.append(resolved.parent.parent)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _find_matching_jdk(major: int) -> Path | None:
    for home in _candidate_jdk_homes():
        java = _jdk_java(home)
        if java.is_file() and os.access(java, os.X_OK) and _java_major(str(java)) == major:
            return home
    return None


def _adoptium_architecture() -> str:
    machine = platform.machine().strip().lower()
    aliases = {
        "x86_64": "x64",
        "amd64": "x64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    architecture = aliases.get(machine)
    if architecture is None:
        raise JDTLSBootstrapError(
            f"Automatic project JDK provisioning does not support architecture {machine!r}."
        )
    return architecture


def _read_json(url: str, *, max_bytes: int) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "M.M.M-JDTLS-bootstrap/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read(max_bytes + 1)
    if not payload:
        raise JDTLSBootstrapError("Project JDK metadata response was empty.")
    if len(payload) > max_bytes:
        raise JDTLSBootstrapError(
            f"Project JDK metadata exceeded the {max_bytes}-byte safety limit."
        )
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JDTLSBootstrapError("Project JDK metadata response was invalid JSON.") from exc


def _project_jdk_package(major: int) -> tuple[str, str]:
    if not sys.platform.startswith("linux"):
        raise JDTLSBootstrapError(
            "Automatic project JDK provisioning is currently supported only on Linux/Colab. "
            f"Install Java {major} and set MMM_PROJECT_JAVA_HOME explicitly."
        )
    architecture = _adoptium_architecture()
    url = (
        f"{ADOPTIUM_BASE_URL}/assets/latest/{major}/hotspot"
        f"?architecture={architecture}&image_type=jdk&os=linux"
    )
    metadata = _read_json(url, max_bytes=_MAX_JDK_METADATA_BYTES)
    if not isinstance(metadata, list) or not metadata:
        raise JDTLSBootstrapError(
            f"Adoptium returned no Linux {architecture} JDK asset for Java {major}."
        )
    for asset in metadata:
        if not isinstance(asset, dict):
            continue
        binary = asset.get("binary")
        if not isinstance(binary, dict):
            continue
        package = binary.get("package")
        if not isinstance(package, dict):
            continue
        link = package.get("link")
        checksum = package.get("checksum")
        if (
            isinstance(link, str)
            and link.startswith("https://")
            and isinstance(checksum, str)
            and _SHA256.fullmatch(checksum)
        ):
            download_url = (
                f"{ADOPTIUM_BASE_URL}/binary/latest/{major}/ga/linux/{architecture}"
                "/jdk/hotspot/normal/eclipse"
            )
            return download_url, checksum.lower()
    raise JDTLSBootstrapError(
        f"Adoptium metadata for Java {major} did not contain a verifiable JDK package."
    )


def _ensure_java_21() -> None:
    for variable in ("MMM_PROJECT_JAVA_HOME", "JAVA_HOME"):
        java_home = os.environ.get(variable, "").strip()
        if java_home:
            configured = _jdk_java(Path(java_home).expanduser())
            if configured.is_file() and (_java_major(str(configured)) or 0) >= 21:
                os.environ["JAVA_HOME"] = str(configured.resolve().parent.parent)
                return

    current = shutil.which("java")
    if current and (_java_major(current) or 0) >= 21:
        resolved = Path(current).resolve()
        os.environ["JAVA_HOME"] = str(resolved.parent.parent)
        return
    candidates = (
        Path("/usr/lib/jvm/java-21-openjdk-amd64/bin/java"),
        Path("/usr/lib/jvm/java-21-openjdk/bin/java"),
        Path("/usr/lib/jvm/temurin-21-jdk-amd64/bin/java"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK) and (_java_major(str(candidate)) or 0) >= 21:
            os.environ["JAVA_HOME"] = str(candidate.parent.parent)
            return
    found = _java_major(current) if current else None
    detail = f"Java {found}" if found is not None else "no Java runtime"
    raise JDTLSBootstrapError(
        f"Eclipse JDT LS {JDTLS_VERSION} requires Java 21+; found {detail}."
    )


@contextmanager
def _install_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            import fcntl
        except ImportError:
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _download(url: str, destination: Path, *, max_bytes: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "M.M.M-JDTLS-bootstrap/1"})
    total = 0
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise JDTLSBootstrapError(f"Bootstrap download exceeded {max_bytes} bytes.")
            output.write(chunk)
    if total == 0:
        raise JDTLSBootstrapError("Bootstrap download returned an empty response.")


def _expected_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="strict")
    match = _SHA256.search(text)
    if match is None:
        raise JDTLSBootstrapError("Eclipse JDT LS checksum response is invalid.")
    return match.group("digest").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_members(
    members: list[tarfile.TarInfo],
    destination: Path,
    *,
    label: str,
    allow_links: bool,
) -> None:
    destination_root = destination.resolve()
    for member in members:
        if member.isdev():
            raise JDTLSBootstrapError(f"Unsafe device entry in {label} archive: {member.name!r}.")
        candidate = (destination / member.name).resolve()
        try:
            candidate.relative_to(destination_root)
        except ValueError as exc:
            raise JDTLSBootstrapError(
                f"Archive entry escapes {label} install root: {member.name!r}."
            ) from exc
        if member.issym() or member.islnk():
            if not allow_links:
                raise JDTLSBootstrapError(f"Unsafe entry in {label} archive: {member.name!r}.")
            if member.issym():
                target = (candidate.parent / member.linkname).resolve()
            else:
                target = (destination / member.linkname).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError as exc:
                raise JDTLSBootstrapError(
                    f"Archive link escapes {label} install root: {member.name!r} -> {member.linkname!r}."
                ) from exc


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        _validate_archive_members(
            members,
            destination,
            label="Eclipse JDT LS",
            allow_links=False,
        )
        archive.extractall(destination, members=members)


def _safe_extract_jdk(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        _validate_archive_members(
            members,
            destination,
            label="project JDK",
            allow_links=True,
        )
        archive.extractall(destination, members=members)


def _find_extracted_jdk(root: Path, major: int) -> Path | None:
    candidates = sorted(
        {
            java.parent.parent
            for java in root.glob("*/bin/java")
            if java.is_file() and os.access(java, os.X_OK)
        }
    )
    matches = [home for home in candidates if _java_major(str(home / "bin" / "java")) == major]
    return matches[0] if len(matches) == 1 else None


def _install_project_jdk(major: int) -> Path:
    jdks_root = (Path.home() / ".jdks").resolve()
    target = jdks_root / f"mmm-temurin-{major}"
    jdks_root.mkdir(parents=True, exist_ok=True)
    with _install_lock(jdks_root / ".mmm-install.lock"):
        existing = _find_matching_jdk(major)
        if existing is not None:
            return existing

        package_url, expected = _project_jdk_package(major)
        with tempfile.TemporaryDirectory(prefix=".mmm-jdk-install-", dir=jdks_root) as temporary:
            stage = Path(temporary)
            archive_path = stage / f"jdk-{major}.tar.gz"
            _download(package_url, archive_path, max_bytes=_MAX_JDK_ARCHIVE_BYTES)
            observed = _sha256(archive_path)
            if observed != expected:
                raise JDTLSBootstrapError(
                    "Project JDK archive checksum mismatch: "
                    f"expected {expected}, observed {observed}."
                )
            extracted = stage / "extracted"
            extracted.mkdir()
            _safe_extract_jdk(archive_path, extracted)
            install_source = _find_extracted_jdk(extracted, major)
            if install_source is None:
                raise JDTLSBootstrapError(
                    f"Downloaded project JDK archive did not contain exactly one Java {major} runtime."
                )
            if target.exists():
                shutil.rmtree(target)
            os.replace(install_source, target)

        java = _jdk_java(target)
        observed_major = _java_major(str(java))
        if observed_major != major:
            if target.exists():
                shutil.rmtree(target)
            raise JDTLSBootstrapError(
                f"Installed project JDK reports Java {observed_major}, expected Java {major}."
            )
        receipt = {
            "schema_version": "mmm/project-jdk-install-v1",
            "major": major,
            "provider": "adoptium",
            "sha256": expected,
        }
        (target / ".mmm-install.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return target.resolve()


def ensure_project_jdk(required_major: int | None = None) -> Path | None:
    """Return the exact project JDK, provisioning it only when verification needs it."""
    major = required_major if required_major is not None else _requested_project_java_major()
    if major is None:
        return None
    if major <= 0:
        raise JDTLSBootstrapError(f"Project Java major must be positive; got {major!r}.")
    home = _find_matching_jdk(major)
    if home is None:
        home = _install_project_jdk(major)
    java = _jdk_java(home)
    observed = _java_major(str(java))
    if observed != major:
        raise JDTLSBootstrapError(
            f"Project JDK validation failed: requested Java {major}, found Java {observed} at {home}."
        )
    resolved = home.resolve()
    os.environ["MMM_PROJECT_JAVA_HOME"] = str(resolved)
    return resolved


def _ensure_project_jdk() -> Path | None:
    return ensure_project_jdk()


def _find_launcher(root: Path) -> Path | None:
    direct = root / "bin" / "jdtls"
    if direct.is_file():
        return direct
    candidates = sorted(root.glob("*/bin/jdtls"))
    return candidates[0] if len(candidates) == 1 and candidates[0].is_file() else None


def _install_jdtls(root: Path) -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jdtls-install-", dir=root.parent) as temporary:
        stage = Path(temporary)
        archive_path = stage / JDTLS_ARCHIVE
        checksum_path = stage / f"{JDTLS_ARCHIVE}.sha256"
        _download(f"{JDTLS_BASE_URL}/{JDTLS_ARCHIVE}", archive_path, max_bytes=_MAX_ARCHIVE_BYTES)
        _download(
            f"{JDTLS_BASE_URL}/{JDTLS_ARCHIVE}.sha256",
            checksum_path,
            max_bytes=4096,
        )
        expected = _expected_sha256(checksum_path)
        observed = _sha256(archive_path)
        if observed != expected:
            raise JDTLSBootstrapError(
                "Eclipse JDT LS archive checksum mismatch: "
                f"expected {expected}, observed {observed}."
            )
        extracted = stage / "extracted"
        extracted.mkdir()
        _safe_extract(archive_path, extracted)
        launcher = _find_launcher(extracted)
        if launcher is None:
            raise JDTLSBootstrapError("Downloaded Eclipse JDT LS has no bin/jdtls launcher.")
        install_source = launcher.parent.parent
        if root.exists():
            shutil.rmtree(root)
        os.replace(install_source, root)
        resolved = root / "bin" / "jdtls"
        resolved.chmod(resolved.stat().st_mode | 0o111)
        receipt = {
            "schema_version": "mmm/jdtls-install-v1",
            "version": JDTLS_VERSION,
            "build": JDTLS_BUILD,
            "archive": JDTLS_ARCHIVE,
            "sha256": observed,
        }
        (root / ".mmm-install.json").write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return resolved


def ensure_jdtls() -> Path:
    """Return a verified, cached Eclipse JDT LS launcher.

    Installation is process-safe on Linux/Colab. When MMM_JAVA_VERSION declares
    the project's Java major, the exact project JDK is reused or provisioned
    before JDT LS starts; the JDT LS launcher itself still requires Java 21+.
    """
    _ensure_project_jdk()
    _ensure_java_21()
    root = _cache_root()
    launcher = root / "bin" / "jdtls"
    if launcher.is_file() and os.access(launcher, os.X_OK):
        return launcher
    with _install_lock(root.parent / ".install.lock"):
        if launcher.is_file() and os.access(launcher, os.X_OK):
            return launcher
        return _install_jdtls(root)


def main() -> None:
    try:
        launcher = ensure_jdtls()
    except Exception as exc:
        raise SystemExit(f"JDT LS bootstrap failed: {type(exc).__name__}: {exc}") from exc
    os.execv(str(launcher), [str(launcher), *sys.argv[1:]])


if __name__ == "__main__":
    main()
