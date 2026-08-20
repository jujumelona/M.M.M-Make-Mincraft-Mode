from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


JDTLS_VERSION = "1.60.0"
JDTLS_BUILD = "202606262232"
JDTLS_ARCHIVE = f"jdt-language-server-{JDTLS_VERSION}-{JDTLS_BUILD}.tar.gz"
JDTLS_BASE_URL = f"https://download.eclipse.org/jdtls/milestones/{JDTLS_VERSION}"
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_JAVA_VERSION = re.compile(r'version\s+"(?P<major>\d+)')
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


def _ensure_java_21() -> None:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        configured = Path(java_home).expanduser() / "bin" / "java"
        if configured.is_file() and (_java_major(str(configured)) or 0) >= 21:
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
                raise JDTLSBootstrapError(f"JDT LS download exceeded {max_bytes} bytes.")
            output.write(chunk)
    if total == 0:
        raise JDTLSBootstrapError("JDT LS download returned an empty response.")


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


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        destination_root = destination.resolve()
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise JDTLSBootstrapError(
                    f"Unsafe entry in Eclipse JDT LS archive: {member.name!r}."
                )
            candidate = (destination / member.name).resolve()
            try:
                candidate.relative_to(destination_root)
            except ValueError as exc:
                raise JDTLSBootstrapError(
                    f"Archive entry escapes JDT LS install root: {member.name!r}."
                ) from exc
        archive.extractall(destination, members=members)


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

    Installation is process-safe on Linux/Colab and only downloads the pinned
    Eclipse milestone when the managed cache is missing.
    """
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
