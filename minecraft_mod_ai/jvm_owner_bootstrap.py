"""Build the pinned owner runtime once per source identity and launch its framework."""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .owner_rpc import OwnerRPCError

GRADLE_VERSION = '8.6'
GRADLE_SHA256 = '9631d53cf3e74bfa726893aee1f8994fee4e060c401335946dba2156f440f24c'


def owner_command(
    workspace: Path,
    *,
    timeout_seconds: int | float = 600,
    java_home: Path | str | None = None,
    required_major: int | None = None,
) -> list[str]:
    if isinstance(timeout_seconds, bool):
        raise OwnerRPCError("JVM owner bootstrap timeout must be a positive number")
    try:
        timeout_value = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise OwnerRPCError(
            "JVM owner bootstrap timeout must be a positive number"
        ) from exc
    if not math.isfinite(timeout_value) or timeout_value <= 0.0:
        raise OwnerRPCError("JVM owner bootstrap timeout must be a positive finite number")
    deadline = time.monotonic() + timeout_value

    def remaining_timeout() -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise OwnerRPCError("JVM owner bootstrap deadline exceeded")
        return max(1, int(math.ceil(remaining)))

    source = Path(__file__).with_name('jvm_owner')
    files = sorted(path for path in source.rglob('*') if path.is_file()
                   and not set(path.relative_to(source).parts) & {'build', '.gradle'})
    identity = hashlib.sha256()
    for path in files:
        identity.update(path.relative_to(source).as_posix().encode())
        identity.update(path.read_bytes())
    cache = Path.home() / '.cache' / 'mmm' / 'jvm-owner'
    target = cache / identity.hexdigest()
    distribution = target / 'build' / 'install' / 'owner'
    from .runner import GradleRunner, _exclusive_cache_lock

    with _exclusive_cache_lock(cache, timeout_seconds=remaining_timeout()):
        if not (distribution / 'configuration' / 'config.ini').is_file():
            target.mkdir(parents=True, exist_ok=True)
            for path in files:
                output = target / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
            gradle_budget = remaining_timeout()
            gradle = GradleRunner(
                cache / 'gradle',
                download_timeout_seconds=min(300, gradle_budget),
            ).ensure_gradle(
                GRADLE_VERSION,
                GRADLE_SHA256,
                lock_timeout_seconds=gradle_budget,
            )
            with tempfile.TemporaryFile(mode='w+b') as log:
                result = subprocess.run([str(gradle), '--no-daemon', '--console=plain', 'installDist'],
                                        cwd=target, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=remaining_timeout(), check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    log.seek(max(0, log.tell() - 12000))
                    raise OwnerRPCError('Owner build failed: ' + log.read().decode('utf-8', errors='replace'))
    frameworks = list((distribution / 'plugins').glob('org.eclipse.osgi-*.jar'))
    if len(frameworks) != 1:
        raise OwnerRPCError('Owner runtime must contain exactly one Equinox framework')
    # The owner hosts both JDT Core and Gradle Tooling API model resolution.
    # Two constraints govern the owner JVM major version:
    #   1. >= 21  – Modern Fabric Loom plugins require Java 21+ even when the
    #      project itself targets an older release (e.g. Minecraft 1.20.1 / Java 17).
    #   2. >= project release target  – JDT Core's COMPILER_RELEASE option resolves
    #      the --release system image from the *running* JVM.  If the owner JVM is
    #      older than the project's --release target (e.g. owner=21, release=25),
    #      JDT fails with "release N is not found in the system".
    # Therefore the owner JVM must be max(21, project_release_major).
    from .java_lsp import (
        _java_major_version,
        _requested_project_java_major,
        _resolve_project_java_home,
    )

    _OWNER_MINIMUM_JAVA = 21
    if java_home is not None:
        owner_java_home = Path(java_home).resolve()
        owner_major = (
            int(required_major)
            if required_major is not None
            else (_java_major_version(owner_java_home) or _OWNER_MINIMUM_JAVA)
        )
    else:
        if required_major is not None:
            project_major = int(required_major)
        else:
            try:
                project_major = _requested_project_java_major()
            except Exception:
                project_major = _OWNER_MINIMUM_JAVA
        owner_major = max(_OWNER_MINIMUM_JAVA, project_major)
        owner_java_home = _resolve_project_java_home(owner_major)

    java = str(
        owner_java_home / 'bin' / ('java.exe' if os.name == 'nt' else 'java')
    )
    if not Path(java).is_file():
        raise OwnerRPCError(
            f'Java {owner_major} or newer is required to launch the JVM owner'
        )
    configuration = workspace / 'configuration'
    configuration.mkdir(parents=True, exist_ok=True)
    shutil.copy2(distribution / 'configuration' / 'config.ini', configuration / 'config.ini')
    return [java, '-cp', str(frameworks[0]), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
            '-configuration', str(configuration), '-data', str(workspace / 'data'),
            '-application', 'mmm.owner.application', '-nosplash']
