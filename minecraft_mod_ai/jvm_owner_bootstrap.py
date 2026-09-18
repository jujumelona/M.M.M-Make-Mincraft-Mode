"""Build the pinned owner runtime once per source identity and launch its framework."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .owner_rpc import OwnerRPCError

GRADLE_VERSION = '8.6'
GRADLE_SHA256 = '9631d53cf3e74bfa726893aee1f8994fee4e060c401335946dba2156f440f24c'


def owner_command(workspace: Path) -> list[str]:
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

    with _exclusive_cache_lock(cache, timeout_seconds=600):
        if not (distribution / 'configuration' / 'config.ini').is_file():
            target.mkdir(parents=True, exist_ok=True)
            for path in files:
                output = target / path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
            gradle = GradleRunner(cache / 'gradle').ensure_gradle(GRADLE_VERSION, GRADLE_SHA256)
            with tempfile.TemporaryFile(mode='w+b') as log:
                result = subprocess.run([str(gradle), '--no-daemon', '--console=plain', 'installDist'],
                                        cwd=target, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=600, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode:
                    log.seek(max(0, log.tell() - 12000))
                    raise OwnerRPCError('Owner build failed: ' + log.read().decode('utf-8', errors='replace'))
    frameworks = list((distribution / 'plugins').glob('org.eclipse.osgi-*.jar'))
    if len(frameworks) != 1:
        raise OwnerRPCError('Owner runtime must contain exactly one Equinox framework')
    java_home = os.environ.get('JAVA_HOME')
    java = str(Path(java_home) / 'bin' / ('java.exe' if os.name == 'nt' else 'java')) if java_home else shutil.which('java')
    if not java:
        raise OwnerRPCError('Java 17 or newer is required to launch JDT Core')
    configuration = workspace / 'configuration'
    configuration.mkdir(parents=True, exist_ok=True)
    shutil.copy2(distribution / 'configuration' / 'config.ini', configuration / 'config.ini')
    return [java, '-cp', str(frameworks[0]), 'org.eclipse.core.runtime.adaptor.EclipseStarter',
            '-configuration', str(configuration), '-data', str(workspace / 'data'),
            '-application', 'mmm.owner.application', '-nosplash']
