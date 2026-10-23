"""Prepare the pinned real Minecraft/Fabric classpath for generated Java tests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

MINECRAFT = "26.2"
FABRIC_API = "0.155.2+26.2"
FABRIC_LOADER = "0.19.3"


def download(url: str, target: Path, expected_sha1: str) -> Path:
    if (
        target.is_file()
        and hashlib.sha1(target.read_bytes()).hexdigest() == expected_sha1
    ):
        return target
    with urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha1(data).hexdigest() != expected_sha1:
        raise ValueError("SDK checksum mismatch: " + url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def fabric_artifact(group: str, name: str, version: str, target: Path) -> Path:
    url = f"https://maven.fabricmc.net/{group}/{name}/{version}/{name}-{version}.jar"
    with urlopen(url + ".sha1", timeout=30) as response:
        digest = response.read(1024).decode().strip().split()[0]
    return download(url, target, digest)


def prepare(cache: Path) -> Path:
    cache = cache.resolve()
    with urlopen(
        "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json", timeout=30
    ) as response:
        manifest = json.load(response)
    release = next(row for row in manifest["versions"] if row["id"] == MINECRAFT)
    metadata = download(release["url"], cache / "version.json", release["sha1"])
    version = json.loads(metadata.read_text())
    client = version["downloads"]["client"]
    download(client["url"], cache / "client.jar", client["sha1"])
    libraries = []
    for row in version["libraries"]:
        artifact = row.get("downloads", {}).get("artifact")
        if artifact and "natives-" not in artifact["path"]:
            libraries.append(artifact)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(
                lambda row: download(
                    row["url"], cache / "libraries" / row["path"], row["sha1"]
                ),
                libraries,
            )
        )
    api = fabric_artifact(
        "net/fabricmc/fabric-api", "fabric-api", FABRIC_API, cache / "fabric-api.jar"
    )
    loader = fabric_artifact(
        "net/fabricmc", "fabric-loader", FABRIC_LOADER, cache / "fabric-loader.jar"
    )
    modules = []
    with ZipFile(api) as archive:
        for row in json.loads(archive.read("fabric.mod.json"))["jars"]:
            member = row["file"]
            target = cache / "fabric-modules" / Path(member).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            modules.append(target)
    classpath = cache / "classpath.txt"
    classpath.write_text(
        os.pathsep.join(map(str, [cache / "client.jar", loader, *paths, *modules]))
    )
    print(
        f"Verified Minecraft {MINECRAFT}, Fabric API {FABRIC_API}, Loader {FABRIC_LOADER}; {len(paths)} libraries, {len(modules)} API modules"
    )
    return classpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    print(prepare(parser.parse_args().cache))
