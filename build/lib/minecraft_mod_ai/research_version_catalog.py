"""Explicit maintenance command: research official release metadata, never at selection time.

Run with --output DIRECTORY. Downloads are read as data, never executed. A complete
Minecraft client download is checksum-verified before extracting its version.json.
The resulting catalog certifies metadata only, not compilation or gameplay.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha1, sha256
import io
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

from .platform_live_discovery import _format_pack_version
from .target_contract import TargetContract, uses_native_names

MAVEN = "https://maven.fabricmc.net/"
REPOSITORY = "https://api.github.com/repos/FabricMC/fabric-example-mod"
MANIFEST = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MMM-version-catalog/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def properties(raw):
    return dict(re.findall(r"(?m)^([a-z_]+)\s*=\s*([^\r\n]+)", raw.decode()))


def version_tuple(value):
    return tuple(int(part) for part in value.split("."))


def enrich_exclusions(output):
    """Retain independently verifiable facts without admitting an incomplete bundle."""
    path = output / "official_version_evidence.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    def source(url):
        raw = fetch(url)
        report["sources"][url] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw

    manifest = json.loads(source(MANIFEST))
    releases = {row["id"]: row for row in manifest["versions"]}
    for row in report["skipped"]:
        version = row["minecraft"]
        release = releases[version]
        raw = source(release["url"])
        if sha1(raw).hexdigest() != release["sha1"]:
            raise ValueError("Excluded-version release checksum mismatch")
        detail = json.loads(raw)
        client = detail["downloads"]["client"]
        jar = source(client["url"])
        if sha1(jar).hexdigest() != client["sha1"] or len(jar) != client["size"]:
            raise ValueError("Excluded-version client checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(jar)) as archive:
            embedded = json.loads(archive.read("version.json"))
        if embedded["release_target"] != version:
            raise ValueError("Excluded-version release identity mismatch")
        yarn = json.loads(source(f"https://meta.fabricmc.net/v2/versions/yarn/{version}"))
        if any(item["gameVersion"] != version for item in yarn):
            raise ValueError("Yarn metadata Minecraft mismatch")
        row["verified_partial_facts"] = {
            "minecraft_java_minimum": detail["javaVersion"]["majorVersion"],
            "mojang_version_json": embedded,
            "mojang_mappings_available": "client_mappings" in detail["downloads"],
            "published_yarn_coordinates": [item["maven"] for item in yarn],
            "release_metadata_url": release["url"],
        }
        row["reason"] = "NO_COHERENT_OFFICIAL_EXAMPLE_BUNDLE; partial Mojang and Fabric facts recorded"
        print(f"PARTIAL {version}: verified Java, packs and {len(yarn)} Yarn coordinates", flush=True)
    report["sources"] = dict(sorted(report["sources"].items()))
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def research(output):
    sources = {}

    def source(url):
        raw = fetch(url)
        sources[url] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw

    manifest = json.loads(source(MANIFEST))
    releases = {row["id"]: row for row in manifest["versions"] if row["type"] == "release"}
    games = json.loads(source("https://meta.fabricmc.net/v2/versions/game"))
    stable = [row["version"] for row in games if row["stable"]]
    branches = json.loads(source(REPOSITORY + "/branches?per_page=100"))
    if len(branches) >= 100:
        raise ValueError("Official branch list needs pagination; refusing incomplete research")
    branches = {row["name"]: row["commit"]["sha"] for row in branches}
    loom_versions = ET.fromstring(source(MAVEN + "net/fabricmc/fabric-loom/maven-metadata.xml"))
    loom_releases = [node.text for node in loom_versions.findall("./versioning/versions/version")
                     if re.fullmatch(r"\d+\.\d+\.\d+", node.text or "")]
    skipped = []
    candidates = []
    for version in stable:
        if version not in branches or version not in releases:
            skipped.append({"minecraft": version, "reason": "NO_OFFICIAL_EXAMPLE_RELEASE_BRANCH"})
        else:
            candidates.append(version)

    def one(version):
        evidence = {"minecraft": version, "example_commit": branches[version],
                    "validation_level": "official_metadata", "compile": "not_run", "gametest": "not_run"}
        try:
            base = "https://raw.githubusercontent.com/FabricMC/fabric-example-mod/" + branches[version] + "/"
            props = properties(source(base + "gradle.properties"))
            if props["minecraft_version"] != version:
                raise ValueError("Official example Minecraft coordinate does not match branch")
            build = source(base + "build.gradle").decode()
            wrapper = source(base + "gradle/wrapper/gradle-wrapper.properties").decode()
            gradle = re.search(r"gradle-([\d.]+)-bin.zip", wrapper).group(1)
            gradle_sha = source(f"https://services.gradle.org/distributions/gradle-{gradle}-bin.zip.sha256").decode().strip()
            requested_loom = props["loom_version"]
            # Pin a release in the exact official example's Loom release line.
            # This is recorded as an explicit maintenance selection, never a runtime fallback.
            if requested_loom.endswith("-SNAPSHOT"):
                line = requested_loom.removesuffix("-SNAPSHOT") + "."
                matches = [value for value in loom_releases if value.startswith(line)]
                if not matches:
                    raise ValueError("No fixed Loom release in official example release line")
                loom = max(matches, key=version_tuple)
            else:
                loom = requested_loom
            module = json.loads(source(MAVEN + f"net/fabricmc/fabric-loom/{loom}/fabric-loom-{loom}.module"))
            runtime = next(row for row in module["variants"] if row["name"] == "runtimeElements")
            loom_sha = runtime["files"][0]["sha256"]
            if requested_loom.endswith("-SNAPSHOT"):
                snapshot_base = MAVEN + f"net/fabricmc/fabric-loom/{requested_loom}/"
                snapshot_meta = ET.fromstring(source(snapshot_base + "maven-metadata.xml"))
                timestamp = next(row.findtext("value") for row in snapshot_meta.findall("./versioning/snapshotVersions/snapshotVersion")
                                 if row.findtext("extension") == "module" and row.findtext("classifier") is None)
                snapshot_module = json.loads(source(snapshot_base + f"fabric-loom-{timestamp}.module"))
                snapshot_runtime = next(row for row in snapshot_module["variants"] if row["name"] == "runtimeElements")
                if snapshot_runtime["files"][0]["sha256"] != loom_sha or snapshot_runtime.get("dependencies") != runtime.get("dependencies"):
                    raise ValueError("Fixed Loom release is not binary/dependency identical to official recommendation")
            loom_binary = source(MAVEN + f"net/fabricmc/fabric-loom/{loom}/fabric-loom-{loom}.jar")
            if sha256(loom_binary).hexdigest() != loom_sha:
                raise ValueError("Fixed Loom binary checksum mismatch")
            attributes = runtime["attributes"]
            if version_tuple(gradle) < version_tuple(attributes["org.gradle.plugin.api-version"]):
                raise ValueError("Official wrapper does not satisfy fixed Loom Gradle requirement")
            plugin = "net.fabricmc.fabric-loom" if uses_native_names(version) else "net.fabricmc.fabric-loom-remap"
            if f"id '{plugin}'" not in build:
                raise ValueError("Unsupported official Loom plugin or naming regime")
            marker = source(MAVEN + plugin.replace(".", "/") + f"/{plugin}.gradle.plugin/{loom}/{plugin}.gradle.plugin-{loom}.pom")
            if f"<version>{loom}</version>" not in marker.decode():
                raise ValueError("Loom plugin marker does not bind the fixed release")
            loader = props["loader_version"]
            api = props["fabric_api_version"]
            for group, artifact, coordinate in (("net/fabricmc", "fabric-loader", loader),
                                                 ("net/fabricmc/fabric-api", "fabric-api", api)):
                source(MAVEN + f"{group}/{artifact}/{coordinate}/{artifact}-{coordinate}.pom")
            detail_raw = source(releases[version]["url"])
            if sha1(detail_raw).hexdigest() != releases[version]["sha1"]:
                raise ValueError("Mojang release metadata checksum mismatch")
            detail = json.loads(detail_raw)
            client = detail["downloads"]["client"]
            client_raw = source(client["url"])
            if len(client_raw) != client["size"] or sha1(client_raw).hexdigest() != client["sha1"]:
                raise ValueError("Mojang client checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(client_raw)) as archive:
                embedded = json.loads(archive.read("version.json"))
            if embedded["id"] != version:
                raise ValueError("Mojang embedded version identity mismatch")
            packs = embedded["pack_version"]
            data = _format_pack_version(packs, "data", version=version)
            resource = _format_pack_version(packs, "resource", version=version)
            java = detail["javaVersion"]["majorVersion"]
            build_java = max(java, attributes["org.gradle.jvm.version"])
            mapped = not uses_native_names(version)
            if mapped:
                if "mappings loom.officialMojangMappings()" not in build:
                    raise ValueError("Official example mapping declaration is not supported")
                mappings = detail["downloads"]["client_mappings"]
                mapping_raw = source(mappings["url"])
                if sha1(mapping_raw).hexdigest() != mappings["sha1"]:
                    raise ValueError("Mojang mappings checksum mismatch")
            evidence.update({"example_properties": props, "loom_selection": {
                "requested": requested_loom, "fixed_release": loom,
                "policy": "fixed_release_binary_and_dependencies_match_official_recommendation",
                "binary_sha256": loom_sha,
                "minimum_gradle": attributes["org.gradle.plugin.api-version"]},
                "minecraft_java_minimum": java, "build_java_minimum": build_java,
                "mojang_version_json": embedded,
            })
            base_facts = {
                "repositories": [MAVEN, "https://libraries.minecraft.net/", "https://repo.maven.apache.org/maven2/"],
                "dependency_coordinates": {"minecraft": f"com.mojang:minecraft:{version}",
                    "fabric_loader": f"net.fabricmc:fabric-loader:{loader}",
                    "fabric_api": f"net.fabricmc.fabric-api:fabric-api:{api}",
                    "fabric_loom": f"net.fabricmc:fabric-loom:{loom}", "loom_plugin": plugin,
                    "gradle_jvm_major": str(build_java)}}
            from .populate_version_artifact_rules import build_version_facts
            facts = build_version_facts(version, base_facts=base_facts)
            evidence.update({"artifact_rules_count": len(facts["artifact_rules"]),
                "capabilities_count": len(facts["capabilities"]),
                "unverified": ["runtime"]})
            revision = "sha256:" + sha256(encode(evidence).encode()).hexdigest()
            facts["host_revision"] = revision
            target = TargetContract(adapter_id="official-fabric-" + version, edition="java", loader="fabric",
                minecraft_version=version, java_version=str(java), yarn_mappings="mojang" if mapped else "",
                mappings_kind="mojang" if mapped else "", mappings_version="mojang" if mapped else "",
                fabric_loader=loader, fabric_api=api, fabric_loom=loom, gradle=gradle, gradle_sha256=gradle_sha,
                data_pack_version=data, resource_pack_version=resource, resource_pack_format=int(resource.split(".")[0]),
                release_metadata_url=releases[version]["url"], source_api_family="mojang",
                deterministic_module_kinds=frozenset(), host_facts_json=encode(facts))
            bundle = target.version_context.to_dict()
            evidence["context_id"] = bundle["context_id"]
            print(f"METADATA {version}: API {api}, Java {java}, packs {data}/{resource}", flush=True)
            return bundle, evidence
        except (OSError, ValueError, KeyError, StopIteration, AttributeError, zipfile.BadZipFile) as exc:
            print(f"SKIP {version}: {exc}", flush=True)
            return None, {**evidence, "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, candidates))
    bundles = [bundle for bundle, _ in results if bundle is not None]
    skipped.extend(evidence for bundle, evidence in results if bundle is None)
    if not bundles:
        raise ValueError("No complete version metadata bundles verified; existing catalog unchanged")
    report = {"schema_version": "mmm/official-version-research-v1", "retrieved_at": datetime.now(timezone.utc).isoformat(),
              "scope": "all stable Fabric game versions with official example release branches",
              "sources": dict(sorted(sources.items())), "versions": [e for b, e in results if b is not None],
              "skipped": skipped}
    catalog = {"schema_version": "mmm/host-version-catalog-v1", "auto_context_id": bundles[0]["context_id"], "bundles": bundles}
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("official_version_evidence.json", report), ("host_version_catalog.json", catalog)):
        (output / filename).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    enrich_exclusions(output)
    print(f"Wrote {len(bundles)} bundles; {len(skipped)} exclusions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enrich-exclusions", action="store_true")
    options = parser.parse_args()
    if options.enrich_exclusions:
        enrich_exclusions(options.output)
    else:
        research(options.output)
