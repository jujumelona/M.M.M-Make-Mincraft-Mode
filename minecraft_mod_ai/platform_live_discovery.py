from __future__ import annotations

import hashlib
import html
import io
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .target_profile_semantics import mappings_applicable


class PlatformDiscoveryError(RuntimeError):
    pass


_LOGGER = logging.getLogger(__name__)
_DEFAULT_DISCOVERY_RETRIES = 4
_MAX_DISCOVERY_RETRIES = 6
_RETRY_DELAYS = (0.25, 0.75, 1.5, 2.5, 4.0)


def _emit_discovery_log(message: str, *, exc_info: bool = False) -> None:
    """Make discovery failures visible in both normal Python and Colab output."""

    _LOGGER.warning("%s", message, exc_info=exc_info)
    print(f"platform discovery: {message}", flush=True)


def _discovery_retries() -> int:
    raw = os.environ.get(
        "MMM_PLATFORM_DISCOVERY_RETRIES",
        str(_DEFAULT_DISCOVERY_RETRIES),
    ).strip()
    try:
        value = int(raw)
    except ValueError:
        value = _DEFAULT_DISCOVERY_RETRIES
    return max(1, min(_MAX_DISCOVERY_RETRIES, value))


def _retry_request_url(url: str, attempt: int, exc: BaseException) -> str:
    """Bust stale edge-cache responses while preserving the recorded source URL."""

    status = getattr(exc, "code", None)
    if attempt <= 1 or status not in {404, 408, 429, 500, 502, 503, 504}:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}mmm_platform_retry={attempt}"


def _error_summary(exc: BaseException) -> str:
    status = getattr(exc, "code", None)
    status_text = f" status={status}" if status is not None else ""
    return f"{type(exc).__name__}{status_text}: {exc}"


@dataclass(frozen=True)
class LiveFabricTarget:
    minecraft_version: str
    stable: bool
    loader_version: str
    fabric_api_version: str
    loom_version: str
    java_version: str
    gradle_version: str
    gradle_sha256: str
    mappings_kind: str
    mappings_version: str
    data_pack_version: str
    resource_pack_version: str
    release_metadata_url: str
    discovery_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "minecraft_version": self.minecraft_version,
            "stable": self.stable,
            "loader_version": self.loader_version,
            "fabric_api_version": self.fabric_api_version,
            "loom_version": self.loom_version,
            "java_version": self.java_version,
            "gradle_version": self.gradle_version,
            "gradle_sha256": self.gradle_sha256,
            "mappings_kind": self.mappings_kind,
            "mappings_version": self.mappings_version,
            "data_pack_version": self.data_pack_version,
            "resource_pack_version": self.resource_pack_version,
            "release_metadata_url": self.release_metadata_url,
            "discovery_sha256": self.discovery_sha256,
        }


_META = "https://meta.fabricmc.net"
_MAVEN = "https://maven.fabricmc.net"
_FABRIC_DEVELOP = "https://fabricmc.net/develop/"
_FABRIC_TEMPLATE_PROPERTIES = (
    "https://raw.githubusercontent.com/FabricMC/fabricmc.net/main/"
    "scripts/src/lib/template/templates/gradle/gradle.properties.eta"
)
_FABRIC_WRAPPER = (
    "https://raw.githubusercontent.com/FabricMC/fabricmc.net/main/"
    "scripts/src/lib/template/templates/gradle/wrapper/gradle/wrapper/"
    "gradle-wrapper.properties"
)
_MOJANG_MANIFEST = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"
_MINECRAFT_ARTICLE = "https://www.minecraft.net/en-us/article/minecraft-java-edition-{}"
_MINECRAFT_FEEDBACK_SEARCH = (
    "https://feedback.minecraft.net/api/v2/help_center/articles/search.json"
)
_API_METADATA_PATH = "/net/fabricmc/fabric-api/fabric-api/maven-metadata.xml"
_PACK_VERSION = re.compile(
    r"\b(The\s+)?(?P<kind>Data|Resource)\s+Pack\s+version\s+is\s+now\s+"
    r"(?P<version>[0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)


def _fetch(
    url: str,
    *,
    timeout: int = 20,
    retries: int | None = None,
) -> bytes:
    retries = _discovery_retries() if retries is None else max(1, int(retries))
    last_error: BaseException | None = None
    for attempt in range(1, retries + 1):
        request_url = url
        if last_error is not None:
            request_url = _retry_request_url(url, attempt, last_error)
        request = urllib.request.Request(
            request_url,
            headers={
                "User-Agent": (
                    "MMM-platform-discovery/2 "
                    "(+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode)"
                ),
                "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if attempt > 1:
                _emit_discovery_log(
                    f"recovered GET {url} on attempt {attempt}/{retries}"
                )
            return payload
        except Exception as exc:  # noqa: BLE001 - bounded retry must see transport failures
            last_error = exc
            _emit_discovery_log(
                f"GET {url} failed attempt {attempt}/{retries}; "
                f"request_url={request_url}; {_error_summary(exc)}",
                exc_info=attempt == retries,
            )
            if attempt >= retries:
                break
            delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
            _emit_discovery_log(
                f"retrying GET {url} in {delay:.2f}s "
                f"(next attempt {attempt + 1}/{retries})"
            )
            time.sleep(delay)

    assert last_error is not None
    raise PlatformDiscoveryError(
        f"official platform discovery failed after {retries} attempt(s): "
        f"{url}: {_error_summary(last_error)}"
    ) from last_error


def _json(url: str) -> Any:
    try:
        return json.loads(_fetch(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _emit_discovery_log(f"invalid official JSON response: {url}: {exc}", exc_info=True)
        raise PlatformDiscoveryError(f"official JSON response was invalid: {url}") from exc


@lru_cache(maxsize=8)
def _maven_versions(path: str) -> tuple[str, ...]:
    raw = _fetch(_MAVEN + path)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise PlatformDiscoveryError(f"invalid Fabric Maven metadata: {path}") from exc
    return tuple(
        str(node.text).strip()
        for node in root.findall("./versioning/versions/version")
        if node.text and str(node.text).strip()
    )


@lru_cache(maxsize=1)
def _discover_game_versions() -> tuple[dict[str, Any], ...]:
    payload = _json(_META + "/v2/versions/game")
    if not isinstance(payload, list):
        raise PlatformDiscoveryError("Fabric Meta game-version response was not a list")
    result: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        version = str(row.get("version", "")).strip()
        if version:
            result.append({"version": version, "stable": bool(row.get("stable"))})
    if not result:
        raise PlatformDiscoveryError("Fabric Meta returned no Minecraft versions")
    return tuple(result)


_GAME_VERSION_LOCK = threading.RLock()
_GAME_VERSION_FUTURE: Future[tuple[dict[str, Any], ...]] | None = None
_PLATFORM_WARMUP_STARTED = False


def _start_game_version_prefetch() -> Future[tuple[dict[str, Any], ...]]:
    global _GAME_VERSION_FUTURE
    with _GAME_VERSION_LOCK:
        future = _GAME_VERSION_FUTURE
        if future is not None and not future.cancelled():
            return future
        future = Future()
        _GAME_VERSION_FUTURE = future

        def worker() -> None:
            try:
                future.set_result(_discover_game_versions())
            except BaseException as exc:  # noqa: BLE001 - propagate worker failure to caller
                future.set_exception(exc)

        threading.Thread(
            target=worker,
            daemon=True,
            name="mmm_platform_prefetch",
        ).start()
        return future


def discover_game_versions() -> tuple[dict[str, Any], ...]:
    global _GAME_VERSION_FUTURE
    future = _start_game_version_prefetch()
    try:
        return future.result()
    except BaseException:
        with _GAME_VERSION_LOCK:
            if _GAME_VERSION_FUTURE is future:
                _GAME_VERSION_FUTURE = None
        raise


def start_platform_prefetch() -> None:
    """Warm official platform metadata without replacing discovery functions."""
    global _PLATFORM_WARMUP_STARTED
    _start_game_version_prefetch()
    with _GAME_VERSION_LOCK:
        if _PLATFORM_WARMUP_STARTED:
            return
        _PLATFORM_WARMUP_STARTED = True

    def warm_catalog() -> None:
        try:
            _common_platform_metadata()
            _stable_java_versions()
        except BaseException as exc:  # noqa: BLE001 - background prefetch must not escape
            _emit_discovery_log(
                f"background platform metadata prefetch failed: {_error_summary(exc)}",
                exc_info=True,
            )
            return

    threading.Thread(
        target=warm_catalog,
        daemon=True,
        name="mmm_platform_catalog_prefetch",
    ).start()


def latest_stable_versions(limit: int = 6) -> tuple[str, ...]:
    stable = [row["version"] for row in discover_game_versions() if row["stable"]]
    return tuple(stable[: max(1, int(limit))])


@lru_cache(maxsize=1)
def _stable_loader() -> str:
    payload = _json(_META + "/v2/versions/loader")
    if not isinstance(payload, list):
        raise PlatformDiscoveryError("Fabric loader response was not a list")
    for row in payload:
        if isinstance(row, dict) and row.get("stable") and row.get("version"):
            return str(row["version"])
    raise PlatformDiscoveryError("Fabric Meta returned no stable loader")


def _api_from_versions(version: str, versions: tuple[str, ...]) -> str:
    exact_suffixes = ("+" + version, "-" + version)
    matches = [value for value in versions if value.endswith(exact_suffixes)]
    if matches:
        return matches[-1]
    release = version.split("-", 1)[0]
    major = release.split(".", 1)[0]
    if major.isdigit() and int(major) >= 26:
        matches = [
            value
            for value in versions
            if value.endswith(("+" + release, "-" + release))
        ]
        if matches:
            return matches[-1]
    raise PlatformDiscoveryError(
        f"Fabric API has no artifact discoverable for Minecraft {version}"
    )


@lru_cache(maxsize=64)
def _api_for(version: str) -> str:
    return _api_from_versions(version, _maven_versions(_API_METADATA_PATH))


@lru_cache(maxsize=1)
def _loom_version() -> str:
    text = _fetch(_FABRIC_TEMPLATE_PROPERTIES).decode("utf-8", errors="replace")
    match = re.search(r"(?m)^loom_version=([^\s]+)\s*$", text)
    if not match:
        raise PlatformDiscoveryError(
            "Fabric official template source exposed no Loom version"
        )
    return match.group(1).strip()


@lru_cache(maxsize=1)
def _gradle_version() -> str:
    text = _fetch(_FABRIC_WRAPPER).decode("utf-8", errors="replace")
    match = re.search(r"gradle-([0-9][0-9A-Za-z_.-]*)-bin\.zip", text)
    if not match:
        raise PlatformDiscoveryError("Fabric template exposed no Gradle wrapper version")
    return match.group(1)


@lru_cache(maxsize=8)
def _gradle_sha256(version: str) -> str:
    raw = _fetch(
        f"https://services.gradle.org/distributions/gradle-{version}-bin.zip.sha256"
    ).decode("ascii", errors="ignore").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", raw):
        raise PlatformDiscoveryError("Gradle distribution checksum was invalid")
    return raw


@lru_cache(maxsize=1)
def _mojang_version_index() -> tuple[tuple[str, str], ...]:
    manifest = _json(_MOJANG_MANIFEST)
    rows = manifest.get("versions", []) if isinstance(manifest, dict) else []
    return tuple(
        (str(row.get("id", "")), str(row.get("url", "")))
        for row in rows
        if isinstance(row, dict) and row.get("id") and row.get("url")
    )


def _mojang_target_url(version: str) -> str:
    target_url = next(
        (url for version_id, url in _mojang_version_index() if version_id == version),
        "",
    )
    if not target_url:
        raise PlatformDiscoveryError(
            f"Mojang version manifest does not contain Minecraft {version}"
        )
    return target_url


def _java_from_detail(version: str, target_url: str) -> str:
    detail = _json(target_url)
    java = detail.get("javaVersion", {}) if isinstance(detail, dict) else {}
    major = java.get("majorVersion") if isinstance(java, dict) else None
    if not isinstance(major, int) or major <= 0:
        raise PlatformDiscoveryError(
            f"Mojang metadata exposed no Java major version for Minecraft {version}"
        )
    return str(major)


@lru_cache(maxsize=64)
def _mojang_java_version(version: str) -> str:
    return _java_from_detail(version, _mojang_target_url(version))


@lru_cache(maxsize=1)
def _stable_java_versions() -> tuple[tuple[str, str], ...]:
    """Resolve candidate Java requirements concurrently, preserving version order."""
    versions = latest_stable_versions(limit=8)
    if not versions:
        return ()
    urls = dict(_mojang_version_index())

    def resolve(version: str) -> tuple[str, str]:
        target_url = urls.get(version, "")
        if not target_url:
            return version, ""
        try:
            return version, _java_from_detail(version, target_url)
        except PlatformDiscoveryError as exc:
            _emit_discovery_log(
                f"Mojang Java metadata unavailable for Minecraft {version}: {exc}"
            )
            return version, ""

    workers = min(8, len(versions))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmm-mojang-java") as pool:
        return tuple(pool.map(resolve, versions))


def _gradle_bundle() -> tuple[str, str]:
    version = _gradle_version()
    return version, _gradle_sha256(version)


@lru_cache(maxsize=1)
def _common_platform_metadata() -> tuple[
    str,
    tuple[str, ...],
    str,
    str,
    str,
    tuple[tuple[str, str], ...],
]:
    """Fetch version-independent official metadata once, with independent I/O overlapped."""
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="mmm-platform-meta") as pool:
        loader_future = pool.submit(_stable_loader)
        api_future = pool.submit(_maven_versions, _API_METADATA_PATH)
        loom_future = pool.submit(_loom_version)
        gradle_future = pool.submit(_gradle_bundle)
        mojang_future = pool.submit(_mojang_version_index)
        gradle, gradle_sha256 = gradle_future.result()
        return (
            loader_future.result(),
            api_future.result(),
            loom_future.result(),
            gradle,
            gradle_sha256,
            mojang_future.result(),
        )


def _release_article_url(version: str) -> str:
    value = str(version).strip()
    if not value or not re.fullmatch(r"[0-9A-Za-z_.-]+", value):
        raise PlatformDiscoveryError(f"invalid Minecraft version for release metadata: {version!r}")
    # Minecraft's release article slugs use hyphens between numeric version
    # components (for example, 26.2 -> minecraft-java-edition-26-2).
    slug = value.casefold().replace(".", "-")
    return _MINECRAFT_ARTICLE.format(slug)


def _article_timeout() -> int:
    raw = os.environ.get("MMM_PLATFORM_ARTICLE_TIMEOUT", "6").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 6
    return max(2, min(value, 20))


def _article_retries() -> int:
    raw = os.environ.get("MMM_PLATFORM_ARTICLE_RETRIES", "2").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 2
    return max(1, min(value, 3))


def _parse_pack_versions(text: str, *, version: str, source_url: str) -> tuple[str, str, str]:
    plain = html.unescape(re.sub(r"<[^>]+>", " ", text))
    plain = " ".join(plain.split())
    found: dict[str, str] = {}
    for match in _PACK_VERSION.finditer(plain):
        found[match.group("kind").casefold()] = match.group("version")
    data_pack = found.get("data", "")
    resource_pack = found.get("resource", "")
    if not data_pack or not resource_pack:
        raise PlatformDiscoveryError(
            f"official Minecraft release metadata did not expose complete pack versions for {version}"
        )
    return data_pack, resource_pack, source_url


def _normalized_release_title(value: str) -> str:
    plain = html.unescape(str(value or "")).casefold()
    plain = re.sub(r"[:\-\u2013\u2014]+", " ", plain)
    return " ".join(plain.split())


def _feedback_release_article(version: str) -> dict[str, Any]:
    """Resolve one exact stable release article from Minecraft's structured API."""

    query = urllib.parse.urlencode(
        {
            "query": f"Minecraft Java Edition {version}",
            "per_page": "25",
        }
    )
    url = f"{_MINECRAFT_FEEDBACK_SEARCH}?{query}"
    raw = _fetch(
        url,
        timeout=_article_timeout(),
        retries=_article_retries(),
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlatformDiscoveryError(
            "official Minecraft feedback search returned invalid JSON"
        ) from exc
    results = payload.get("results", ()) if isinstance(payload, dict) else ()
    expected = _normalized_release_title(f"Minecraft Java Edition {version}")
    accepted_titles = {expected, f"{expected} hotfix"}
    article = next(
        (
            item
            for item in results
            if isinstance(item, dict)
            and _normalized_release_title(str(item.get("title") or "")) in accepted_titles
            and not bool(item.get("draft"))
        ),
        None,
    )
    if article is None:
        raise PlatformDiscoveryError(
            f"official Minecraft feedback search found no exact stable release article for {version}"
        )
    source_url = str(article.get("html_url") or "").strip()
    if not source_url.startswith("https://feedback.minecraft.net/"):
        raise PlatformDiscoveryError(
            "official Minecraft feedback result exposed an invalid article URL"
        )
    return article


def _feedback_pack_versions(version: str) -> tuple[str, str, str]:
    article = _feedback_release_article(version)
    source_url = str(article.get("html_url") or "").strip()
    return _parse_pack_versions(
        str(article.get("body") or ""),
        version=version,
        source_url=source_url,
    )


def _format_pack_version(pack: Any, kind: str, *, version: str) -> str:
    if isinstance(pack, int) and not isinstance(pack, bool):
        return str(pack)
    if not isinstance(pack, dict):
        raise PlatformDiscoveryError(
            f"Mojang version.json exposed an invalid pack_version for Minecraft {version}"
        )

    direct = pack.get(kind)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return str(direct)
    if isinstance(direct, float):
        return format(direct, "g")

    major = pack.get(f"{kind}_major")
    minor = pack.get(f"{kind}_minor")
    if not isinstance(major, int) or isinstance(major, bool) or major < 0:
        raise PlatformDiscoveryError(
            f"Mojang version.json exposed no {kind} pack major for Minecraft {version}"
        )
    if minor is None:
        return str(major)
    if not isinstance(minor, int) or isinstance(minor, bool) or minor < 0:
        raise PlatformDiscoveryError(
            f"Mojang version.json exposed an invalid {kind} pack minor for Minecraft {version}"
        )
    return f"{major}.{minor}"


@lru_cache(maxsize=64)
def _mojang_pack_versions(version: str) -> tuple[str, str]:
    """Read exact pack versions from Mojang's checksummed client version.json."""

    detail = _json(_mojang_target_url(version))
    downloads = detail.get("downloads", {}) if isinstance(detail, dict) else {}
    client = downloads.get("client", {}) if isinstance(downloads, dict) else {}
    jar_url = str(client.get("url") or "") if isinstance(client, dict) else ""
    expected_sha1 = str(client.get("sha1") or "").strip().casefold() if isinstance(client, dict) else ""
    if not jar_url.startswith(("https://piston-data.mojang.com/", "https://launcher.mojang.com/")):
        raise PlatformDiscoveryError(
            f"Mojang metadata exposed no official client JAR URL for Minecraft {version}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha1):
        raise PlatformDiscoveryError(
            f"Mojang metadata exposed no valid client JAR SHA-1 for Minecraft {version}"
        )

    jar = _fetch(jar_url, timeout=90, retries=2)
    actual_sha1 = hashlib.sha1(jar, usedforsecurity=False).hexdigest()
    if actual_sha1 != expected_sha1:
        raise PlatformDiscoveryError(
            f"Mojang client JAR checksum mismatch for Minecraft {version}"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(jar)) as archive:
            raw_version_json = archive.read("version.json")
        version_json = json.loads(raw_version_json.decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise PlatformDiscoveryError(
            f"Mojang client JAR exposed no valid version.json for Minecraft {version}"
        ) from exc

    pack = version_json.get("pack_version") if isinstance(version_json, dict) else None
    data_pack = _format_pack_version(pack, "data", version=version)
    resource_pack = _format_pack_version(pack, "resource", version=version)
    return data_pack, resource_pack


@lru_cache(maxsize=64)
def _official_pack_versions(version: str) -> tuple[str, str, str]:
    """Resolve exact pack versions from official machine-readable metadata first.

    Mojang's version manifest plus checksummed client ``version.json`` is the primary
    evidence path. Human release articles are bounded fallbacks only, so an unavailable
    Feedback or minecraft.net presentation endpoint cannot invalidate valid Mojang data.
    """

    errors: list[str] = []
    try:
        data_pack, resource_pack = _mojang_pack_versions(version)
        return data_pack, resource_pack, _mojang_target_url(version)
    except PlatformDiscoveryError as exc:
        errors.append(f"mojang version metadata: {exc}")
        _emit_discovery_log(
            f"Mojang pack metadata unavailable for {version}: {exc}; "
            "trying official release article metadata"
        )

    try:
        return _feedback_pack_versions(version)
    except PlatformDiscoveryError as exc:
        errors.append(f"feedback.minecraft.net: {exc}")
        _emit_discovery_log(
            f"structured official pack metadata unavailable for {version}: {exc}; "
            "trying the canonical release article"
        )

    url = _release_article_url(version)
    try:
        raw = _fetch(
            url,
            timeout=_article_timeout(),
            retries=1,
        ).decode("utf-8", errors="replace")
        return _parse_pack_versions(raw, version=version, source_url=url)
    except PlatformDiscoveryError as exc:
        errors.append(f"minecraft.net: {exc}")
        raise PlatformDiscoveryError(
            f"official pack metadata unavailable for {version}; " + "; ".join(errors)
        ) from exc


@lru_cache(maxsize=32)
def discover_fabric_target(version: str) -> LiveFabricTarget:
    version = str(version).strip()
    rows = discover_game_versions()
    row = next((item for item in rows if item["version"] == version), None)
    if row is None:
        raise PlatformDiscoveryError(
            f"Minecraft {version} is not advertised by the official Fabric Meta API"
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mmm-platform-target") as pool:
        common_future = pool.submit(_common_platform_metadata)
        pack_future = pool.submit(_official_pack_versions, version)
        (
            loader,
            api_versions,
            loom,
            gradle,
            gradle_sha256,
            _mojang_index,
        ) = common_future.result()
        data_pack_version, resource_pack_version, release_metadata_url = pack_future.result()

    api = _api_from_versions(version, api_versions)
    prefetched_java = dict(_stable_java_versions()).get(version, "")
    java = prefetched_java or _mojang_java_version(version)

    if mappings_applicable(version):
        mappings_kind = "mojang"
        mappings_version = "mojang"
    else:
        mappings_kind = ""
        mappings_version = ""
    payload = {
        "source": "official-live-discovery-v5",
        "minecraft_version": version,
        "stable": bool(row["stable"]),
        "loader_version": loader,
        "fabric_api_version": api,
        "loom_version": loom,
        "java_version": java,
        "gradle_version": gradle,
        "gradle_sha256": gradle_sha256,
        "mappings_kind": mappings_kind,
        "mappings_version": mappings_version,
        "data_pack_version": data_pack_version,
        "resource_pack_version": resource_pack_version,
        "release_metadata_url": release_metadata_url,
        "sources": [
            _META,
            _MAVEN,
            _FABRIC_DEVELOP,
            _FABRIC_TEMPLATE_PROPERTIES,
            _FABRIC_WRAPPER,
            _MOJANG_MANIFEST,
            release_metadata_url,
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LiveFabricTarget(
        minecraft_version=version,
        stable=bool(row["stable"]),
        loader_version=loader,
        fabric_api_version=api,
        loom_version=loom,
        java_version=java,
        gradle_version=gradle,
        gradle_sha256=gradle_sha256,
        mappings_kind=mappings_kind,
        mappings_version=mappings_version,
        data_pack_version=data_pack_version,
        resource_pack_version=resource_pack_version,
        release_metadata_url=release_metadata_url,
        discovery_sha256="sha256:" + digest,
    )