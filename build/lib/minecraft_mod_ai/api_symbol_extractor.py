"""Extract API symbols from Minecraft/Fabric JARs with full descriptors.

P0-7: Research phase extracts actual symbols from target artifacts, not manual thresholds.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APISymbol:
    """API symbol extracted from JAR with full metadata."""
    qualified_name: str  # com.example.ClassName.methodName
    owner: str  # com.example.ClassName (or com/example/ClassName in JVM format)
    name: str  # methodName
    descriptor: str  # JVM descriptor: "(Ljava/lang/String;)V"
    kind: str  # METHOD | FIELD | CONSTRUCTOR | CLASS
    is_static: bool
    is_public: bool
    side: str  # CLIENT | SERVER | COMMON
    namespace: str  # NAMED | INTERMEDIARY | OFFICIAL
    source_jar: str  # Which JAR this came from


class APISymbolExtractionError(Exception):
    """Error during API symbol extraction."""
    pass


class MinecraftAPIExtractor:
    """Extract all public/protected declarations, retaining overload identities."""
    def __init__(self, *, java_version=17, namespace="NAMED"):
        self.extracted_symbols = {}
        self.java_version = java_version
        self.namespace = namespace

    def extract_from_minecraft_jar(self, minecraft_jar, mappings_file=None):
        from .jar_api_extractor import inspect_jar
        from .tiny_mappings import TinyMappings
        try:
            classes = inspect_jar(Path(minecraft_jar), java_version=self.java_version)
            symbols = {}
            for name, item in classes.items():
                if not item.access & 5:
                    continue
                symbols[name] = APISymbol(name, name, name.rsplit("/", 1)[-1], "L"+name+";",
                    "CLASS", False, bool(item.access & 1), item.side, self.namespace, str(minecraft_jar))
                for member in item.members:
                    if not member.access & 5:
                        continue
                    key = f"{name}#{member.name}{member.descriptor}"
                    symbols[key] = APISymbol(key, name, member.name, member.descriptor, member.kind,
                        member.is_static, bool(member.access & 1), member.side, self.namespace, str(minecraft_jar))
            if mappings_file:
                mapping = TinyMappings.parse(Path(mappings_file).read_text(encoding="utf-8"))
                symbols = self.apply_mappings(symbols, mapping, mapping.namespaces[-1])
            self.extracted_symbols.update(symbols)
            return symbols
        except (ValueError, OSError) as exc:
            raise APISymbolExtractionError(str(exc)) from exc

    def extract_from_fabric_jar(self, fabric_jar):
        return self.extract_from_minecraft_jar(fabric_jar)

    def extract_from_maven(self, minecraft_version, fabric_version=None, *, artifacts=None):
        """Download exact URL/hash pairs in parallel, then extract in declared order."""
        del minecraft_version, fabric_version  # Coordinates alone cannot authorize artifacts.
        import tempfile
        from urllib.request import urlopen
        from .implementation_identity import compute_content_hash
        if not artifacts:
            raise APISymbolExtractionError("PINNED_ARTIFACT_URLS_AND_HASHES_REQUIRED")

        declared = tuple(artifacts)
        for artifact in declared:
            if not artifact["url"].startswith("https://"):
                raise APISymbolExtractionError("HTTPS_ARTIFACT_REQUIRED")

        def download(index_artifact):
            index, artifact = index_artifact
            with urlopen(artifact["url"], timeout=60) as response:
                data = response.read()
            if compute_content_hash(data) != artifact["sha256"]:
                raise APISymbolExtractionError("ARTIFACT_HASH_MISMATCH")
            return index, data

        indexed = tuple(enumerate(declared))
        if len(indexed) == 1:
            downloaded = [download(indexed[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=len(indexed),
                thread_name_prefix="api-artifact-download",
            ) as pool:
                futures = [pool.submit(download, item) for item in indexed]
                try:
                    downloaded = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise

        combined = {}
        with tempfile.TemporaryDirectory(prefix="mmm-api-") as temp:
            for index, data in downloaded:
                path = Path(temp) / f"{index}.jar"
                path.write_bytes(data)
                for key, symbol in self.extract_from_minecraft_jar(path).items():
                    if key in combined and combined[key] != symbol:
                        raise APISymbolExtractionError("DUPLICATE_API_SYMBOL")
                    combined[key] = symbol
        return combined

    def apply_mappings(self, symbols, mappings, target_namespace):
        from dataclasses import replace
        from .tiny_mappings import TinyMappings
        if not isinstance(mappings, TinyMappings):
            raise APISymbolExtractionError("TINY_V2_MAPPING_REQUIRED")
        result = {}
        for symbol in symbols.values():
            source = next((n for n in mappings.namespaces if n.casefold() == symbol.namespace.casefold()), None)
            if source is None:
                raise APISymbolExtractionError("SOURCE_NAMESPACE_NOT_IN_MAPPINGS")
            if symbol.kind == "CLASS":
                owner = mappings.class_name(symbol.owner, source, target_namespace)
                name, descriptor = owner.rsplit("/", 1)[-1], "L"+owner+";"
                key = owner
            else:
                owner, name, descriptor = mappings.member(symbol.owner, symbol.name, symbol.descriptor,
                    "f" if symbol.kind == "FIELD" else "m", source, target_namespace)
                key = f"{owner}#{name}{descriptor}"
            if key in result:
                raise APISymbolExtractionError("MAPPING_COLLISION")
            result[key] = replace(symbol, qualified_name=key, owner=owner, name=name,
                                  descriptor=descriptor, namespace=target_namespace)
        return result

    def detect_side(self, class_name, jar_type):
        if jar_type.upper() not in {"COMMON", "CLIENT", "SERVER"}:
            raise APISymbolExtractionError("INSPECTED_SIDE_REQUIRED")
        return jar_type.upper()


def extract_and_save_symbols(minecraft_version, output_dir=Path("data/api_symbols"), *, jars, java_version, namespace):
    from dataclasses import asdict
    extractor = MinecraftAPIExtractor(java_version=java_version, namespace=namespace)
    for jar in jars:
        extractor.extract_from_minecraft_jar(jar)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{minecraft_version}.json"
    path.write_text(json.dumps({k: asdict(v) for k, v in extractor.extracted_symbols.items()}, indent=2), encoding="utf-8")
    return path