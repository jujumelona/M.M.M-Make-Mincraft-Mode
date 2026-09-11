"""Extract API symbols from Minecraft/Fabric JARs with full descriptors.

P0-7: Research phase extracts actual symbols from target artifacts, not manual thresholds.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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
    """Extract API symbols from Minecraft and Fabric JARs.
    
    P0-7: Replaces version thresholds with actual extracted symbols.
    """
    
    def __init__(self):
        self.extracted_symbols: dict[str, APISymbol] = {}
    
    def extract_from_minecraft_jar(
        self,
        minecraft_jar: Path,
        mappings_file: Path | None = None,
    ) -> dict[str, APISymbol]:
        """Extract all public API symbols from Minecraft JAR.
        
        P0-7: Reads actual class files and extracts signatures.
        
        Args:
            minecraft_jar: Path to Minecraft client/server JAR
            mappings_file: Optional mappings (yarn, mojmap, etc.)
            
        Returns:
            Dict of qualified_name -> APISymbol
        """
        symbols = {}
        
        try:
            with zipfile.ZipFile(minecraft_jar, 'r') as jar:
                # Find all .class files
                class_files = [f for f in jar.namelist() if f.endswith('.class')]
                
                for class_file in class_files:
                    # TODO P0-7: Parse .class file using:
                    # 1. Python struct for reading class file format
                    # 2. Or use javaobj-py3 / jawa libraries
                    # 3. Extract: methods, fields, constructors with descriptors
                    pass
        except Exception as exc:
            raise APISymbolExtractionError(
                f"Failed to extract from {minecraft_jar}: {exc}"
            ) from exc
        
        return symbols
    
    def extract_from_fabric_jar(
        self,
        fabric_jar: Path,
    ) -> dict[str, APISymbol]:
        """Extract Fabric API symbols.
        
        P0-7: Same as Minecraft extraction but for Fabric API.
        """
        # Same implementation as extract_from_minecraft_jar
        return self.extract_from_minecraft_jar(fabric_jar)
    
    def extract_from_maven(
        self,
        minecraft_version: str,
        fabric_version: str,
    ) -> dict[str, APISymbol]:
        """Download and extract symbols from Maven repositories.
        
        P0-7: Automated extraction for any version.
        
        Args:
            minecraft_version: e.g., "1.21.5"
            fabric_version: e.g., "0.110.5+1.21.5"
            
        Returns:
            Combined symbols from Minecraft + Fabric
        """
        # TODO P0-7: Implement Maven download + extraction
        # 1. Download from https://maven.fabricmc.net/
        # 2. Download from https://launcher.mojang.com/
        # 3. Extract symbols from both
        # 4. Merge with side detection
        
        raise NotImplementedError(
            f"P0-7: Maven extraction stub for MC {minecraft_version} + Fabric {fabric_version}"
        )
    
    def apply_mappings(
        self,
        symbols: dict[str, APISymbol],
        mappings: dict[str, str],
        target_namespace: str,
    ) -> dict[str, APISymbol]:
        """Apply mappings to convert between namespaces.
        
        Args:
            symbols: Symbols in source namespace
            mappings: Source name -> target name
            target_namespace: NAMED | INTERMEDIARY | OFFICIAL
            
        Returns:
            Symbols remapped to target namespace
        """
        remapped = {}
        for qual_name, symbol in symbols.items():
            # Apply mapping to owner and name
            # TODO P0-7: Implement proper mapping application
            remapped[qual_name] = symbol
        return remapped
    
    def detect_side(self, class_name: str, jar_type: str) -> str:
        """Detect if a class is CLIENT, SERVER, or COMMON.
        
        P0-7: Uses annotations, package names, or separate JARs.
        """
        # Heuristics for side detection
        if "client" in class_name.lower():
            return "CLIENT"
        elif "server" in class_name.lower():
            return "SERVER"
        elif jar_type == "client":
            return "CLIENT"
        elif jar_type == "server":
            return "SERVER"
        else:
            return "COMMON"


def extract_and_save_symbols(
    minecraft_version: str,
    output_dir: Path = Path("data/api_symbols"),
) -> Path:
    """Extract API symbols and save to JSON file.
    
    P0-7: Command-line tool for building symbol database.
    
    Usage:
        python -m minecraft_mod_ai.api_symbol_extractor 1.21.5
    
    Returns:
        Path to saved JSON file
    """
    extractor = MinecraftAPIExtractor()
    
    try:
        # Extract symbols (TODO: implement actual extraction)
        symbols = {}  # extractor.extract_from_maven(minecraft_version, "latest")
        
        # Prepare output
        output = {
            "minecraft_version": minecraft_version,
            "extraction_date": datetime.utcnow().isoformat() + "Z",
            "symbol_count": len(symbols),
            "api_symbols": {
                name: {
                    "owner": sym.owner,
                    "name": sym.name,
                    "descriptor": sym.descriptor,
                    "kind": sym.kind,
                    "static": sym.is_static,
                    "public": sym.is_public,
                    "side": sym.side,
                    "namespace": sym.namespace,
                    "source_jar": sym.source_jar,
                }
                for name, sym in symbols.items()
            }
        }
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"api_symbols_{minecraft_version}.json"
        output_file.write_text(
            json.dumps(output, indent=2, sort_keys=True),
            encoding="utf-8"
        )
        
        print(f"Extracted {len(symbols)} symbols to {output_file}")
        return output_file
        
    except Exception as exc:
        raise APISymbolExtractionError(
            f"Failed to extract symbols for {minecraft_version}: {exc}"
        ) from exc


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m minecraft_mod_ai.api_symbol_extractor <minecraft_version>")
        print("Example: python -m minecraft_mod_ai.api_symbol_extractor 1.21.5")
        sys.exit(1)
    
    minecraft_version = sys.argv[1]
    output_file = extract_and_save_symbols(minecraft_version)
    print(f"✓ Symbols saved to {output_file}")
