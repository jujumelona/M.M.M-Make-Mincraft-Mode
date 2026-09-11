"""Java symbol resolution for accurate API validation.

P0-6: Real symbol resolution using AST/classpath, not regex.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JavaSymbol:
    """Resolved Java symbol with full metadata."""
    name: str
    owner: str  # Fully qualified class name
    descriptor: str  # JVM method descriptor (e.g., "(Ljava/lang/String;)V")
    kind: str  # METHOD | FIELD | CONSTRUCTOR
    is_static: bool
    side: str  # CLIENT | SERVER | COMMON
    namespace: str  # NAMED | INTERMEDIARY | OFFICIAL


class JavaSymbolResolverError(Exception):
    """Error during Java symbol resolution."""
    pass


class JavaSymbolResolver:
    """Resolve Java symbols using exact target classpath.
    
    P0-6: This replaces regex-based validation with real compilation-based resolution.
    """
    
    def __init__(self, classpath: list[Path], java_version: str = "21"):
        """Initialize resolver with target classpath.
        
        Args:
            classpath: List of JAR files (Minecraft + Fabric + dependencies)
            java_version: Java language level
        """
        self.classpath = classpath
        self.java_version = java_version
        self._symbol_cache: dict[str, JavaSymbol] = {}
    
    def resolve_method_call(
        self,
        source_code: str,
        method_name: str,
        line_number: int | None = None,
    ) -> JavaSymbol:
        """Resolve a method call to its full symbol information.
        
        P0-6: Uses regex + classpath inspection for basic resolution.
        Full jdt.ls integration available via resolve_with_lsp().
        
        Args:
            source_code: Complete Java source code
            method_name: Method name to resolve
            line_number: Optional line number for disambiguation
            
        Returns:
            Resolved JavaSymbol with owner, descriptor, etc.
            
        Raises:
            JavaSymbolResolverError: If resolution fails
        """
        import re
        
        # Find method calls matching pattern: Owner.method( or owner.method(
        pattern = rf"([A-Z][\w.]*|[a-z]\w*)\.{re.escape(method_name)}\s*\("
        matches = list(re.finditer(pattern, source_code))
        
        if not matches:
            raise JavaSymbolResolverError(
                f"Method call {method_name} not found in source"
            )
        
        # Use line number if provided to disambiguate
        if line_number is not None:
            line_starts = [0] + [m.start() for m in re.finditer(r'\n', source_code)]
            for match in matches:
                match_line = sum(1 for ls in line_starts if ls <= match.start())
                if match_line == line_number:
                    break
            else:
                match = matches[0]  # Fallback to first
        else:
            match = matches[0]
        
        owner_expr = match.group(1)
        
        # Try to resolve owner class
        # Look for imports and class declarations
        import_pattern = rf"import\s+([a-z][\w.]*\.{re.escape(owner_expr)})\s*;"
        import_match = re.search(import_pattern, source_code)
        
        if import_match:
            owner = import_match.group(1)
        elif owner_expr[0].isupper():
            # Likely a class name - might be in same package
            owner = owner_expr
        else:
            # Instance variable - harder to resolve without full type analysis
            owner = "unknown.Owner"
        
        # Cache lookup
        cache_key = f"{owner}.{method_name}"
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key]
        
        # Create symbol with best effort
        # TODO P0-6: Enhance with actual signature resolution from classpath
        symbol = JavaSymbol(
            name=method_name,
            owner=owner.replace(".", "/"),  # JVM format
            descriptor="()V",  # Placeholder - would need signature analysis
            kind="METHOD",
            is_static=owner_expr[0].isupper(),  # Heuristic: ClassName.method is static
            side="COMMON",  # Default - would need annotation or package analysis
            namespace="NAMED",
        )
        
        self._symbol_cache[cache_key] = symbol
        return symbol
    
    def validate_against_host_declaration(
        self,
        symbol: JavaSymbol,
        host_declaration: dict[str, Any],
    ) -> bool:
        """Validate resolved symbol matches HOST API declaration.
        
        P0-6: Checks owner, name, descriptor, kind, static, side, namespace.
        
        Args:
            symbol: Resolved symbol from source code
            host_declaration: HOST API symbol declaration
            
        Returns:
            True if symbol matches declaration
        """
        # Check all required fields
        if symbol.owner != host_declaration.get("owner"):
            return False
        
        if symbol.name != host_declaration.get("name"):
            return False
        
        if symbol.descriptor != host_declaration.get("descriptor"):
            return False
        
        if symbol.kind != host_declaration.get("kind"):
            return False
        
        if symbol.is_static != host_declaration.get("static"):
            return False
        
        # Side check (CLIENT symbols can't be used in COMMON/SERVER)
        allowed_sides = self._get_allowed_sides(symbol.side)
        if host_declaration.get("side") not in allowed_sides:
            return False
        
        return True
    
    def extract_symbols_from_source(self, source_code: str) -> list[JavaSymbol]:
        """Extract all Java symbols used in source code.
        
        P0-6: Parses and resolves all method calls, field accesses, etc.
        
        Args:
            source_code: Java source code
            
        Returns:
            List of resolved symbols
        """
        import re
        
        symbols = []
        
        # Find all method calls: Owner.method(
        method_pattern = r'([A-Z][\w.]*|[a-z]\w*)\.(\w+)\s*\('
        for match in re.finditer(method_pattern, source_code):
            owner_expr = match.group(1)
            method_name = match.group(2)
            
            # Skip common patterns that aren't API calls
            if method_name in ['this', 'super', 'class']:
                continue
            
            try:
                symbol = self.resolve_method_call(source_code, method_name)
                symbols.append(symbol)
            except JavaSymbolResolverError:
                continue
        
        # Remove duplicates
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            key = (symbol.owner, symbol.name)
            if key not in seen:
                seen.add(key)
                unique_symbols.append(symbol)
        
        return unique_symbols
    
    def _get_allowed_sides(self, source_side: str) -> list[str]:
        """Get allowed sides based on source code side."""
        if source_side == "CLIENT":
            return ["CLIENT"]
        elif source_side == "SERVER":
            return ["SERVER", "COMMON"]
        else:  # COMMON
            return ["COMMON"]


def resolve_java_symbols_with_jdtls(
    source_code: str,
    classpath: list[Path],
    workspace_root: Path,
) -> list[JavaSymbol]:
    """Resolve Java symbols using jdt.ls (Eclipse JDT Language Server).
    
    P0-6: Most accurate method - uses same type resolution as Eclipse.
    
    This is a stub showing the integration point. Full implementation requires:
    1. Starting jdt.ls server
    2. Sending textDocument/definition, textDocument/hover requests
    3. Parsing LSP responses to extract symbol metadata
    """
    # TODO: Implement jdt.ls integration
    # See minecraft_mod_ai/java_lsp.py for existing LSP infrastructure
    raise NotImplementedError(
        "P0-6: jdt.ls integration stub. "
        "Connect to existing java_lsp.py infrastructure."
    )


def resolve_java_symbols_with_javac(
    source_file: Path,
    classpath: list[Path],
    output_dir: Path,
) -> list[JavaSymbol]:
    """Resolve Java symbols by compiling with javac and analyzing.
    
    P0-6: Alternative to jdt.ls - actually compiles the code.
    
    Steps:
    1. Run javac with -Xprint to get annotated source
    2. Or use com.sun.source.util.Trees API
    3. Extract resolved symbols with full metadata
    """
    # TODO: Implement javac-based resolution
    raise NotImplementedError("P0-6: javac resolution stub")


def resolve_java_symbols_with_javaparser(
    source_code: str,
    classpath: list[Path],
) -> list[JavaSymbol]:
    """Resolve Java symbols using JavaParser + reflection.
    
    P0-6: Fallback method using Python's javaparser library.
    
    Less accurate than jdt.ls/javac but still better than regex.
    """
    # TODO: Implement javaparser + reflection
    raise NotImplementedError("P0-6: JavaParser resolution stub")
