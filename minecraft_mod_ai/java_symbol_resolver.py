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
    """Resolve overloads, instance calls, fields and constructors using javac Trees."""

    def __init__(self, classpath: list[Path], java_version: str = "21", *,
                 namespace: str = "NAMED", classpath_sides: dict[str, str] | None = None):
        self.classpath = [Path(p) for p in classpath]
        self.java_version = java_version
        self.namespace = namespace
        self.classpath_sides = classpath_sides or {}

    def _resolve(self, source_code: str, filename: str | None = None):
        from .javac_bridge import analyze_java
        from .jar_api_extractor import inspect_jar, parse_class
        import re
        # Only filename discovery uses text; all symbol identity comes from javac.
        if filename is None:
            match = re.search(r"public\s+(?:(?:final|abstract|sealed)\s+)*(?:class|interface|enum|record)\s+(\w+)", source_code)
            filename = (match[1] if match else "IntegrityInput") + ".java"
        try:
            rows = analyze_java(source_code, classpath=self.classpath,
                                java_version=self.java_version, filename=filename)
            classes = {}
            for path in self.classpath:
                default_side = self.classpath_sides.get(str(path), "COMMON")
                if path.is_dir():
                    found = {c.name: c for c in (parse_class(p.read_bytes(), default_side=default_side)
                                               for p in path.rglob("*.class"))}
                else:
                    found = inspect_jar(path, java_version=int(self.java_version), default_side=default_side)
                for name, item in found.items():
                    if name in classes and classes[name].content_hash != item.content_hash:
                        raise JavaSymbolResolverError("AMBIGUOUS_CLASSPATH_CLASS: " + name)
                    classes[name] = item
            output = []
            for row in rows:
                owner = classes.get(row["owner"])
                side = row["side"]
                if owner:
                    side = owner.side
                    if row["kind"] != "CLASS":
                        members = [m for m in owner.members if (m.name, m.descriptor) == (row["name"], row["descriptor"])]
                        if len(members) != 1:
                            raise JavaSymbolResolverError("MEMBER_NOT_IN_INSPECTED_CLASSPATH")
                        side = members[0].side
                # JDK and current compilation-unit declarations are resolved by the compiler.
                output.append((JavaSymbol(row["name"], row["owner"], row["descriptor"],
                                          row["kind"], row["is_static"], side, self.namespace), row["line"]))
            return output
        except (ValueError, OSError, RuntimeError) as exc:
            raise JavaSymbolResolverError(str(exc)) from exc

    def resolve_method_call(self, source_code: str, method_name: str,
                            line_number: int | None = None) -> JavaSymbol:
        matches = [s for s, line in self._resolve(source_code)
                   if s.name == method_name and s.kind in {"METHOD", "CONSTRUCTOR"}
                   and (line_number is None or line == line_number)]
        if len(matches) != 1:
            raise JavaSymbolResolverError(f"JAVA_CALL_NOT_UNIQUE: {method_name}: {len(matches)} matches")
        return matches[0]

    def extract_symbols_from_source(self, source_code: str) -> list[JavaSymbol]:
        return list(dict.fromkeys(s for s, _ in self._resolve(source_code)))

    def validate_against_host_declaration(self, symbol: JavaSymbol, host_declaration: dict[str, Any]) -> bool:
        return all(host_declaration.get(key) == value for key, value in {
            "owner": symbol.owner, "name": symbol.name, "descriptor": symbol.descriptor,
            "kind": symbol.kind, "static": symbol.is_static, "side": symbol.side,
            "namespace": symbol.namespace,
        }.items())


def resolve_java_symbols_with_javac(source_file: Path, classpath: list[Path], output_dir: Path) -> list[JavaSymbol]:
    # Tree attribution needs no emitted class files; preserve the historical API.
    resolver = JavaSymbolResolver(classpath)
    return [s for s, _ in resolver._resolve(source_file.read_text(encoding="utf-8"), source_file.name)]


def resolve_java_symbols_with_jdtls(source_code: str, classpath: list[Path], workspace_root: Path) -> list[JavaSymbol]:
    """Compatibility entry point using the installed JDK compiler backend."""
    return JavaSymbolResolver(classpath).extract_symbols_from_source(source_code)


def resolve_java_symbols_with_javaparser(source_code: str, classpath: list[Path]) -> list[JavaSymbol]:
    """Compatibility entry point using the installed JDK compiler backend."""
    return JavaSymbolResolver(classpath).extract_symbols_from_source(source_code)
