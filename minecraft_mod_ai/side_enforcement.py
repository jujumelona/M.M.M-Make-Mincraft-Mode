"""Side enforcement for CLIENT/SERVER/COMMON separation.

P1-4: Prevent client APIs from leaking into common/server code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SideViolation:
    """A detected side violation."""
    leaf_id: str
    leaf_side: str  # The side of the leaf being executed
    symbol_name: str
    symbol_side: str  # The side of the API symbol being referenced
    line_number: int | None
    message: str


class SideEnforcementError(Exception):
    """Error when side rules are violated."""
    
    def __init__(self, violations: list[SideViolation]):
        self.violations = violations
        messages = [f"  - {v.message}" for v in violations]
        super().__init__(
            f"Side enforcement violations ({len(violations)}):\n" + "\n".join(messages)
        )


# P1-4: Source set rules
SOURCE_SET_RULES = {
    "client": {
        "allowed_source_sets": ["client", "common"],
        "allowed_symbol_sides": ["CLIENT", "COMMON"],
        "description": "Client code can reference client and common APIs",
    },
    "common": {
        "allowed_source_sets": ["common"],
        "allowed_symbol_sides": ["COMMON"],
        "description": "Common code can only reference common APIs",
    },
    "server": {
        "allowed_source_sets": ["server", "common"],
        "allowed_symbol_sides": ["SERVER", "COMMON"],
        "description": "Server code can reference server and common APIs",
    },
}


def validate_side_constraints(
    leaf_id: str,
    leaf_side: str,
    symbols_used: list[Any],
    host_symbols: dict[str, Any],
) -> list[SideViolation]:
    """Validate that a leaf doesn't violate side constraints.
    
    P1-4: Checks that leaf only references symbols from allowed sides.
    
    Args:
        leaf_id: Canonical leaf ID
        leaf_side: Side of the leaf (CLIENT, SERVER, COMMON)
        symbols_used: List of symbols extracted from template/code
        host_symbols: HOST API symbol declarations
        
    Returns:
        List of violations (empty if valid)
    """
    violations = []
    
    # Get rules for this side
    rules = SOURCE_SET_RULES.get(leaf_side.lower())
    if not rules:
        raise ValueError(f"UNKNOWN_SOURCE_SIDE: {leaf_side}")
    
    allowed_sides = set(rules["allowed_symbol_sides"])
    
    # Check each symbol
    for symbol in symbols_used:
        symbol_name = getattr(symbol, 'name', str(symbol))
        symbol_qualified = getattr(symbol, 'qualified_name', symbol_name)
        
        # Look up in HOST symbols
        host_decl = host_symbols.get(symbol_qualified, {})
        symbol_side = getattr(symbol, "side", None) or host_decl.get("side")
        if symbol_side is None:
            raise ValueError(f"SYMBOL_SIDE_UNRESOLVED: {symbol_qualified}")
        
        # Check if allowed
        if symbol_side not in allowed_sides:
            line_number = getattr(symbol, 'line_number', None)
            violations.append(SideViolation(
                leaf_id=leaf_id,
                leaf_side=leaf_side,
                symbol_name=symbol_name,
                symbol_side=symbol_side,
                line_number=line_number,
                message=f"Leaf {leaf_id} (side={leaf_side}) cannot reference "
                       f"{symbol_name} (side={symbol_side}). "
                       f"Allowed sides: {allowed_sides}"
            ))
    
    return violations


def enforce_side_at_compile_time(
    java_files: list[str],
    leaf_side: str,
    classpath: list[str],
) -> bool:
    """Resolve source references against the supplied actual classpath and enforce side."""
    from pathlib import Path
    from .integrity_validators import validate_side
    import subprocess
    import shutil
    import re
    compiler = shutil.which("javac")
    if not compiler:
        raise ValueError("JAVAC_REQUIRED")
    version = subprocess.run([compiler, "-version"], capture_output=True, text=True, check=True)
    match = re.search(r"javac (\d+)", version.stdout + version.stderr)
    if not match or not java_files:
        raise ValueError("JAVA_TOOLCHAIN_OR_SOURCES_MISSING")
    for file in java_files:
        validate_side(Path(file).read_text(encoding="utf-8"), leaf_id=str(file), side=leaf_side,
                      classpath=[Path(p) for p in classpath], java_version=match[1])
    return True


def get_allowed_sides_for_leaf(leaf_side: str) -> list[str]:
    """Get list of symbol sides that a leaf side can reference.
    
    Args:
        leaf_side: CLIENT, SERVER, or COMMON
        
    Returns:
        List of allowed symbol sides
    """
    rules = SOURCE_SET_RULES.get(leaf_side.lower())
    if rules is None:
        raise ValueError(f"UNKNOWN_SOURCE_SIDE: {leaf_side}")
    return rules["allowed_symbol_sides"]


def check_side_compatibility(
    source_side: str,
    target_side: str,
) -> tuple[bool, str]:
    """Check if source side can reference target side.
    
    P1-4: Simple side compatibility check.
    
    Returns:
        (is_compatible, reason)
    """
    allowed = get_allowed_sides_for_leaf(source_side)
    
    if target_side in allowed:
        return True, "OK"
    else:
        return False, (
            f"Side {source_side} cannot reference side {target_side}. "
            f"Allowed: {allowed}"
        )
