from __future__ import annotations

"""Compatibility gate for source-owned atomic evidence routing.

Evidence routing is part of the atomic IR schema and is therefore computed directly by
atomic_requirement_contract. This module no longer mutates compile_ir/semantic_review at
runtime; install() only verifies that the canonical source owns the contract.
"""

from collections.abc import Mapping
from typing import Any


def _routes_for_atom(
    proposal: Any,
    atom: Mapping[str, Any],
    production_contract_module: Any = None,
) -> list[str]:
    del production_contract_module
    from .atomic_requirement_contract import _evidence_dimensions

    return _evidence_dimensions(proposal, atom)


def _route_ir(
    proposal: Any,
    ir: dict[str, Any],
    atomic_module: Any,
    production_contract_module: Any = None,
) -> dict[str, Any]:
    del production_contract_module
    atoms = []
    for raw in ir.get('atoms', []):
        atom = dict(raw)
        atom['evidence_dimensions'] = atomic_module._evidence_dimensions(
            proposal, atom
        )
        atoms.append(atom)
    routed = {**ir, 'atoms': atoms, 'ir_sha256': ''}
    routed['ir_sha256'] = atomic_module._hash_without(routed, 'ir_sha256')
    return routed


def install(atomic_module: Any, production_contract_module: Any) -> None:
    del production_contract_module
    for name in ('compile_ir', 'semantic_review'):
        value = getattr(atomic_module, name)
        if not getattr(value, '_mmm_atomic_evidence_routes', False):
            raise RuntimeError(
                f'Atomic evidence routing must be source-owned: {name}'
            )


__all__ = ['install']
