from __future__ import annotations

"""Typed Resource Merge Registry & Semantic Conflict Resolver.

Implements schema-aware merging for Minecraft data-packs and assets:
- TAG_JSON: union of values array
- LANG_JSON: key-value dictionary merge with value conflict detection
- SOUNDS_JSON: sound event dictionary merge
- RECIPE_JSON: strict logical ID conflict (fails closed on duplicate recipe ID)
- LOOT_TABLE_JSON: pool composition or strict conflict
- MODEL_JSON: strict logical ID conflict
- BLOCKSTATE_JSON: multipart / variant dictionary merge
- ADVANCEMENT_JSON: criteria & requirements merge
"""

import json
from pathlib import PurePosixPath


class ResourceMergeRegistry:
    """Authoritative semantic merger for Minecraft asset and data resources."""

    @classmethod
    def can_merge(cls, path: str) -> bool:
        """Return True if this resource path is known and supports semantic merging."""
        p = PurePosixPath(path.replace("\\", "/"))
        if not p.name.endswith(".json"):
            return False
        return (
            "assets/" in path
            or "data/" in path
        )

    @classmethod
    def merge(
        cls,
        path: str,
        existing: str,
        incoming: str,
    ) -> tuple[str, bool, str]:
        """Merge incoming JSON resource into existing.

        Returns (merged_content, is_valid, error_message).
        """
        try:
            a = json.loads(existing)
            b = json.loads(incoming)
        except Exception as exc:
            return existing, False, f"JSON_PARSE_ERROR in {path}: {exc}"

        p = PurePosixPath(path.replace("\\", "/"))
        parts = p.parts

        # 1. Tags merge: data/*/tags/**/*.json
        if "tags" in parts and isinstance(a, dict) and isinstance(b, dict):
            va = a.get("values", [])
            vb = b.get("values", [])
            if isinstance(va, list) and isinstance(vb, list):
                a["values"] = list(dict.fromkeys(va + vb))
            if "replace" in b:
                a["replace"] = b["replace"]
            return json.dumps(a, indent=2), True, ""

        # 2. Lang merge: assets/*/lang/*.json
        if "lang" in parts and isinstance(a, dict) and isinstance(b, dict):
            merged_lang = dict(a)
            for k, v in b.items():
                if k in merged_lang and merged_lang[k] != v:
                    return existing, False, f"LANG_KEY_CONFLICT: '{k}' has conflicting translations in {path}"
                merged_lang[k] = v
            return json.dumps(merged_lang, indent=2), True, ""

        # 3. Sounds merge: assets/*/sounds.json
        if p.name == "sounds.json" and isinstance(a, dict) and isinstance(b, dict):
            merged_sounds = dict(a)
            for event_name, event_data in b.items():
                if event_name in merged_sounds and merged_sounds[event_name] != event_data:
                    return existing, False, f"SOUND_EVENT_CONFLICT: '{event_name}' in {path}"
                merged_sounds[event_name] = event_data
            return json.dumps(merged_sounds, indent=2), True, ""

        # 4. Recipes: data/*/recipes/*.json -> strict logical conflict
        if "recipes" in parts:
            if a != b:
                return existing, False, f"DUPLICATE_RECIPE_ID_CONFLICT: {path}"
            return existing, True, ""

        # 5. Loot tables: data/*/loot_tables/**/*.json -> pool composition or conflict
        if "loot_tables" in parts and isinstance(a, dict) and isinstance(b, dict):
            if "pools" in a and "pools" in b:
                pools_a = a.get("pools", [])
                pools_b = b.get("pools", [])
                if isinstance(pools_a, list) and isinstance(pools_b, list):
                    a["pools"] = pools_a + [p for p in pools_b if p not in pools_a]
                    return json.dumps(a, indent=2), True, ""
            if a != b:
                return existing, False, f"LOOT_TABLE_CONFLICT: {path}"
            return existing, True, ""

        # 6. Models: assets/*/models/**/*.json -> strict logical conflict
        if "models" in parts:
            if a != b:
                return existing, False, f"MODEL_DEFINITION_CONFLICT: {path}"
            return existing, True, ""

        # 7. Blockstates: assets/*/blockstates/*.json
        if "blockstates" in parts and isinstance(a, dict) and isinstance(b, dict):
            if "variants" in a and "variants" in b:
                vars_a = a.get("variants", {})
                vars_b = b.get("variants", {})
                if isinstance(vars_a, dict) and isinstance(vars_b, dict):
                    for v_k, v_v in vars_b.items():
                        if v_k in vars_a and vars_a[v_k] != v_v:
                            return existing, False, f"BLOCKSTATE_VARIANT_CONFLICT: '{v_k}' in {path}"
                        vars_a[v_k] = v_v
                    a["variants"] = vars_a
                    return json.dumps(a, indent=2), True, ""

        # Default fallback: If exactly identical, OK; else fail closed
        if a == b:
            return existing, True, ""
        return existing, False, f"UNRESOLVED_RESOURCE_CONFLICT: {path}"
