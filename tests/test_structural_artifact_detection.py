from __future__ import annotations

import unittest

from minecraft_mod_ai.structural_artifact_detection import (
    CANONICAL_ARTIFACT_KINDS,
    detect_artifacts,
    expand_artifact_dependencies,
    resolve_artifacts,
    structural_artifact_evidence,
    validate_artifact_kinds,
)


class StructuralArtifactDetectionTests(unittest.TestCase):
    def _feature(self, *, name: str, description: str) -> dict[str, object]:
        return {
            "id": name, "name": name, "description": description,
            "purpose": description, "behavior": description,
            "trigger": {"event": True}, "state": {"known": True},
            "persistence": {"required": True}, "networking": {"required": True},
            "ui": ["hud"], "resources": ["language"],
            "assets": ["texture", "sound"], "implementation_surfaces": ["mob"],
        }

    def test_different_names_same_structure_resolve_identically(self) -> None:
        left = self._feature(name="boss.skill.dungeon", description="boss trading economy skill")
        right = self._feature(name="totally_unrelated", description="plain neutral words")
        self.assertEqual(resolve_artifacts(left), resolve_artifacts(right))

    def test_name_and_behavior_tokens_never_enter_routing_evidence(self) -> None:
        evidence = structural_artifact_evidence(self._feature(name="boss_economy_trade", description="dimension block entity item"))
        for forbidden in ("id", "name", "description", "purpose", "behavior"):
            self.assertNotIn(forbidden, evidence)

    def test_domain_tokens_without_structural_surfaces_select_nothing(self) -> None:
        feature = {"id": "boss.entity", "name": "economy trading dungeon skill", "description": "item block mob dimension worldgen", "purpose": "network persistence ui", "behavior": "recipe loot texture"}
        self.assertEqual(detect_artifacts(feature), ())

    def test_dependencies_expand_deterministically(self) -> None:
        self.assertEqual(expand_artifact_dependencies(["structure", "mob", "block_entity"]), ("block", "block_entity", "entity", "mob", "worldgen", "structure"))

    def test_resolution_uses_structural_requirements(self) -> None:
        self.assertEqual(resolve_artifacts(self._feature(name="x", description="y")), ("entity", "mob", "hud", "event", "network_payload", "saved_data", "sound", "texture", "language"))

    def test_validator_rejects_legacy_artifact_labels(self) -> None:
        for legacy in ("entity_model", "gametest", "boss_combat"):
            with self.assertRaises(ValueError):
                validate_artifact_kinds([legacy])

    def test_canonical_allowlist_has_no_profile_labels(self) -> None:
        forbidden = {"boss_combat", "economy", "dungeon", "skill", "custom_gameplay"}
        self.assertTrue(forbidden.isdisjoint(CANONICAL_ARTIFACT_KINDS))


if __name__ == "__main__":
    unittest.main()
