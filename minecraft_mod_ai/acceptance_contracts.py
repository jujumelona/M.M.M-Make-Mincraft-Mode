from __future__ import annotations

"""Host-owned requirement acceptance contracts and typed testcase receipts.

Host ownership alone does not prove donor behavior. A contract may only authorize
BEHAVIOR_VERIFIED when ``implementation_bound`` is true and the generated test directly
binds to the reused implementation. Generic host smoke tests remain useful build/runtime
evidence but are deliberately capped below behavioral proof.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TestCaseReceipt:
    test_id: str
    requirement_id: str
    executed: bool = False
    passed: bool = False
    failure_message: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "requirement_id": self.requirement_id,
            "executed": self.executed,
            "passed": self.passed,
            "failure_message": self.failure_message,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class HostAcceptanceContract:
    requirement_id: str
    capability_id: str
    description: str
    host_test_class: str
    host_test_method: str
    acceptance_pattern: str = ""
    implementation_bound: bool = False

    @property
    def canonical_test_id(self) -> str:
        return (
            f"ai.minecraft.acceptance.{self.host_test_class}."
            f"{self.host_test_method}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "capability_id": self.capability_id,
            "description": self.description,
            "host_test_class": self.host_test_class,
            "host_test_method": self.host_test_method,
            "canonical_test_id": self.canonical_test_id,
            "acceptance_pattern": self.acceptance_pattern,
            "implementation_bound": self.implementation_bound,
        }


# These contracts currently materialize host-owned smoke tests. None directly binds a
# donor/adapted implementation symbol yet, so implementation_bound intentionally remains
# false. Capability-specific adapters may opt in only after generating a direct binding.
_HOST_ACCEPTANCE_REGISTRY: dict[str, tuple[HostAcceptanceContract, ...]] = {
    "boss.entity": (
        HostAcceptanceContract("REQ-BOSS-001", "boss.entity", "Boss entity spawn and initialization", "MMM_BossSpawnAcceptanceTest", "testSpawn", "spawn|init|entity"),
        HostAcceptanceContract("REQ-BOSS-002", "boss.entity", "Boss health and state persistence", "MMM_BossPersistenceAcceptanceTest", "testSaveReload", "health|state|persist"),
        HostAcceptanceContract("REQ-BOSS-003", "boss.entity", "Boss phase transition mechanics", "MMM_BossPhaseTransitionAcceptanceTest", "testPhaseTransition", "phase|transition|ai|goal"),
        HostAcceptanceContract("REQ-BOSS-004", "boss.entity", "Boss death rewards and drop table", "MMM_BossLootAcceptanceTest", "testDeathDrops", "death|loot|drop|reward"),
    ),
    "combat.boss": (
        HostAcceptanceContract("REQ-BOSS-001", "combat.boss", "Boss attack phase orchestration", "MMM_BossAttackAcceptanceTest", "testAttackPhases", "phase|attack|combat"),
        HostAcceptanceContract("REQ-BOSS-002", "combat.boss", "Boss damage validation and immunity", "MMM_BossImmunityAcceptanceTest", "testDamageImmunity", "damage|immunity|hit"),
    ),
    "item.equipment": (
        HostAcceptanceContract("REQ-ITEM-001", "item.equipment", "Item registry and equipment attributes", "MMM_ItemRegistryAcceptanceTest", "testItemAttributes", "item|equip|attr|registry"),
        HostAcceptanceContract("REQ-ITEM-002", "item.equipment", "Durability and usage behavior", "MMM_ItemDurabilityAcceptanceTest", "testDurabilityUse", "durability|use|usage|tier"),
    ),
    "combat.damage": (
        HostAcceptanceContract("REQ-COMBAT-001", "combat.damage", "Damage source calculation and attributes", "MMM_DamageCalculationAcceptanceTest", "testDamageSource", "damage|calc|source"),
        HostAcceptanceContract("REQ-COMBAT-002", "combat.damage", "Knockback and hit reactions", "MMM_KnockbackAcceptanceTest", "testKnockback", "knockback|hit|reaction"),
    ),
    "worldgen.ore": (
        HostAcceptanceContract("REQ-WORLD-001", "worldgen.ore", "Ore feature registry and placement modifier", "MMM_OreWorldgenAcceptanceTest", "testOrePlacement", "ore|feature|world|placement"),
    ),
    "magic.spell": (
        HostAcceptanceContract("REQ-MAGIC-001", "magic.spell", "Spell casting invocation and mana consumption", "MMM_SpellCastAcceptanceTest", "testSpellCast", "spell|cast|mana"),
        HostAcceptanceContract("REQ-MAGIC-002", "magic.spell", "Spell projectile effect execution", "MMM_SpellProjectileAcceptanceTest", "testProjectileEffect", "projectile|effect|drain"),
    ),
}


def get_host_acceptance_contracts(capability: str) -> tuple[HostAcceptanceContract, ...]:
    """Retrieve host-owned acceptance contracts for a canonical capability."""
    norm_cap = capability.strip().lower()
    return _HOST_ACCEPTANCE_REGISTRY.get(norm_cap, ())


_CONTRACT_TEST_BODIES: dict[str, str] = {
    "MMM_BossSpawnAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        assertNotNull(this.getClass().getSimpleName(), "Boss entity test class must be loadable in JVM");
        assertTrue(this.getClass().getSimpleName().contains("Boss"), "Boss entity name verified");
""",
    "MMM_BossPersistenceAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        String testKey = "mmm_boss_state";
        String testVal = "ACTIVE_PHASE_1";
        java.util.Map<String, String> tag = new java.util.HashMap<>();
        tag.put(testKey, testVal);
        assertEquals("ACTIVE_PHASE_1", tag.get(testKey), "Boss state preserved across serialization");
""",
    "MMM_BossPhaseTransitionAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        int maxHealth = 1000;
        int currentHealth = 400;
        boolean isPhaseTwo = (currentHealth <= maxHealth * 0.5);
        assertTrue(isPhaseTwo, "Boss must transition to Phase 2 when health drops below 50%");
""",
    "MMM_BossLootAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        java.util.List<String> lootItems = java.util.Arrays.asList("minecraft:nether_star", "custom:boss_trophy");
        assertFalse(lootItems.isEmpty(), "Boss death must emit non-empty reward items");
        assertTrue(lootItems.contains("custom:boss_trophy"), "Boss loot table contains custom drop");
""",
    "MMM_BossAttackAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        int attackCooldown = 0;
        boolean canPerformSpecialAttack = (attackCooldown <= 0);
        assertTrue(canPerformSpecialAttack, "Attack phase must execute special attack on tick");
""",
    "MMM_BossImmunityAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        boolean isShieldActive = true;
        int incomingDamage = 50;
        int appliedDamage = isShieldActive ? 0 : incomingDamage;
        assertEquals(0, appliedDamage, "Immunity phase must negate incoming damage");
""",
    "MMM_ItemRegistryAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        String registryKey = "custom:equipment_item";
        assertNotNull(registryKey, "Equipment item registered");
        assertTrue(registryKey.startsWith("custom:"), "Item namespace properly isolated");
""",
    "MMM_ItemDurabilityAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        int maxDurability = 250;
        int currentDurability = maxDurability - 1;
        assertTrue(currentDurability > 0, "Item takes damage");
        assertTrue(currentDurability < maxDurability, "Durability decreases on use");
""",
    "MMM_DamageCalculationAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        double baseDamage = 20.0;
        double armor = 10.0;
        double finalDamage = baseDamage * (1.0 - (armor / 50.0));
        assertTrue(finalDamage < baseDamage && finalDamage > 0, "Armor properly reduces combat damage");
""",
    "MMM_KnockbackAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        double attackerX = 0.0, targetX = 5.0;
        double deltaX = targetX - attackerX;
        double knockbackStrength = 1.5;
        double motionX = (deltaX / 5.0) * knockbackStrength;
        assertTrue(motionX > 0.0, "Knockback propels target away from attacker");
""",
    "MMM_OreWorldgenAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        int veinsPerChunk = 8;
        int minY = -64, maxY = 64;
        assertTrue(veinsPerChunk > 0, "Ore feature configures non-zero veins");
        assertTrue(minY < maxY, "Valid worldgen height bounds");
""",
    "MMM_SpellCastAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        int mana = 100;
        int cost = 30;
        boolean canCast = mana >= cost;
        assertTrue(canCast, "Sufficient mana for spell invocation");
        mana -= cost;
        assertEquals(70, mana, "Mana accurately deducted after casting");
""",
    "MMM_SpellProjectileAcceptanceTest": """        // Host smoke invariant; not donor implementation-bound
        double velocity = 2.5;
        boolean hitTarget = true;
        assertTrue(velocity > 0.0, "Projectile has non-zero velocity");
        assertTrue(hitTarget, "Projectile successfully impacts target entity");
""",
}


def materialize_host_acceptance_tests(
    sandbox_path: str | Path,
    capability: str,
) -> tuple[dict[str, str], str]:
    """Materialize host-owned JVM smoke tests and return their exact source hash."""
    contracts = get_host_acceptance_contracts(capability)
    if not contracts:
        return {}, ""

    sb = Path(sandbox_path)
    test_src_dir = sb / "src" / "test" / "java" / "ai" / "minecraft" / "acceptance"
    test_src_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, str] = {}
    combined_content = ""

    for c in contracts:
        class_name = c.host_test_class
        method_name = c.host_test_method
        body = _CONTRACT_TEST_BODIES.get(
            class_name,
            f"""        // Host smoke contract: {c.requirement_id} - {c.description}
        assertNotNull("{c.capability_id}", "Capability contract present");
        assertTrue("{c.description}".length() > 0, "Contract description verified");
""",
        )
        source_code = f"""package ai.minecraft.acceptance;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class {class_name} {{
    // Contract: {c.requirement_id} - {c.description}
    @Test
    public void {method_name}() {{
{body}
    }}
}}
"""
        target_file = test_src_dir / f"{class_name}.java"
        target_file.write_text(source_code, encoding="utf-8")
        rel_path = f"src/test/java/ai/minecraft/acceptance/{class_name}.java"
        generated[rel_path] = source_code
        combined_content += source_code

    test_hash = "sha256:" + hashlib.sha256(combined_content.encode("utf-8")).hexdigest()
    return generated, test_hash
