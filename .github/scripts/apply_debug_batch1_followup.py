from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def read(rel: str) -> tuple[Path, str]:
    path = ROOT / rel
    return path, path.read_text(encoding="utf-8")


# Late work-graph mutation owner must not reclassify audio-binding as LLM.
path, text = read("minecraft_mod_ai/work_graph_mutation_contract.py")
text = once(
    text,
    '_LLM_CAPABLE_STAGES = frozenset({"custom", "audio-binding"})\n',
    '_LLM_CAPABLE_STAGES = frozenset({"custom"})\n',
    "audio-binding commit lane",
)
path.write_text(text, encoding="utf-8")


# Runtime assertions used by autonomous playtesting. These predicates are bounded,
# read-only, and operate only on the already connected disposable localhost bot.
path, text = read("integrations/mineflayer-1201/bridge.mjs")
start = text.find("async function waitFor(params) {\n")
end = text.find("\nasync function disconnect()", start)
if start < 0 or end < 0:
    raise SystemExit("Mineflayer waitFor function markers not found")
wait_for = '''async function waitFor(params) {
  const current = requireBot();
  const spec = (params.condition && typeof params.condition === "object" && !Array.isArray(params.condition))
    ? params.condition
    : { type: String(params.condition || "") };
  const type = String(spec.type || "");
  const supported = new Set([
    "inventory_contains",
    "held_item",
    "health",
    "food",
    "position_near",
    "block_at",
    "entity_present",
    "window_open",
    "window_closed",
    "spawned",
    "healthy"
  ]);
  if (!supported.has(type)) {
    throw new Error(`Unsupported wait_for condition: ${type || "<empty>"}`);
  }
  const timeoutMs = boundedInteger(params.timeout_ms ?? 30000, "wait timeout", 1, 60000);
  const started = Date.now();

  const numericRangeMatches = (actual, conditionSpec) => {
    if (conditionSpec.value != null && actual !== finiteNumber(conditionSpec.value, `${type} value`)) return false;
    if (conditionSpec.min != null && actual < finiteNumber(conditionSpec.min, `${type} min`)) return false;
    if (conditionSpec.max != null && actual > finiteNumber(conditionSpec.max, `${type} max`)) return false;
    return conditionSpec.value != null || conditionSpec.min != null || conditionSpec.max != null;
  };

  while (Date.now() - started < timeoutMs) {
    if (type === "inventory_contains") {
      const raw = safeRegistryName(spec.item ?? spec.name, "inventory item");
      const shortName = raw.includes(":") ? raw.split(":", 2)[1] : raw;
      const minimum = boundedInteger(spec.count ?? spec.min_count ?? 1, "inventory count", 1, 2304);
      const total = current.inventory.items()
        .filter(item => item.name === shortName)
        .reduce((sum, item) => sum + item.count, 0);
      if (total >= minimum) return { matched: true, type, item: raw, count: total };
    } else if (type === "held_item") {
      const raw = safeRegistryName(spec.item ?? spec.name, "held item");
      const shortName = raw.includes(":") ? raw.split(":", 2)[1] : raw;
      if (current.heldItem?.name === shortName) {
        return { matched: true, type, item: raw };
      }
    } else if (type === "health") {
      if (numericRangeMatches(Number(current.health), spec)) {
        return { matched: true, type, health: current.health };
      }
    } else if (type === "food") {
      if (numericRangeMatches(Number(current.food), spec)) {
        return { matched: true, type, food: current.food };
      }
    } else if (type === "position_near") {
      const x = finiteNumber(spec.x, "position x");
      const y = finiteNumber(spec.y, "position y");
      const z = finiteNumber(spec.z, "position z");
      const range = Math.max(0, Math.min(64, finiteNumber(spec.range ?? 1, "position range")));
      const distance = current.entity.position.distanceTo(new Vec3(x, y, z));
      if (distance <= range) return { matched: true, type, distance };
    } else if (type === "block_at") {
      const { block, x, y, z } = blockAtParams(current, spec);
      const expectedRaw = spec.name ?? spec.block;
      if (expectedRaw == null) {
        return { matched: true, type, name: block.name, position: { x, y, z } };
      }
      const raw = safeRegistryName(expectedRaw, "block name");
      const shortName = raw.includes(":") ? raw.split(":", 2)[1] : raw;
      if (block.name === shortName) {
        return { matched: true, type, name: block.name, position: { x, y, z } };
      }
    } else if (type === "entity_present") {
      const raw = safeRegistryName(spec.name ?? spec.entity, "entity name");
      const shortName = raw.includes(":") ? raw.split(":", 2)[1] : raw;
      const maxDistance = Math.max(1, Math.min(64, finiteNumber(spec.max_distance ?? spec.range ?? 16, "entity distance")));
      const entity = current.nearestEntity(candidate => {
        const candidateName = String(candidate.name || candidate.mobType || "").toLowerCase();
        return candidateName === shortName && candidate.position.distanceTo(current.entity.position) <= maxDistance;
      });
      if (entity) return { matched: true, type, name: raw, entityId: entity.id };
    } else if (type === "window_open") {
      if (current.currentWindow) return { matched: true, type };
    } else if (type === "window_closed") {
      if (!current.currentWindow) return { matched: true, type };
    } else if (type === "spawned") {
      if (current.entity) return { matched: true, type };
    } else if (type === "healthy") {
      if (current.health > 0) return { matched: true, type, health: current.health };
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  return { matched: false, type };
}
'''
text = text[:start] + wait_for + text[end:]
path.write_text(text, encoding="utf-8")


# Canonical skills are target-dynamic. The approved PlatformLock supplies exact
# Minecraft/loader/Java/mapping versions; skill prose must not create new authority.
def dynamicize_skill(rel: str, *, description: str) -> None:
    path, text = read(rel)
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith("description:"):
            out.append(f"description: {description}")
        elif "Minecraft target is Java 1.20.1" in line:
            out.append("  - Minecraft target, loader, Java version and mappings come from the approved PlatformLock.")
        elif "Fabric 1.20.1 official documentation" in line:
            out.append("  - Official Fabric documentation and metadata for the approved PlatformLock target")
        elif "Yarn 1.20.1+build.1 symbols" in line:
            out.append("  - Mapping symbols for the exact approved PlatformLock target")
        elif "Generate the approved Fabric 1.20.1 core project" in line:
            out.append("  - Generate the approved Fabric target core project and registrations.")
        elif "mixing Fabric with Forge/NeoForge or another Minecraft version" in line:
            out.append("  - mixing the approved PlatformLock with another loader or Minecraft version")
        else:
            out.append(line)
    updated = "\n".join(out) + "\n"
    if "1.20.1" in updated or "Java 17" in updated or "Yarn 1.20.1" in updated:
        raise SystemExit(f"hard-coded target remains in {rel}")
    path.write_text(updated, encoding="utf-8")


dynamicize_skill(
    "skills/generate-fabric-core/SKILL.md",
    description="Generate the approved Fabric target core project and registrations.",
)
dynamicize_skill(
    "skills/runtime-playtest/SKILL.md",
    description="Run a disposable server/client for the approved target and complete bounded player interactions.",
)

# Keep wheel/package skill data byte-identical to source checkout skills.
packaged_path = ROOT / "minecraft_mod_ai/packaged_skills.json"
payload = json.loads(packaged_path.read_text(encoding="utf-8"))
skills = payload.get("skills")
if not isinstance(skills, dict):
    raise SystemExit("packaged_skills.json has no skills mapping")
for name in ("generate-fabric-core", "runtime-playtest"):
    source = (ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    if name not in skills:
        raise SystemExit(f"packaged skill missing: {name}")
    skills[name] = source
packaged_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# Either exact repeated model text or an unchanged semantic validator state is a
# valid cycle proof. The implementation may detect the semantic state first.
path, text = read("tests/test_production_page_durable_contract.py")
text = text.replace(
    '        match="repeated_model_output",\n',
    '        match="repeated_(validation_state|model_output)",\n',
    1,
)
path.write_text(text, encoding="utf-8")

print("debug batch 1 follow-up prepared")
