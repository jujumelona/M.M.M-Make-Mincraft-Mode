import readline from "node:readline";
import mineflayer from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

let bot = null;
let targetVersion = null;

const WAIT_FOR_CONDITIONS = new Set([
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
const WAIT_POLL_MS = 100;

function resolveTargetVersion(params = {}) {
  const requested = String(params.minecraft_version ?? params.version ?? "").trim();
  const discovered = String(process.env.MMM_MINEFLAYER_MC_VERSION || "").trim();
  if (requested && discovered && requested !== discovered) {
    throw new Error(`Mineflayer target mismatch: request=${requested}, runtime=${discovered}`);
  }
  const resolved = requested || discovered;
  if (!resolved) {
    throw new Error("Mineflayer requires a runtime-discovered Minecraft target");
  }
  return resolved;
}

function requireBot() {
  if (!bot) throw new Error("Mineflayer bot is not connected");
  return bot;
}

function finiteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`Invalid ${label}`);
  return number;
}

function boundedInteger(value, label, min, max) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < min || number > max) {
    throw new Error(`Invalid ${label}`);
  }
  return number;
}

function safeRegistryName(value, label = "registry name") {
  const name = String(value || "").toLowerCase();
  if (!/^[a-z0-9_.-]+(?::[a-z0-9_./-]+)?$/.test(name) || name.length > 120) {
    throw new Error(`Invalid ${label}`);
  }
  return name;
}

function localRegistryName(name) {
  return name.includes(":") ? name.split(":", 2)[1] : name;
}

function numericRangeMatches(actual, conditionSpec, label) {
  if (conditionSpec.value != null && actual !== finiteNumber(conditionSpec.value, `${label} value`)) return false;
  if (conditionSpec.min != null && actual < finiteNumber(conditionSpec.min, `${label} min`)) return false;
  if (conditionSpec.max != null && actual > finiteNumber(conditionSpec.max, `${label} max`)) return false;
  return conditionSpec.value != null || conditionSpec.min != null || conditionSpec.max != null;
}

function blockAtParams(current, params) {
  const x = boundedInteger(params.x, "block x", -30000000, 30000000);
  const y = boundedInteger(params.y, "block y", -2048, 2048);
  const z = boundedInteger(params.z, "block z", -30000000, 30000000);
  const block = current.blockAt(new Vec3(x, y, z));
  if (!block) throw new Error("Block not loaded");
  return { block, x, y, z };
}

function abortBot(current, reason) {
  if (bot === current) bot = null;
  try {
    current.quit(reason);
    return;
  } catch (_) {
    // A connection that failed before login may not support quit(). Fall through to
    // the lower-level close path when available.
  }
  try {
    current.end?.();
  } catch (_) {
    // The Python supervisor also has a hard request deadline and will kill/reap this
    // bridge process if Mineflayer itself cannot close a broken socket.
  }
}

async function connect(params) {
  if (bot) throw new Error("Mineflayer bot is already connected");
  const host = String(params.host || "127.0.0.1");
  const port = Number(params.port || 25565);
  const username = String(params.username || "MMMTestBot");
  const resolvedTarget = resolveTargetVersion(params);
  if (!["127.0.0.1", "localhost"].includes(host)) {
    throw new Error("The local MMM profile permits Mineflayer only on localhost");
  }
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("Invalid Minecraft server port");
  }
  if (!/^[A-Za-z0-9_]{1,16}$/.test(username)) {
    throw new Error("Invalid Minecraft bot username");
  }

  targetVersion = resolvedTarget;
  const current = mineflayer.createBot({
    host,
    port,
    username,
    version: resolvedTarget,
    auth: "offline"
  });
  bot = current;
  current.loadPlugin(pathfinder);

  try {
    await new Promise((resolve, reject) => {
      let settled = false;
      let timeout = null;

      const cleanup = () => {
        if (timeout !== null) clearTimeout(timeout);
        current.removeListener("spawn", onSpawn);
        current.removeListener("error", onError);
        current.removeListener("kicked", onKicked);
        current.removeListener("end", onEnd);
      };
      const finish = error => {
        if (settled) return;
        settled = true;
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const onSpawn = () => {
        try {
          const movement = new Movements(current);
          current.pathfinder.setMovements(movement);
          finish(null);
        } catch (error) {
          finish(error);
        }
      };
      const onError = error => finish(error instanceof Error ? error : new Error(String(error)));
      const onKicked = reason => finish(new Error(`kicked: ${reason}`));
      const onEnd = reason => finish(new Error(`ended before spawn: ${reason || "unknown"}`));

      timeout = setTimeout(
        () => finish(new Error("Mineflayer spawn timed out")),
        60000
      );
      current.once("spawn", onSpawn);
      current.once("error", onError);
      current.once("kicked", onKicked);
      current.once("end", onEnd);
    });

    if (current.version !== resolvedTarget) {
      throw new Error(`Mineflayer connected as ${current.version}, expected ${resolvedTarget}`);
    }

    current.once("end", () => {
      if (bot === current) bot = null;
    });
    return status();
  } catch (error) {
    abortBot(current, "MMM connect failed");
    throw error;
  }
}

function status() {
  if (!bot) return { connected: false, version: targetVersion };
  return {
    connected: true,
    version: bot.version,
    username: bot.username,
    position: {
      x: bot.entity.position.x,
      y: bot.entity.position.y,
      z: bot.entity.position.z
    },
    health: bot.health,
    food: bot.food,
    heldItem: bot.heldItem?.name || null,
    windowOpen: Boolean(bot.currentWindow)
  };
}

async function walkTo(params) {
  const current = requireBot();
  const x = finiteNumber(params.x, "walk target x");
  const y = finiteNumber(params.y, "walk target y");
  const z = finiteNumber(params.z, "walk target z");
  const range = Math.max(1, Math.min(4, finiteNumber(params.range || 1, "walk range")));
  await current.pathfinder.goto(new goals.GoalNear(x, y, z, range));
  return status();
}

async function interactBlock(params) {
  const current = requireBot();
  const { block, x, y, z } = blockAtParams(current, params);
  await current.activateBlock(block);
  return { name: block.name, position: { x, y, z } };
}

async function useItem() {
  const current = requireBot();
  current.activateItem();
  await current.waitForTicks(10);
  current.deactivateItem();
  return { used: current.heldItem?.name || null };
}

async function attackEntity(params) {
  const current = requireBot();
  const allowedName = safeRegistryName(params.name, "entity name");
  const maxDistance = Math.max(1, Math.min(32, finiteNumber(params.max_distance || 8, "entity distance")));
  const entity = current.nearestEntity(candidate => {
    const name = String(candidate.name || candidate.mobType || "").toLowerCase();
    return name === allowedName.toLowerCase() && candidate.position.distanceTo(current.entity.position) <= maxDistance;
  });
  if (!entity) throw new Error(`No allowlisted entity found: ${allowedName}`);
  current.attack(entity);
  return { attacked: allowedName, entityId: entity.id };
}

function inventory() {
  const current = requireBot();
  return {
    items: current.inventory.items().map(item => ({
      name: item.name,
      count: item.count,
      slot: item.slot
    }))
  };
}

function chat(params) {
  const current = requireBot();
  const message = String(params.message || "");
  if (!message || message.length > 120 || /[\r\n\u0000-\u001f]/.test(message)) {
    throw new Error("Invalid chat message");
  }
  if (message.trimStart().startsWith("/")) {
    throw new Error("Mineflayer chat may not execute server commands");
  }
  current.chat(message);
  return { sent: message };
}

async function craft(params) {
  const current = requireBot();
  const count = boundedInteger(params.count ?? 1, "craft count", 1, 64);
  const rawName = safeRegistryName(params.item, "craft item");
  const shortName = localRegistryName(rawName);
  const item = current.registry?.itemsByName?.[shortName];
  if (!item) throw new Error(`Unknown craft item: ${rawName}`);
  let craftingTable = null;
  if (params.crafting_table != null) {
    if (typeof params.crafting_table !== "object" || Array.isArray(params.crafting_table)) {
      throw new Error("crafting_table must be coordinate object");
    }
    craftingTable = blockAtParams(current, params.crafting_table).block;
  }
  const recipes = current.recipesFor(item.id, null, count, craftingTable);
  if (!recipes.length) throw new Error(`No available recipe for: ${rawName}`);
  await current.craft(recipes[0], count, craftingTable);
  return { crafted: rawName, count };
}

async function openContainer(params) {
  const current = requireBot();
  const { block, x, y, z } = blockAtParams(current, params);
  const window = await current.openContainer(block);
  return {
    opened: block.name,
    position: { x, y, z },
    window: {
      title: String(window.title || ""),
      slots: window.slots.length
    }
  };
}

async function clickSlot(params) {
  const current = requireBot();
  if (!current.currentWindow) throw new Error("No container window is open");
  const slot = boundedInteger(params.slot, "slot", 0, current.currentWindow.slots.length - 1);
  const mouseButton = boundedInteger(params.mouse_button ?? 0, "mouse button", 0, 1);
  const mode = boundedInteger(params.mode ?? 0, "click mode", 0, 6);
  await current.clickWindow(slot, mouseButton, mode);
  return { clicked: slot, mouseButton, mode };
}

async function waitFor(params) {
  const current = requireBot();
  const spec = (params.condition && typeof params.condition === "object" && !Array.isArray(params.condition))
    ? params.condition
    : { type: String(params.condition || "") };
  const type = String(spec.type || "");
  if (!WAIT_FOR_CONDITIONS.has(type)) {
    throw new Error(`Unsupported wait_for condition: ${type || "<empty>"}`);
  }
  const timeoutMs = boundedInteger(params.timeout_ms ?? 30000, "wait timeout", 1, 60000);
  const started = Date.now();

  while (Date.now() - started < timeoutMs) {
    if (type === "inventory_contains") {
      const raw = safeRegistryName(spec.item ?? spec.name, "inventory item");
      const shortName = localRegistryName(raw);
      const minimum = boundedInteger(spec.count ?? spec.min_count ?? 1, "inventory count", 1, 2304);
      const total = current.inventory.items()
        .filter(item => item.name === shortName)
        .reduce((sum, item) => sum + item.count, 0);
      if (total >= minimum) return { matched: true, type, item: raw, count: total };
    } else if (type === "held_item") {
      const raw = safeRegistryName(spec.item ?? spec.name, "held item");
      const shortName = localRegistryName(raw);
      if (current.heldItem?.name === shortName) {
        return { matched: true, type, item: raw };
      }
    } else if (type === "health") {
      if (numericRangeMatches(Number(current.health), spec, type)) {
        return { matched: true, type, health: current.health };
      }
    } else if (type === "food") {
      if (numericRangeMatches(Number(current.food), spec, type)) {
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
      const shortName = localRegistryName(raw);
      if (block.name === shortName) {
        return { matched: true, type, name: block.name, position: { x, y, z } };
      }
    } else if (type === "entity_present") {
      const raw = safeRegistryName(spec.name ?? spec.entity, "entity name");
      const shortName = localRegistryName(raw);
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
    await new Promise(resolve => setTimeout(resolve, WAIT_POLL_MS));
  }
  return { matched: false, type };
}

async function disconnect() {
  if (!bot) return { disconnected: true };
  const current = bot;
  bot = null;
  current.quit("MMM test complete");
  return { disconnected: true };
}

const actions = {
  connect,
  status,
  walk_to: walkTo,
  interact_block: interactBlock,
  use_item: useItem,
  attack_entity: attackEntity,
  inventory,
  chat,
  craft,
  wait_for: waitFor,
  open_container: openContainer,
  click_slot: clickSlot,
  disconnect
};

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  let request = null;
  try {
    request = JSON.parse(line);
    const action = String(request.action || "");
    if (!Object.prototype.hasOwnProperty.call(actions, action)) {
      throw new Error(`Unsupported action: ${action}`);
    }
    const result = await actions[action](request.params || {});
    process.stdout.write(JSON.stringify({ id: request.id, ok: true, result }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: request?.id ?? null,
      ok: false,
      error: String(error?.message || error)
    }) + "\n");
  }
}
