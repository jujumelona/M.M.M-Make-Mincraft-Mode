import readline from "node:readline";
import mineflayer from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

let bot = null;

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

function blockAtParams(current, params) {
  const x = boundedInteger(params.x, "block x", -30000000, 30000000);
  const y = boundedInteger(params.y, "block y", -2048, 2048);
  const z = boundedInteger(params.z, "block z", -30000000, 30000000);
  const block = current.blockAt(new Vec3(x, y, z));
  if (!block) throw new Error("Block not loaded");
  return { block, x, y, z };
}

async function connect(params) {
  if (bot) throw new Error("Mineflayer bot is already connected");
  const host = String(params.host || "127.0.0.1");
  const port = Number(params.port || 25565);
  const username = String(params.username || "MMMTestBot");
  if (!["127.0.0.1", "localhost"].includes(host)) {
    throw new Error("The T4/local profile permits Mineflayer only on localhost");
  }
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("Invalid Minecraft server port");
  }
  if (!/^[A-Za-z0-9_]{1,16}$/.test(username)) {
    throw new Error("Invalid Minecraft bot username");
  }
  bot = mineflayer.createBot({
    host,
    port,
    username,
    version: "1.20.1",
    auth: "offline"
  });
  bot.loadPlugin(pathfinder);
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Mineflayer spawn timed out")), 60000);
    bot.once("spawn", () => {
      clearTimeout(timeout);
      const movement = new Movements(bot);
      bot.pathfinder.setMovements(movement);
      resolve();
    });
    bot.once("error", reject);
    bot.once("kicked", reason => reject(new Error(`kicked: ${reason}`)));
  });
  return status();
}

function status() {
  if (!bot) return { connected: false, version: "1.20.1" };
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
  current.chat(message);
  return { sent: message };
}

async function craft(params) {
  const current = requireBot();
  const count = boundedInteger(params.count ?? 1, "craft count", 1, 64);
  const rawName = safeRegistryName(params.item, "craft item");
  const shortName = rawName.includes(":") ? rawName.split(":", 2)[1] : rawName;
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
    title: String(window.title || ""),
    slotCount: Number(window.slots?.length || 0)
  };
}

async function clickSlot(params) {
  const current = requireBot();
  if (!current.currentWindow) throw new Error("No container window is open");
  const slot = boundedInteger(params.slot, "window slot", 0, Math.max(0, current.currentWindow.slots.length - 1));
  const mouseButton = boundedInteger(params.mouse_button ?? 0, "mouse button", 0, 1);
  const mode = boundedInteger(params.mode ?? 0, "click mode", 0, 0);
  await current.clickWindow(slot, mouseButton, mode);
  const item = current.currentWindow?.slots?.[slot] || null;
  return {
    slot,
    mouseButton,
    mode,
    item: item ? { name: item.name, count: item.count } : null
  };
}

function observeCondition(current, condition) {
  if (!condition || typeof condition !== "object" || Array.isArray(condition)) {
    throw new Error("wait_for condition must be an object");
  }
  const type = String(condition.type || "");
  if (type === "inventory_contains") {
    const name = safeRegistryName(condition.item, "inventory item");
    const shortName = name.includes(":") ? name.split(":", 2)[1] : name;
    const minCount = boundedInteger(condition.min_count ?? 1, "inventory min_count", 1, 4096);
    const count = current.inventory.items()
      .filter(item => item.name === shortName || item.name === name)
      .reduce((sum, item) => sum + item.count, 0);
    return { matched: count >= minCount, observation: { type, item: name, count, minCount } };
  }
  if (type === "held_item") {
    const name = safeRegistryName(condition.item, "held item");
    const shortName = name.includes(":") ? name.split(":", 2)[1] : name;
    const held = current.heldItem?.name || null;
    return { matched: held === name || held === shortName, observation: { type, expected: name, held } };
  }
  if (type === "health") {
    const comparison = String(condition.comparison || "gte");
    const value = finiteNumber(condition.value, "health value");
    if (!["gte", "lte", "eq"].includes(comparison)) throw new Error("Invalid health comparison");
    const actual = Number(current.health);
    const matched = comparison === "gte" ? actual >= value : comparison === "lte" ? actual <= value : actual === value;
    return { matched, observation: { type, comparison, value, actual } };
  }
  if (type === "food") {
    const comparison = String(condition.comparison || "gte");
    const value = finiteNumber(condition.value, "food value");
    if (!["gte", "lte", "eq"].includes(comparison)) throw new Error("Invalid food comparison");
    const actual = Number(current.food);
    const matched = comparison === "gte" ? actual >= value : comparison === "lte" ? actual <= value : actual === value;
    return { matched, observation: { type, comparison, value, actual } };
  }
  if (type === "position_near") {
    const x = finiteNumber(condition.x, "position x");
    const y = finiteNumber(condition.y, "position y");
    const z = finiteNumber(condition.z, "position z");
    const range = Math.max(0, Math.min(16, finiteNumber(condition.range ?? 1, "position range")));
    const distance = current.entity.position.distanceTo(new Vec3(x, y, z));
    return { matched: distance <= range, observation: { type, x, y, z, range, distance } };
  }
  if (type === "block_at") {
    const { block, x, y, z } = blockAtParams(current, condition);
    const expected = safeRegistryName(condition.name, "block name");
    const shortName = expected.includes(":") ? expected.split(":", 2)[1] : expected;
    return {
      matched: block.name === expected || block.name === shortName,
      observation: { type, expected, actual: block.name, x, y, z }
    };
  }
  if (type === "entity_present") {
    const expected = safeRegistryName(condition.name, "entity name");
    const maxDistance = Math.max(1, Math.min(64, finiteNumber(condition.max_distance ?? 16, "entity max_distance")));
    const entity = current.nearestEntity(candidate => {
      const name = String(candidate.name || candidate.mobType || "").toLowerCase();
      return name === expected.toLowerCase() && candidate.position.distanceTo(current.entity.position) <= maxDistance;
    });
    return {
      matched: Boolean(entity),
      observation: { type, expected, maxDistance, entityId: entity?.id ?? null }
    };
  }
  if (type === "window_open") {
    const matched = Boolean(current.currentWindow);
    return {
      matched,
      observation: { type, title: matched ? String(current.currentWindow.title || "") : "" }
    };
  }
  throw new Error(`Unsupported wait_for condition: ${type}`);
}

async function waitFor(params) {
  const current = requireBot();
  const timeoutMs = boundedInteger(params.timeout_ms ?? 10000, "wait_for timeout_ms", 100, 30000);
  const condition = params.condition;
  const deadline = Date.now() + timeoutMs;
  let last = null;
  while (Date.now() <= deadline) {
    last = observeCondition(current, condition);
    if (last.matched) {
      return { matched: true, condition, observation: last.observation, timeoutMs };
    }
    await current.waitForTicks(2);
  }
  return {
    matched: false,
    condition,
    observation: last?.observation || null,
    timeoutMs
  };
}

function disconnect() {
  if (bot) {
    bot.quit("MMM test complete");
    bot = null;
  }
  return { connected: false };
}

async function dispatch(action, params) {
  switch (action) {
    case "connect": return await connect(params);
    case "status": return status();
    case "walk_to": return await walkTo(params);
    case "interact_block": return await interactBlock(params);
    case "use_item": return await useItem(params);
    case "attack_entity": return await attackEntity(params);
    case "inventory": return inventory();
    case "chat": return chat(params);
    case "craft": return await craft(params);
    case "wait_for": return await waitFor(params);
    case "open_container": return await openContainer(params);
    case "click_slot": return await clickSlot(params);
    case "disconnect": return disconnect();
    default: throw new Error(`Unknown action: ${action}`);
  }
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  let request;
  try {
    request = JSON.parse(line);
    const result = await dispatch(request.action, request.params || {});
    process.stdout.write(JSON.stringify({ id: request.id, ok: true, result }) + "\n");
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: request?.id ?? null,
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    }) + "\n");
  }
}
