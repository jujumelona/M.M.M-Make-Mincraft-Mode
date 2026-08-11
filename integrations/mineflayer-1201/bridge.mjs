import readline from "node:readline";
import mineflayer from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

const SUPPORTED_MINECRAFT_VERSIONS = new Set(["1.20.1", "1.21.1"]);
const TARGET_VERSION = String(process.env.MMM_MINEFLAYER_MC_VERSION || "1.20.1").trim();
if (!SUPPORTED_MINECRAFT_VERSIONS.has(TARGET_VERSION)) {
  throw new Error(`Unsupported MMM Mineflayer target: ${TARGET_VERSION}`);
}

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
    throw new Error("The local MMM profile permits Mineflayer only on localhost");
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
    version: TARGET_VERSION,
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
  if (bot.version !== TARGET_VERSION) {
    throw new Error(`Mineflayer connected as ${bot.version}, expected ${TARGET_VERSION}`);
  }
  return status();
}

function status() {
  if (!bot) return { connected: false, version: TARGET_VERSION };
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
  const condition = String(params.condition || "");
  const timeoutMs = boundedInteger(params.timeout_ms ?? 10000, "wait timeout", 1, 60000);
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    if (condition === "spawned" && current.entity) {
      return { matched: true, condition };
    }
    if (condition === "window_open" && current.currentWindow) {
      return { matched: true, condition };
    }
    if (condition === "window_closed" && !current.currentWindow) {
      return { matched: true, condition };
    }
    if (condition === "healthy" && current.health > 0) {
      return { matched: true, condition, health: current.health };
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  return { matched: false, condition };
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
