import readline from "node:readline";
import mineflayer from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

let bot = null;

function requireBot() {
  if (!bot) throw new Error("Mineflayer bot is not connected");
  return bot;
}

function finiteNumber(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`Invalid ${name}`);
  return number;
}

function boundedInteger(value, name, minimum, maximum) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) {
    throw new Error(`Invalid ${name}`);
  }
  return number;
}

function itemName(value) {
  const text = String(value || "");
  if (!/^[a-z0-9_.-]+(?::[a-z0-9_./-]+)?$/.test(text)) throw new Error("Invalid item name");
  return text.includes(":") ? text.split(":", 2)[1] : text;
}

async function connect(params) {
  if (bot) throw new Error("Mineflayer bot is already connected");
  const host = String(params.host || "127.0.0.1");
  const port = boundedInteger(params.port || 25565, "port", 1, 65535);
  const username = String(params.username || "MMMTestBot");
  if (!/^[A-Za-z0-9_]{3,16}$/.test(username)) throw new Error("Invalid bot username");
  if (!["127.0.0.1", "localhost"].includes(host)) {
    throw new Error("The local profile permits Mineflayer only on localhost");
  }
  bot = mineflayer.createBot({ host, port, username, version: "1.20.1", auth: "offline" });
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
    position: { x: bot.entity.position.x, y: bot.entity.position.y, z: bot.entity.position.z },
    health: bot.health,
    food: bot.food,
    window: bot.currentWindow ? { id: bot.currentWindow.id, title: String(bot.currentWindow.title || "") } : null
  };
}

async function walkTo(params) {
  const current = requireBot();
  const x = finiteNumber(params.x, "x");
  const y = finiteNumber(params.y, "y");
  const z = finiteNumber(params.z, "z");
  const range = Math.max(1, Math.min(16, finiteNumber(params.range || 1, "range")));
  await current.pathfinder.goto(new goals.GoalNear(x, y, z, range));
  return status();
}

async function interactBlock(params) {
  const current = requireBot();
  const x = finiteNumber(params.x, "x");
  const y = finiteNumber(params.y, "y");
  const z = finiteNumber(params.z, "z");
  const block = current.blockAt(new Vec3(x, y, z));
  if (!block) throw new Error("Block not loaded");
  await current.activateBlock(block);
  return { name: block.name, position: { x, y, z } };
}

async function useItem() {
  const current = requireBot();
  current.activateItem();
  await new Promise(resolve => setTimeout(resolve, 500));
  current.deactivateItem();
  return { used: current.heldItem?.name || null };
}

async function attackEntity(params) {
  const current = requireBot();
  const allowedName = String(params.name || "");
  if (!/^[a-z0-9_:-]{1,80}$/.test(allowedName)) throw new Error("Invalid entity name");
  const entity = current.nearestEntity(candidate => {
    const name = String(candidate.name || candidate.mobType || "").toLowerCase();
    return name === allowedName.toLowerCase();
  });
  if (!entity) throw new Error(`No entity found: ${allowedName}`);
  current.attack(entity);
  return { attacked: allowedName, entityId: entity.id };
}

function inventory() {
  const current = requireBot();
  return {
    items: current.inventory.items().map(item => ({ name: item.name, count: item.count, slot: item.slot }))
  };
}

function chat(params) {
  const current = requireBot();
  const message = String(params.message || "");
  if (!message || message.length > 256 || /[\r\n\0]/.test(message)) throw new Error("Invalid chat message");
  current.chat(message);
  return { sent: message };
}

async function craft(params) {
  const current = requireBot();
  const name = itemName(params.item);
  const count = boundedInteger(params.count || 1, "count", 1, 4096);
  const item = current.registry.itemsByName[name];
  if (!item) throw new Error(`Unknown item: ${name}`);
  const recipes = current.recipesFor(item.id, null, 1, current.currentWindow || null);
  if (!recipes.length) throw new Error(`No available recipe for ${name}`);
  await current.craft(recipes[0], count, null);
  return { crafted: name, count };
}

async function waitFor(params) {
  const current = requireBot();
  const kind = String(params.kind || "");
  const timeoutMs = boundedInteger(params.timeout_ms || 30000, "timeout_ms", 50, 300000);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (kind === "inventory") {
      const name = itemName(params.item);
      const count = boundedInteger(params.count || 1, "count", 1, 4096);
      const total = current.inventory.items().filter(item => item.name === name).reduce((sum, item) => sum + item.count, 0);
      if (total >= count) return { matched: true, kind, item: name, count: total };
    } else if (kind === "entity") {
      const name = String(params.name || "").toLowerCase();
      const entity = current.nearestEntity(candidate => String(candidate.name || candidate.mobType || "").toLowerCase() === name);
      if (entity) return { matched: true, kind, entityId: entity.id };
    } else if (kind === "health") {
      const minimum = finiteNumber(params.minimum || 1, "minimum");
      if (current.health >= minimum) return { matched: true, kind, health: current.health };
    } else if (kind === "position") {
      const x = finiteNumber(params.x, "x");
      const y = finiteNumber(params.y, "y");
      const z = finiteNumber(params.z, "z");
      const range = finiteNumber(params.range || 1, "range");
      if (current.entity.position.distanceTo(new Vec3(x, y, z)) <= range) return { matched: true, kind };
    } else {
      throw new Error("Unknown wait_for kind");
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`wait_for timed out: ${kind}`);
}

async function openContainer(params) {
  const current = requireBot();
  const x = finiteNumber(params.x, "x");
  const y = finiteNumber(params.y, "y");
  const z = finiteNumber(params.z, "z");
  const block = current.blockAt(new Vec3(x, y, z));
  if (!block) throw new Error("Container block not loaded");
  const windowPromise = new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Container open timed out")), 10000);
    current.once("windowOpen", window => { clearTimeout(timeout); resolve(window); });
  });
  await current.activateBlock(block);
  const window = await windowPromise;
  return { id: window.id, title: String(window.title || ""), slots: window.slots.length };
}

async function clickSlot(params) {
  const current = requireBot();
  if (!current.currentWindow) throw new Error("No container is open");
  const slot = boundedInteger(params.slot, "slot", 0, current.currentWindow.slots.length - 1);
  const mouseButton = boundedInteger(params.mouse_button || 0, "mouse_button", 0, 2);
  const mode = boundedInteger(params.mode || 0, "mode", 0, 6);
  await current.clickWindow(slot, mouseButton, mode);
  return { clicked: slot, mouseButton, mode };
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
