import readline from "node:readline";
import mineflayer from "mineflayer";
import { pathfinder, Movements, goals } from "mineflayer-pathfinder";
import { Vec3 } from "vec3";

let bot = null;

function requireBot() {
  if (!bot) throw new Error("Mineflayer bot is not connected");
  return bot;
}

async function connect(params) {
  if (bot) throw new Error("Mineflayer bot is already connected");
  const host = String(params.host || "127.0.0.1");
  const port = Number(params.port || 25565);
  const username = String(params.username || "MMMTestBot");
  if (!["127.0.0.1", "localhost"].includes(host)) {
    throw new Error("The T4/local profile permits Mineflayer only on localhost");
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
    food: bot.food
  };
}

async function walkTo(params) {
  const current = requireBot();
  const x = Number(params.x);
  const y = Number(params.y);
  const z = Number(params.z);
  const range = Math.max(1, Math.min(4, Number(params.range || 1)));
  if (![x, y, z].every(Number.isFinite)) throw new Error("Invalid walk target");
  await current.pathfinder.goto(new goals.GoalNear(x, y, z, range));
  return status();
}

async function interactBlock(params) {
  const current = requireBot();
  const x = Number(params.x);
  const y = Number(params.y);
  const z = Number(params.z);
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
