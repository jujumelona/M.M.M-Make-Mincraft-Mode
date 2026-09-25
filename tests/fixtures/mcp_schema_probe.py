"""Local stdio provider exercising the real MCP schema and invocation boundaries."""

import json
import sys
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    from mcp.server import MCPServer as FastMCP

server = FastMCP("schema-probe")
invocations = Path(sys.argv[1])


@server.tool()
def search(query: str, version: str) -> dict:
    with invocations.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"query": query, "version": version}) + "\n")
    if query == "provider failure":
        print("schema-probe child failure detail", file=sys.stderr, flush=True)
        raise RuntimeError("deliberate provider transport boundary failure")
    return {"hits": [{"path": "net/minecraft/Example.java", "text": query}],
            "version": version}


if __name__ == "__main__":
    server.run(transport="stdio")
