# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Namash Aggarwal (https://github.com/namashworks/paralingo)
"""MCP server exposing the paralingo tools to any MCP-capable voice agent
(Claude, agent frameworks, IDEs...).

    pip install "mcp>=1.0"
    python -m paralingo.mcp_server --root ./profiles --audio-root ./recordings --agent-id my-agent

Claude Code / Desktop config example::

    {"mcpServers": {"paralingo": {
        "command": "python",
        "args": ["-m", "paralingo.mcp_server", "--root", "C:/data/profiles", "--audio-root", "C:/data/audio"]}}}
"""
from __future__ import annotations

import argparse
import asyncio

from .agent_tools import SpeechProfileTools


def main() -> None:
    ap = argparse.ArgumentParser(description="Speech profile MCP server (stdio)")
    ap.add_argument("--root", required=True, help="directory holding *.speech.json files")
    ap.add_argument("--audio-root", default=None, help="only WAV files under this folder may be analysed")
    ap.add_argument("--agent-id", default="mcp-agent")
    ap.add_argument("--no-pipeline-tools", action="store_true", help="expose only read/observe tools")
    args = ap.parse_args()

    try:
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as e:  # pragma: no cover
        raise SystemExit("The 'mcp' package is required: pip install mcp") from e

    tools = SpeechProfileTools(args.root, agent_id=args.agent_id, audio_root=args.audio_root,
                               include_pipeline_tools=not args.no_pipeline_tools)
    server = Server("paralingo")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(**t) for t in tools.definitions("mcp")]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        out = await asyncio.to_thread(tools.execute, name, arguments or {})
        return [types.TextContent(type="text", text=out)]

    async def run() -> None:
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
