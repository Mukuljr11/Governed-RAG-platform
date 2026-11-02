import asyncio
import json
from typing import Any

from mcp import ClientSession
from datetime import timedelta
from app.config import settings

try:
    from mcp.client.streamable_http import streamablehttp_client as _connect
except ImportError:  # newer SDK versions renamed it
    from mcp.client.streamable_http import streamable_http_client as _connect


async def _call(name: str, arguments: dict[str, Any]):
    async with _connect(
        settings.mcp_url,
        timeout=timedelta(seconds=600),
        sse_read_timeout=timedelta(seconds=600),
    ) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=600)
        ) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)

def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(_call(name, arguments))
    if result.isError:
        raise RuntimeError(f"MCP tool '{name}' failed: {result.content}")
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured
    return json.loads(result.content[0].text)