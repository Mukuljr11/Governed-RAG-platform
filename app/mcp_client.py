import asyncio
import json
from typing import Any
import time
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
        timeout=timedelta(seconds=120),
        sse_read_timeout=timedelta(seconds=120),
    ) as (read, write, _):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=120)
        ) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)

class ToolError(RuntimeError):
    """The tool ran and reported a failure. Not retried."""


def call_tool(
    name: str, arguments: dict[str, Any], retries: int = 2, backoff: float = 1.0
) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            result = asyncio.run(_call(name, arguments))
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * 2**attempt)  # waits 1s, then 2s, then 4s...

    if result.isError:
        raise ToolError(f"MCP tool '{name}' failed: {result.content}")
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured
    return json.loads(result.content[0].text)