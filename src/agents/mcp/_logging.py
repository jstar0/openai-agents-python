from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from .. import _debug

_URL_DERIVED_NAME_PREFIXES = ("sse: ", "streamable_http: ", "streamable-http: ")


class _MCPServerNameSource(Protocol):
    @property
    def name(self) -> str: ...


def get_mcp_server_log_name(name: str) -> str:
    """Remove URL credentials, query parameters, and fragments from MCP log names."""
    prefix = next(
        (candidate for candidate in _URL_DERIVED_NAME_PREFIXES if name.startswith(candidate)),
        "",
    )
    candidate = name[len(prefix) :] if prefix else name

    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return name

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return name

    host = parsed.netloc.rsplit("@", 1)[-1]
    sanitized = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return f"{prefix}{sanitized}"


def get_mcp_server_log_message(message: str, server: _MCPServerNameSource) -> str:
    """Build an MCP log message without reading the server name in redacted mode."""
    if _debug.DONT_LOG_TOOL_DATA:
        return message
    return f"{message} '{get_mcp_server_log_name(server.name)}'"
