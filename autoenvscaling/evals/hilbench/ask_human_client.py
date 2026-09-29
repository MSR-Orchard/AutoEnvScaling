#!/usr/bin/env python3
"""ask_human: a shell command that asks the task's ask-human server one question.

HiL-Bench exposes ``ask_human`` as an MCP tool served by a sidecar next to the task container.
The Pi coding agent has no MCP client, so, as for HiL-Bench's own text-to-SQL tasks, the agent
gets a directly callable command instead::

    ask_human "Which invoice statuses are in scope for the reconciliation?"

The command sends the question to the same server over MCP's streamable HTTP transport and
prints the server's reply (the withheld detail, or "irrelevant question"). It uses only the
Python standard library so it can be copied into any task image as /usr/local/bin/ask_human.

Environment: ASK_HUMAN_URL (default http://ask-human:8000/mcp), ASK_HUMAN_TIMEOUT (seconds).
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "http://ask-human:8000/mcp"
PROTOCOL_VERSION = "2025-03-26"


def _post(url, payload, session_id=None, timeout=120.0, redirects=3):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (response.status, dict(response.headers.items()),
                    response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as err:
        # Servers mounted at "/mcp/" redirect "/mcp" with 307, which urllib does not follow
        # for POST requests.
        location = err.headers.get("Location") if err.headers else None
        if err.code in (307, 308) and location and redirects > 0:
            target = urllib.parse.urljoin(url, location)
            return _post(target, payload, session_id, timeout, redirects - 1)
        raise


def _messages(content_type, text):
    """JSON-RPC messages from a JSON body or a server-sent-events body."""
    if "text/event-stream" in (content_type or ""):
        out, data = [], []
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line.strip() and data:
                try:
                    out.append(json.loads("\n".join(data)))
                except ValueError:
                    pass
                data = []
        return out
    if not text.strip():
        return []
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


def _header(headers, name):
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _response(headers, text, request_id):
    for message in _messages(_header(headers, "Content-Type"), text):
        if isinstance(message, dict) and message.get("id") == request_id:
            if "error" in message:
                raise RuntimeError(f"MCP error: {message['error']}")
            return message.get("result") or {}
    raise RuntimeError(f"no response to request {request_id}")


def ask(question, url=DEFAULT_URL, timeout=120.0):
    """Send one question to the ask_human MCP tool and return the text of its reply."""
    _, headers, text = _post(url, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                   "clientInfo": {"name": "ask_human-cli", "version": "1"}},
    }, timeout=timeout)
    _response(headers, text, 1)
    session_id = _header(headers, "Mcp-Session-Id")
    _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id, timeout)
    _, headers, text = _post(url, {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "ask_human", "arguments": {"question": question}},
    }, session_id, timeout)
    result = _response(headers, text, 2)
    parts = [c.get("text", "") for c in result.get("content") or []
             if isinstance(c, dict) and c.get("type") == "text"]
    return "\n".join(parts).strip()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    question = " ".join(argv).strip()
    if not question:
        sys.stderr.write('usage: ask_human "<question>"\n')
        return 2
    url = os.environ.get("ASK_HUMAN_URL", DEFAULT_URL)
    timeout = float(os.environ.get("ASK_HUMAN_TIMEOUT", "120"))
    try:
        print(ask(question, url=url, timeout=timeout))
    except Exception as exc:  # noqa: BLE001  (the agent should see a readable failure, not a traceback)
        sys.stderr.write(f"ask_human failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
