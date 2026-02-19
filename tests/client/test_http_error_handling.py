"""Tests for HTTP error handling in StreamableHTTP client.

Validates that HTTP error status codes are properly converted to JSONRPCError
responses and delivered to the read stream, instead of being silently swallowed.
"""

from __future__ import annotations as _annotations

import anyio
import httpx
import pytest

from mcp.client.streamable_http import RequestContext, StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import (
    INTERNAL_ERROR,
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
)


def make_mock_client(status_code: int, body: str = "") -> httpx.AsyncClient:
    """Create an httpx.AsyncClient that always returns the given status code."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_http_500_error_returns_jsonrpc_error_for_request():
    """When a POST for a JSONRPCRequest gets HTTP 500, a JSONRPCError should be
    delivered to the read stream so the caller doesn't hang."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    request_msg = JSONRPCRequest(jsonrpc="2.0", id=42, method="tools/list")
    session_msg = SessionMessage(request_msg)

    client = make_mock_client(500)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        await transport._handle_post_request(ctx)

        # Should have received a JSONRPCError on the read stream
        received = read_stream.receive_nowait()
        assert isinstance(received, SessionMessage)
        assert isinstance(received.message, JSONRPCError)
        assert received.message.id == 42
        assert received.message.error.code == INTERNAL_ERROR
        assert "500" in received.message.error.message
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()


@pytest.mark.anyio
async def test_http_502_error_returns_jsonrpc_error_for_request():
    """HTTP 502 Bad Gateway should also be converted to a JSONRPCError."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    request_msg = JSONRPCRequest(jsonrpc="2.0", id="req-1", method="resources/read", params={"uri": "test://resource"})
    session_msg = SessionMessage(request_msg)

    client = make_mock_client(502)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        await transport._handle_post_request(ctx)

        received = read_stream.receive_nowait()
        assert isinstance(received, SessionMessage)
        assert isinstance(received.message, JSONRPCError)
        assert received.message.id == "req-1"
        assert received.message.error.code == INTERNAL_ERROR
        assert "502" in received.message.error.message
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()


@pytest.mark.anyio
async def test_http_503_error_returns_jsonrpc_error_for_request():
    """HTTP 503 Service Unavailable should also be converted to a JSONRPCError."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    request_msg = JSONRPCRequest(jsonrpc="2.0", id=99, method="ping")
    session_msg = SessionMessage(request_msg)

    client = make_mock_client(503)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        await transport._handle_post_request(ctx)

        received = read_stream.receive_nowait()
        assert isinstance(received, SessionMessage)
        assert isinstance(received.message, JSONRPCError)
        assert received.message.id == 99
        assert received.message.error.code == INTERNAL_ERROR
        assert "503" in received.message.error.message
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()


@pytest.mark.anyio
async def test_http_error_for_notification_does_not_send_to_stream():
    """When a POST for a JSONRPCNotification gets HTTP 500, no JSONRPCError
    should be sent to the read stream (notifications don't expect responses)."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    notification_msg = JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")
    session_msg = SessionMessage(notification_msg)

    client = make_mock_client(500)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        # Should not raise
        await transport._handle_post_request(ctx)

        # Should NOT have written anything to the stream
        assert write_stream.statistics().current_buffer_used == 0
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()


@pytest.mark.anyio
async def test_http_202_accepted_does_not_error():
    """HTTP 202 Accepted should be handled normally (no error)."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    notification_msg = JSONRPCNotification(jsonrpc="2.0", method="notifications/cancelled")
    session_msg = SessionMessage(notification_msg)

    client = make_mock_client(202)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        await transport._handle_post_request(ctx)
        # No error should be on the stream
        assert write_stream.statistics().current_buffer_used == 0
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()


@pytest.mark.anyio
async def test_http_404_sends_session_terminated_error():
    """HTTP 404 should send a 'Session terminated' error for requests (existing behavior)."""
    transport = StreamableHTTPTransport(url="http://localhost:9999/mcp")
    write_stream, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](1)

    request_msg = JSONRPCRequest(jsonrpc="2.0", id=7, method="tools/list")
    session_msg = SessionMessage(request_msg)

    client = make_mock_client(404)
    ctx = RequestContext(
        client=client,
        session_id=None,
        session_message=session_msg,
        metadata=None,
        read_stream_writer=write_stream,
    )

    try:
        await transport._handle_post_request(ctx)

        received = read_stream.receive_nowait()
        assert isinstance(received, SessionMessage)
        assert isinstance(received.message, JSONRPCError)
        assert received.message.id == 7
        assert "Session terminated" in received.message.error.message
    finally:
        await write_stream.aclose()
        await read_stream.aclose()
        await client.aclose()
