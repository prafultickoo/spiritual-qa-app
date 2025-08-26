"""
Tests for centralized LLM timeout wrappers.
Uses a fake OpenAI-like client to avoid network calls.
"""

import asyncio
import time
import types
import pytest

from utils.llm_timeout import (
    call_chat_completion_with_timeout,
    call_chat_completion_with_async_timeout,
)


def make_fake_client(create_fn):
    """Create a minimal fake client with the attribute chain
    client.chat.completions.create mapping to the provided function.
    """
    completions = types.SimpleNamespace(create=create_fn)
    chat = types.SimpleNamespace(completions=completions)
    client = types.SimpleNamespace(chat=chat)
    return client


def make_fake_response(content: str = "ok"):
    """Construct a minimal response object with id, usage and choices/message.content."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(id="fake", usage=None, choices=[choice])


def test_sync_timeout_raises_timeouterror():
    """Slow fake client should cause the sync wrapper to raise TimeoutError."""
    def slow_create(**kwargs):
        time.sleep(0.3)
        return make_fake_response()

    client = make_fake_client(slow_create)
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 10,
    }

    with pytest.raises(TimeoutError):
        call_chat_completion_with_timeout(
            client=client,
            params=params,
            timeout_seconds=0.05,
            request_id="test-sync",
            logger=None,
        )


def test_sync_completes_under_timeout():
    """Fast fake client should complete within timeout and return a response object."""
    def fast_create(**kwargs):
        return make_fake_response("fast-ok")

    client = make_fake_client(fast_create)
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 10,
    }

    resp = call_chat_completion_with_timeout(
        client=client,
        params=params,
        timeout_seconds=1,
        request_id="test-sync-fast",
        logger=None,
    )
    assert getattr(resp, "id", None) == "fake"
    assert resp.choices[0].message.content == "fast-ok"


@pytest.mark.asyncio
async def test_async_timeout_raises_timeouterror():
    """Slow fake client should cause the async wrapper to raise TimeoutError."""
    def slow_create(**kwargs):
        time.sleep(0.3)
        return make_fake_response()

    client = make_fake_client(slow_create)
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 10,
    }

    with pytest.raises(TimeoutError):
        await call_chat_completion_with_async_timeout(
            client=client,
            params=params,
            timeout_seconds=0.05,
            request_id="test-async",
            logger=None,
        )


@pytest.mark.asyncio
async def test_async_completes_under_timeout():
    """Fast fake client should complete within timeout and return a response object."""
    def fast_create(**kwargs):
        return make_fake_response("fast-async-ok")

    client = make_fake_client(fast_create)
    params = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
        "max_tokens": 10,
    }

    resp = await call_chat_completion_with_async_timeout(
        client=client,
        params=params,
        timeout_seconds=1,
        request_id="test-async-fast",
        logger=None,
    )
    assert getattr(resp, "id", None) == "fake"
    assert resp.choices[0].message.content == "fast-async-ok"
