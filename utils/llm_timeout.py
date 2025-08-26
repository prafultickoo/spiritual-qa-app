"""
LLM timeout and logging utilities.

Provides a safe, reusable wrapper to call OpenAI chat completions with a hard timeout
so backend requests never hang indefinitely.
"""
from typing import Any, Dict, Optional
import logging
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


def get_llm_timeout_default() -> int:
    """Return default LLM timeout seconds from env (LLM_REQUEST_TIMEOUT) or 30.

    Returns:
        int: Timeout in seconds.
    """
    try:
        return int(os.getenv("LLM_REQUEST_TIMEOUT", "30"))
    except Exception:
        return 30


async def call_chat_completion_with_async_timeout(
    client: Any,
    params: Dict[str, Any],
    timeout_seconds: int,
    request_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """Async-friendly timeout wrapper for OpenAI chat.completions.create.

    Uses asyncio.to_thread to avoid blocking the event loop and asyncio.wait_for
    to enforce a hard timeout.

    Args:
        client: OpenAI client instance.
        params: Dict of parameters for chat.completions.create.
        timeout_seconds: Maximum time to wait for the API call.
        request_id: Optional request correlation ID for logs.
        logger: Optional logger to emit diagnostic messages.

    Returns:
        OpenAI API response object.

    Raises:
        TimeoutError: If the request does not complete within timeout_seconds.
        Exception: Any exception raised by the API call is propagated.
    """
    start = time.time()
    rid = request_id or "n/a"

    if logger:
        safe_keys = list(params.keys())
        model_name = params.get("model", "unknown")
        logger.info(
            f"[LLM:{rid}] (async) Dispatch chat.completions.create model={model_name} "
            f"timeout={timeout_seconds}s keys={safe_keys}"
        )

    async def _invoke_async():
        return await asyncio.to_thread(lambda: client.chat.completions.create(**params))

    try:
        resp = await asyncio.wait_for(_invoke_async(), timeout=timeout_seconds)
        if logger:
            duration = time.time() - start
            resp_id = getattr(resp, "id", "n/a")
            usage = getattr(resp, "usage", None)
            logger.info(
                f"[LLM:{rid}] (async) Completed in {duration:.2f}s id={resp_id} usage={usage}"
            )
        return resp
    except asyncio.TimeoutError as exc:
        msg = f"LLM request timed out after {timeout_seconds}s (request_id={rid})"
        if logger:
            logger.error(f"[LLM:{rid}] (async) {msg}")
        raise TimeoutError(msg) from exc
    except Exception as e:
        if logger:
            logger.error(f"[LLM:{rid}] (async) LLM request failed: {type(e).__name__}: {e}")
        raise

def call_chat_completion_with_timeout(
    client: Any,
    params: Dict[str, Any],
    timeout_seconds: int,
    request_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    """Call OpenAI chat.completions.create with a hard timeout and detailed logging.

    This runs the synchronous API call in a separate thread and waits up to
    `timeout_seconds`. If it doesn't finish in time, a TimeoutError is raised.

    Args:
        client: OpenAI client instance.
        params: Dict of parameters for chat.completions.create.
        timeout_seconds: Maximum time to wait for the API call.
        request_id: Optional request correlation ID for logs.
        logger: Optional logger to emit diagnostic messages.

    Returns:
        OpenAI API response object.

    Raises:
        TimeoutError: If the request does not complete within timeout_seconds.
        Exception: Any exception raised by the API call is propagated.
    """
    start = time.time()
    rid = request_id or "n/a"

    if logger:
        safe_keys = list(params.keys())
        model_name = params.get("model", "unknown")
        logger.info(
            f"[LLM:{rid}] Dispatch chat.completions.create model={model_name} "
            f"timeout={timeout_seconds}s keys={safe_keys}"
        )

    def _invoke():
        return client.chat.completions.create(**params)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_invoke)
        try:
            resp = future.result(timeout=timeout_seconds)
            if logger:
                duration = time.time() - start
                resp_id = getattr(resp, "id", "n/a")
                usage = getattr(resp, "usage", None)
                logger.info(
                    f"[LLM:{rid}] Completed in {duration:.2f}s id={resp_id} usage={usage}"
                )
            return resp
        except FuturesTimeout as exc:
            # Attempt to cancel to avoid dangling thread (cancellation may fail if already running)
            future.cancel()
            msg = f"LLM request timed out after {timeout_seconds}s (request_id={rid})"
            if logger:
                logger.error(f"[LLM:{rid}] {msg}")
            raise TimeoutError(msg) from exc
        except Exception as e:
            if logger:
                logger.error(f"[LLM:{rid}] LLM request failed: {type(e).__name__}: {e}")
            raise
