"""Timeouts and retries for calls to live sources.

Each attempt has its own timeout. Connector errors and timeouts are retried with a growing pause;
authorization errors and anything else are never retried, because trying again cannot fix them.
After the last attempt the failure surfaces as a `ConnectorError` with a safe message.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from packages.common.errors import ConnectorError

logger = logging.getLogger(__name__)


class RetryPolicy(BaseModel):
    attempts: int = Field(default=2, ge=1, le=5)
    timeout_seconds: float = Field(default=20.0, gt=0)
    backoff_seconds: float = Field(default=0.5, ge=0)


async def call_with_retry[T](
    call: Callable[[], Awaitable[T]], policy: RetryPolicy, label: str
) -> T:
    """Run `call` until it succeeds or the policy's attempts are used up."""
    for attempt in range(1, policy.attempts + 1):
        try:
            async with asyncio.timeout(policy.timeout_seconds):
                return await call()
        except TimeoutError as exc:
            failure: ConnectorError = ConnectorError(
                "The source did not answer in time.", {"call": label}
            )
            failure.__cause__ = exc
        except ConnectorError as exc:
            failure = exc
        if attempt < policy.attempts:
            logger.warning(
                "%s failed (attempt %d of %d): %s", label, attempt, policy.attempts, failure.message
            )
            await asyncio.sleep(policy.backoff_seconds * attempt)
    logger.error("%s failed after %d attempt(s): %s", label, policy.attempts, failure.message)
    raise failure
