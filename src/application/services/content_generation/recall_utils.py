"""内容生成：并发召回与限流工具。"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def gather_with_concurrency(
    items: Iterable[T],
    *,
    worker: Callable[[T], Awaitable[R]],
    concurrency: int = 6,
    on_progress: Callable[[int, int], Any] | None = None,
    progress_every: int = 10,
) -> list[R]:
    seq = list(items)
    if not seq:
        return []

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _guarded(it: T) -> R:
        async with sem:
            return await worker(it)

    tasks = [asyncio.create_task(_guarded(it)) for it in seq]
    out: list[R] = []
    done = 0
    total = len(tasks)
    for fut in asyncio.as_completed(tasks):
        out.append(await fut)
        done += 1
        if on_progress is not None:
            if done == total or (progress_every > 0 and (done % int(progress_every)) == 0):
                r = on_progress(done, total)
                if inspect.isawaitable(r):
                    await r  # type: ignore[misc]
    return out
