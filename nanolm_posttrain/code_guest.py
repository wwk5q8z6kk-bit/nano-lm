"""Bounded, capability-free guest for generated Python and trusted verifier checks."""

import ast
from functools import lru_cache
import heapq
from pathlib import Path

from pydantic_monty import MemoryFile, Monty, MontyError, OSAccess


@lru_cache(maxsize=1)
def _heap_module_source() -> str:
    """Expose CPython's min-heap primitives to the guest as an in-memory module."""
    module = ast.parse(Path(heapq.__file__).read_text(encoding="utf-8"))
    names = {"heappush", "heappop", "heapify", "heappushpop", "heapreplace", "_siftup", "_siftdown"}
    module.body = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    return ast.unparse(module)


def run_guest(code: str, checks: tuple[str, ...] = (), *, collect_output: bool = False) -> tuple[bool, str]:
    if len(code.encode("utf-8")) > 64 * 1024:
        return False, "code_size_limit"
    output: list[str] = []
    remaining = 8192

    def receive(stream: str, message: str) -> None:
        nonlocal remaining
        if collect_output and stream == "stdout" and remaining:
            piece = message[:remaining]
            output.append(piece)
            remaining -= len(piece)

    limits = {
        "max_memory": 16 * 1024 * 1024,
        "max_feed_duration_secs": 1.0,
        "max_recursion_depth": 128,
        "max_total_sleep_secs": 0,
        "max_suspensions": 128,
    }
    # Only virtual files are exposed. No host mounts, network, environment or functions.
    guest_os = OSAccess(
        [MemoryFile("/heapq.py", _heap_module_source())], environ={},
        max_urandom_bytes=0, max_sleep=0, max_total_file_bytes=64 * 1024,
    )
    with Monty(max_processes=1, request_timeout=4) as pool, pool.checkout(limits=limits) as session:
        try:
            session.feed_run(code, os=guest_os, print_callback=receive)
            for check in checks:
                session.feed_run(check, os=guest_os, print_callback=lambda *_: None)
        except MontyError as exc:
            return False, str(exc)[:160]
    return True, "".join(output).strip()
