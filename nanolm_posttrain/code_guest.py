"""Bounded, capability-free guest for generated Python and trusted verifier checks."""

from pydantic_monty import Monty, MontyError


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
        "max_suspensions": 0,
    }
    # No mounts, OS handler, external functions, or inherited Python objects.
    with Monty(max_processes=1, request_timeout=4) as pool, pool.checkout(limits=limits) as session:
        try:
            session.feed_run(code, print_callback=receive)
            for check in checks:
                session.feed_run(check, print_callback=lambda *_: None)
        except MontyError as exc:
            return False, str(exc)[:160]
    return True, "".join(output).strip()
