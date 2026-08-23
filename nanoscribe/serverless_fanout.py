"""Async RunPod Serverless fan-out via native /run (non-blocking)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from nanoscribe.serverless_inference import (
    _resolve_api_key,
    endpoint_native_urls,
    parse_endpoint_id,
)

CONTRACT_VERSION = "nano.fanout.v1"
SERVERLESS_RATE_PER_HR = 1.39
DEFAULT_AVG_EXEC_S = 45.0
TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
TRANSIENT_STATUSES = frozenset({"FAILED", "TIMED_OUT"})


@dataclass(frozen=True, slots=True)
class FanoutJobSpec:
    case_id: str
    experiment_id: str
    contract_version: str
    mode: str  # span_port | structured
    atom_id: str | None
    openai_input: dict[str, Any]


@dataclass
class FanoutJobRecord:
    case_id: str
    experiment_id: str
    contract_version: str
    mode: str
    atom_id: str | None
    request_id: str | None = None
    submission_time: float | None = None
    start_time: float | None = None
    completion_time: float | None = None
    queue_delay_ms: int | None = None
    execution_time_ms: int | None = None
    retry_count: int = 0
    status: str = "pending"
    error: str | None = None
    response: Any = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.submission_time is not None:
            payload["submission_time_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.submission_time)
            )
        if self.completion_time is not None:
            payload["completion_time_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.completion_time)
            )
        return payload


def extract_openai_content(output: Any) -> str:
    """Normalize RunPod /run output to assistant text."""
    if output is None:
        return ""
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, dict):
        choices = output.get("choices")
        if choices:
            message = choices[0].get("message") or {}
            return str(message.get("content") or "").strip()
        nested = output.get("openai_output")
        if isinstance(nested, dict):
            return extract_openai_content(nested)
    return str(output).strip()


def estimate_worker_cost_usd(
    num_jobs: int,
    *,
    avg_exec_s: float = DEFAULT_AVG_EXEC_S,
    rate_per_hr: float = SERVERLESS_RATE_PER_HR,
) -> float:
    worker_seconds = num_jobs * avg_exec_s
    return round((worker_seconds / 3600.0) * rate_per_hr, 4)


def build_openai_run_payload(
    openai_input: Mapping[str, Any],
    *,
    case_id: str,
    experiment_id: str,
    contract_version: str,
    mode: str,
    atom_id: str | None,
) -> dict[str, Any]:
    return {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": dict(openai_input),
            "campaign_meta": {
                "case_id": case_id,
                "experiment_id": experiment_id,
                "contract_version": contract_version,
                "mode": mode,
                "atom_id": atom_id,
            },
        }
    }


class ServerlessFanoutRunner:
    """Bounded-concurrency async fan-out against RunPod /run + /status."""

    def __init__(
        self,
        endpoint_id: str,
        *,
        start_concurrency: int = 32,
        max_concurrency: int = 64,
        max_retries: int = 2,
        poll_interval_s: float = 0.5,
        job_timeout_s: float = 300.0,
    ) -> None:
        self.endpoint_id = parse_endpoint_id(endpoint_id)
        self.start_concurrency = start_concurrency
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.poll_interval_s = poll_interval_s
        self.job_timeout_s = job_timeout_s
        self._api_key = _resolve_api_key(None)
        self._urls = endpoint_native_urls(self.endpoint_id)
        self._concurrency = start_concurrency
        self._records: list[FanoutJobRecord] = []

    @property
    def records(self) -> list[FanoutJobRecord]:
        return list(self._records)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _submit(
        self,
        client: httpx.AsyncClient,
        spec: FanoutJobSpec,
    ) -> tuple[str, float]:
        payload = build_openai_run_payload(
            spec.openai_input,
            case_id=spec.case_id,
            experiment_id=spec.experiment_id,
            contract_version=spec.contract_version,
            mode=spec.mode,
            atom_id=spec.atom_id,
        )
        submission_time = time.time()
        response = await client.post(
            self._urls["run"],
            headers=self._headers(),
            json=payload,
        )
        if response.status_code in TRANSIENT_HTTP_CODES:
            raise httpx.HTTPStatusError(
                f"transient submit {response.status_code}",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        data = response.json()
        request_id = str(data.get("id") or "")
        if not request_id:
            raise RuntimeError(f"RunPod /run missing id: {data}")
        return request_id, submission_time

    async def _poll(
        self,
        client: httpx.AsyncClient,
        request_id: str,
        *,
        submission_time: float,
    ) -> FanoutJobRecord:
        status_url = f"{self._urls['run'].rsplit('/', 1)[0]}/status/{request_id}"
        deadline = time.time() + self.job_timeout_s
        start_time: float | None = None
        while time.time() < deadline:
            response = await client.get(status_url, headers=self._headers())
            if response.status_code in TRANSIENT_HTTP_CODES:
                await asyncio.sleep(self.poll_interval_s)
                continue
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status") or "")
            if status == "IN_PROGRESS" and start_time is None:
                start_time = time.time()
            if status == "COMPLETED":
                completion_time = time.time()
                return FanoutJobRecord(
                    case_id="",
                    experiment_id="",
                    contract_version="",
                    mode="",
                    atom_id=None,
                    request_id=request_id,
                    submission_time=submission_time,
                    start_time=start_time or submission_time,
                    completion_time=completion_time,
                    queue_delay_ms=data.get("delayTime"),
                    execution_time_ms=data.get("executionTime"),
                    status="completed",
                    response=extract_openai_content(data.get("output")),
                )
            if status in TRANSIENT_STATUSES:
                raise RuntimeError(
                    f"transient job status {status}: {data.get('error')}"
                )
            if status in {"FAILED", "CANCELLED"}:
                completion_time = time.time()
                return FanoutJobRecord(
                    case_id="",
                    experiment_id="",
                    contract_version="",
                    mode="",
                    atom_id=None,
                    request_id=request_id,
                    submission_time=submission_time,
                    start_time=start_time or submission_time,
                    completion_time=completion_time,
                    queue_delay_ms=data.get("delayTime"),
                    execution_time_ms=data.get("executionTime"),
                    status="failed",
                    error=str(data.get("error") or status),
                    response=None,
                )
            await asyncio.sleep(self.poll_interval_s)
        raise TimeoutError(f"job {request_id} timed out after {self.job_timeout_s}s")

    async def _run_one(
        self,
        client: httpx.AsyncClient,
        spec: FanoutJobSpec,
    ) -> FanoutJobRecord:
        record = FanoutJobRecord(
            case_id=spec.case_id,
            experiment_id=spec.experiment_id,
            contract_version=spec.contract_version,
            mode=spec.mode,
            atom_id=spec.atom_id,
        )
        for attempt in range(self.max_retries + 1):
            try:
                request_id, submission_time = await self._submit(client, spec)
                record.request_id = request_id
                record.submission_time = submission_time
                polled = await self._poll(
                    client,
                    request_id,
                    submission_time=submission_time,
                )
                record.start_time = polled.start_time
                record.completion_time = polled.completion_time
                record.queue_delay_ms = polled.queue_delay_ms
                record.execution_time_ms = polled.execution_time_ms
                record.status = polled.status
                record.error = polled.error
                record.response = polled.response
                record.retry_count = attempt
                return record
            except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
                record.retry_count = attempt
                if attempt >= self.max_retries:
                    record.status = "failed"
                    record.error = f"{type(exc).__name__}: {exc}"
                    record.completion_time = time.time()
                    return record
                await asyncio.sleep(0.5 * (attempt + 1))
        record.status = "failed"
        record.error = "exhausted retries"
        record.completion_time = time.time()
        return record

    def _maybe_scale_concurrency(self, completed: Sequence[FanoutJobRecord]) -> None:
        if self._concurrency >= self.max_concurrency or len(completed) < 10:
            return
        failures = sum(1 for item in completed if item.status != "completed")
        failure_rate = failures / len(completed)
        delays = [item.queue_delay_ms for item in completed if item.queue_delay_ms is not None]
        avg_delay = (sum(delays) / len(delays)) if delays else 0.0
        if failure_rate <= 0.05 and avg_delay <= 5000:
            self._concurrency = self.max_concurrency

    async def run_jobs(
        self,
        specs: Sequence[FanoutJobSpec],
        *,
        on_checkpoint: Callable[[list[FanoutJobRecord]], Awaitable[None] | None] | None = None,
    ) -> list[FanoutJobRecord]:
        if not specs:
            return []
        concurrency_limit = self._concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        results: list[FanoutJobRecord | None] = [None] * len(specs)
        scaled = False

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0)) as client:
            async def worker(index: int, spec: FanoutJobSpec) -> None:
                nonlocal concurrency_limit, semaphore, scaled
                async with semaphore:
                    record = await self._run_one(client, spec)
                    results[index] = record
                    done = [item for item in results if item is not None]
                    if not scaled:
                        self._maybe_scale_concurrency(done)
                        if self._concurrency > concurrency_limit:
                            concurrency_limit = self._concurrency
                            semaphore = asyncio.Semaphore(concurrency_limit)
                            scaled = True
                    if on_checkpoint and len(done) % 8 == 0:
                        maybe = on_checkpoint(done)
                        if asyncio.iscoroutine(maybe):
                            await maybe

            await asyncio.gather(*(worker(i, spec) for i, spec in enumerate(specs)))

        self._records = [item for item in results if item is not None]
        if on_checkpoint:
            maybe = on_checkpoint(self._records)
            if asyncio.iscoroutine(maybe):
                await maybe
        return self._records

    async def fetch_health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(self._urls["health"], headers=self._headers())
            response.raise_for_status()
            return response.json()
