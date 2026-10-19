"""A contract-faithful, in-process double of the router's `/v1/jobs*` API.

It runs on `httpx.MockTransport`, so no socket is opened, and follows contract
v1: the evaluation order of section 1, the key format of section 2, the
idempotency rules of section 3, the JobOut and JobResultOut shapes of
sections 5, 6 and 8 and the error envelope of section 9. Results are computed
with the kit's numpy reference simulator, which is the ground truth the
router's own answers are compared with.

Knobs are plain attributes, so a test can change them after construction.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from eigenkit.core.backends.numpy_backend import NumpySimulator
from eigenkit.core.wire import circuit_spec_from_json, observables_from_json


VALID_KEY = "eo_test_" + "A" * 43
ALL_METHODS = frozenset({"run", "sample", "probabilities", "zero_state_probability"})
MAX_SHOTS = 100_000
# Methods whose shot range starts at 1 on every device (contract section 4).
METHOD_MIN_SHOTS = {"zero_state_probability": 1, "sample": 1}
REQUEST_FIELDS = frozenset(
    {"device", "method", "shots", "circuit_spec", "observables", "weights", "inputs"}
)
SETTLED_COST_USD_MICRO = 4500
# The vendor part of the settled cost: braket/sv1@1 at 3000 ms (contract 5.5).
SETTLED_VENDOR_COST_USD_MICRO = 3750
TERMINAL = frozenset({"completed", "failed", "cancelled", "rejected"})
# Job error code -> category (contract section 9.2).
ERROR_CATEGORIES = {
    "insufficient_balance": "rejected",
    "negative_balance": "rejected",
    "vendor_rejected": "vendor",
    "vendor_failed": "vendor",
    "vendor_unavailable": "vendor",
    "internal_error": "internal",
    "ledger_unavailable": "internal",
    "ibm_credentials_rejected": "vendor",
    "ibm_quota_exhausted": "vendor",
    "ibm_target_mismatch": "vendor",
    "ibm_device_not_on_instance": "vendor",
}
# IBM devices: these methods only, every shot range starts at 1.
IBM_METHODS = frozenset({"run", "sample", "probabilities"})
# The router's 403 admission refusals of an IBM job (contract section 9.1), with
# the messages of the contract examples.
IBM_REFUSAL_MESSAGES = {
    "ibm_account_not_connected": (
        "IBM jobs need an IBM Quantum account. Connect yours under IBM account in the "
        "router web app."
    ),
    "ibm_connection_invalid": (
        "Your IBM account connection stopped working. Reconnect it under IBM account "
        "in the router web app."
    ),
}

_BEARER = re.compile(r"Bearer eo_test_[A-Za-z0-9_-]{43}")
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_JOB_ROUTE = re.compile(r"/v1/jobs/(?P<job_id>[^/]+)(?P<suffix>/result|/cancel)?")


def _is_ibm(device: str) -> bool:
    return device.startswith("ibm/")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error(status: int, code: str, message: str, details: dict | None = None):
    return httpx.Response(
        status,
        json={"error": {"code": code, "message": message, "details": details or {}}},
        headers={"EO-Protocol-Version": "1"},
    )


def _ok(status: int, body: dict, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, json=body, headers={"EO-Protocol-Version": "1", **(headers or {})}
    )


def _validation_error(loc: list, msg: str) -> httpx.Response:
    return _error(
        422,
        "validation_error",
        "Request validation failed",
        {"errors": [{"loc": loc, "msg": msg, "type": "value_error"}]},
    )


def upgrade_message(protocol_min: int, protocol_max: int) -> str:
    """The router's 426 message (contract v1)."""
    return (
        f"This router speaks EO-Protocol-Version {protocol_min}..{protocol_max}; "
        f"the request's version is missing or unsupported. "
        f"Upgrade eigenorbit-router-client."
    )


def largest_remainder_counts(probabilities: np.ndarray, shots: int, n_qubits: int):
    """Deterministic counts: floor(p * shots), then the leftover shots go to
    the largest fractional parts (lowest index first on ties). Keys are
    qubit0_msb bitstrings; zero counts are left out."""
    raw = np.asarray(probabilities, dtype=np.float64) * shots
    floors = np.floor(raw).astype(np.int64)
    leftover = int(shots - floors.sum())
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:leftover]:
        floors[i] += 1
    return {
        format(i, f"0{n_qubits}b"): int(count)
        for i, count in enumerate(floors)
        if count > 0
    }


class FakeRouter:
    """In-process router double. Use `transport` as an httpx transport.

    Args:
        status_steps: Statuses successive GET /v1/jobs/{id} calls report; the
            last one repeats, so a tuple with no terminal status never ends.
        fail_with: When set, new jobs end `failed` with this error code
            instead of `completed`, and their hold is released.
        complete_billing: The billing value of completed jobs (`settled` or
            `held`).
        devices: The device catalog; anything else is `unknown_device`.
        supported_methods: Methods submit accepts; anything else is
            `method_not_supported` (Braket: run, zero_state_probability).
        estimate_usd_micro: What the estimate and the hold cost.
        available_usd_micro: The balance the estimate reports.
        would_reject_reason: When set, the estimate reports this reason
            instead of the one it derives from the request and the balance.
        protocol_min, protocol_max: The EO-Protocol-Version range accepted;
            anything else answers 426 with the contract's upgrade message.
        completion_order: "fifo" walks every job through `status_steps` at
            the same pace; "reverse" holds each batch row back by (highest
            row of its call - its row) polls, so the last row finishes first.
            A batch row is a job whose Idempotency-Key ends in ":<row>".
        fail_rows: Batch rows (by that key suffix) whose jobs end `failed`
            with the given error code, overriding `fail_with`.
        submit_delay_seconds: A real sleep inside every POST /v1/jobs, taken
            outside the double's lock, so concurrent submits overlap and
            `max_in_flight` can observe them.
        estimate_vendor_cost_usd_micro: The vendor part of the estimate; the
            fee is the rest. None derives it for a 20% markup (5/6 of the
            estimate: 90000 = 75000 + 15000). `estimate_breakdown=False`
            leaves both fields out, as a router that sends no vendor/fee split
            does.
        fail_category: The category of failed jobs' errors; None maps the
            code with the contract's table (section 9.2).
        device_status: Device -> vendor status (`offline`, `retired`); a
            device not listed is online. Submit answers 422
            `device_unavailable`, the estimate reports it first.
        stale_devices: Devices whose cached status is too old to trust:
            `device_unavailable` with `stale: true`.
        concurrency_limit: The per-user cap of jobs in flight (not terminal),
            or None for no cap. A submit at the cap answers 429
            `concurrency_limit_exceeded`.
        cancel_mode: How POST /v1/jobs/{id}/cancel treats a job that is not
            terminal (contract section 6): "immediate" (not at the vendor
            yet: the 202 body says cancelled), "vendor" (the 202 body keeps
            the status and the next GET says cancelled) or "vendor_finishes"
            (the vendor runs the task to the end: the job completes).
        ibm_refusal: For an `ibm/` device, `ibm_account_not_connected` or
            `ibm_connection_invalid`: submit answers that 403 and the estimate
            reports it as its would_reject_reason. An IBM
            job is free: estimate 0, billing null, no costs, available 0.
        credential_source: Whose IBM account runs an IBM job: `own` (the
            user's) or `organization` (an IBM account EigenOrbit provides); a
            Braket job reports null.
        ibm_usage_seconds: The IBM quantum seconds a completed IBM job reports
            (null before it completes, and for a Braket job).
        ibm_job_fields: False leaves `credential_source` and
            `ibm_usage_seconds` out of JobOut, as a router that does not send
            them does.

    Attributes:
        max_in_flight: The most POST /v1/jobs handled at the same time.
        terminal_order: Job IDs in the order they reached a terminal status.
    """

    def __init__(
        self,
        *,
        status_steps: tuple[str, ...] = ("queued", "running", "completed"),
        fail_with: str | None = None,
        complete_billing: str = "settled",
        devices: frozenset[str] = frozenset(
            {"braket/sv1", "braket/dm1", "braket/ionq_forte", "ibm/ibm_fez"}
        ),
        supported_methods: frozenset[str] = ALL_METHODS,
        estimate_usd_micro: int = 90_000,
        available_usd_micro: int = 910_000,
        would_reject_reason: str | None = None,
        protocol_min: int = 1,
        protocol_max: int = 1,
        completion_order: str = "fifo",
        fail_rows: dict[int, str] | None = None,
        submit_delay_seconds: float = 0.0,
        estimate_vendor_cost_usd_micro: int | None = None,
        estimate_breakdown: bool = True,
        fail_category: str | None = None,
        device_status: dict[str, str] | None = None,
        stale_devices: frozenset[str] = frozenset(),
        concurrency_limit: int | None = None,
        cancel_mode: str = "immediate",
        ibm_refusal: str | None = None,
        credential_source: str = "organization",
        ibm_usage_seconds: float | None = 2.315,
        ibm_job_fields: bool = True,
    ):
        self.status_steps = status_steps
        self.fail_with = fail_with
        self.complete_billing = complete_billing
        self.devices = frozenset(devices)
        self.supported_methods = frozenset(supported_methods)
        self.estimate_usd_micro = estimate_usd_micro
        self.available_usd_micro = available_usd_micro
        self.would_reject_reason = would_reject_reason
        self.protocol_min = protocol_min
        self.protocol_max = protocol_max
        self.completion_order = completion_order
        self.fail_rows = dict(fail_rows or {})
        self.submit_delay_seconds = submit_delay_seconds
        self.estimate_vendor_cost_usd_micro = estimate_vendor_cost_usd_micro
        self.estimate_breakdown = estimate_breakdown
        self.fail_category = fail_category
        self.device_status = dict(device_status or {})
        self.stale_devices = frozenset(stale_devices)
        self.concurrency_limit = concurrency_limit
        self.cancel_mode = cancel_mode
        self.ibm_refusal = ibm_refusal
        self.credential_source = credential_source
        self.ibm_usage_seconds = ibm_usage_seconds
        self.ibm_job_fields = ibm_job_fields
        self.max_in_flight = 0
        self.terminal_order: list[str] = []
        self._in_flight = 0
        self._flight_lock = threading.Lock()
        self.requests: list[httpx.Request] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, tuple[dict, str]] = {}
        self._injections: list[dict[str, Any]] = []
        self._drop_creates = 0
        self._lock = threading.Lock()
        self.transport = httpx.MockTransport(self.handle)

    # ---- entry point ----

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or request.url.path != "/v1/jobs":
            return self._handle_locked(request)
        with self._flight_lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.submit_delay_seconds:
                time.sleep(self.submit_delay_seconds)
            return self._handle_locked(request)
        finally:
            with self._flight_lock:
                self._in_flight -= 1

    def _handle_locked(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            injected = self._injected(request)
            if injected is not None:
                return injected
            response = self._dispatch(request)
            if (
                self._drop_creates > 0
                and request.method == "POST"
                and request.url.path == "/v1/jobs"
                and response.status_code in (200, 202)
            ):
                # The job now exists (or was replayed), but the client never
                # sees the answer.
                self._drop_creates -= 1
                raise httpx.ReadError("connection lost after create", request=request)
            return response

    # ---- fault injection ----

    def fail_next(
        self,
        method: str,
        path_prefix: str,
        *,
        status: int | None = None,
        body: dict | str | bytes | None = None,
        exc: type[httpx.RequestError] | None = None,
        times: int = 1,
        key_suffix: str | None = None,
    ) -> None:
        """Answer the next `times` matching requests with `status` and `body`,
        or raise `exc` (an httpx request error class) for them instead.

        A request matches when its method is `method` and its path starts with
        `path_prefix`. The estimate is its own endpoint: POST
        /v1/jobs/estimate matches only a prefix that names it, so a failure
        aimed at submits ("/v1/jobs") never hits the compile-time estimate.
        With `key_suffix`, only requests whose Idempotency-Key ends with it
        match (":1" picks batch row 1). The request is still recorded in
        `requests`.
        """
        if (status is None) == (exc is None):
            raise ValueError("give exactly one of status or exc")
        self._injections.append(
            {
                "method": method,
                "prefix": path_prefix,
                "status": status,
                "body": body,
                "exc": exc,
                "left": times,
                "key_suffix": key_suffix,
            }
        )

    def drop_response_after_create(self, *, times: int = 1) -> None:
        """Create the next submitted job normally, then lose the response:
        the client gets httpx.ReadError while the job exists. With `times`
        above 1 the answers to the replays of that key are lost too."""
        self._drop_creates = times

    def _injected(self, request: httpx.Request) -> httpx.Response | None:
        for injection in self._injections:
            if injection["left"] <= 0:
                continue
            if request.method != injection["method"]:
                continue
            if not request.url.path.startswith(injection["prefix"]):
                continue
            if request.url.path == "/v1/jobs/estimate" and not injection[
                "prefix"
            ].startswith("/v1/jobs/estimate"):
                continue
            suffix = injection["key_suffix"]
            if suffix is not None and not request.headers.get(
                "Idempotency-Key", ""
            ).endswith(suffix):
                continue
            injection["left"] -= 1
            if injection["exc"] is not None:
                raise injection["exc"]("injected transport failure", request=request)
            body = injection["body"]
            if isinstance(body, dict):
                return httpx.Response(
                    injection["status"], json=body, headers={"EO-Protocol-Version": "1"}
                )
            content = body.encode() if isinstance(body, str) else (body or b"")
            return httpx.Response(injection["status"], content=content)
        return None

    def _dispatch(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not path.startswith("/v1/jobs"):
            return _error(404, "not_found", "Not found")
        rejected = self._check_protocol(request) or self._check_key(request)
        if rejected is not None:
            return rejected
        if path == "/v1/jobs/estimate":
            return self._only(request, "POST") or self._estimate(request)
        if path == "/v1/jobs":
            return self._only(request, "POST") or self._submit(request)
        match = _JOB_ROUTE.fullmatch(path)
        if match is None:
            return _error(404, "not_found", "Not found")
        job_id, suffix = match["job_id"], match["suffix"]
        if suffix == "/cancel":
            return self._only(request, "POST") or self._cancel(job_id)
        if suffix == "/result":
            return self._only(request, "GET") or self._result(job_id)
        return self._only(request, "GET") or self._get_job(job_id)

    @staticmethod
    def _only(request: httpx.Request, method: str) -> httpx.Response | None:
        if request.method != method:
            return _error(405, "method_not_allowed", "Method not allowed")
        return None

    # ---- contract section 1 and 2 checks ----

    def _check_protocol(self, request: httpx.Request) -> httpx.Response | None:
        raw = request.headers.get("EO-Protocol-Version")
        client_version = int(raw) if raw is not None and raw.isdigit() else None
        if client_version is not None and (
            self.protocol_min <= client_version <= self.protocol_max
        ):
            return None
        return _error(
            426,
            "protocol_unsupported",
            upgrade_message(self.protocol_min, self.protocol_max),
            {
                "client_version": client_version,
                "supported_min": self.protocol_min,
                "supported_max": self.protocol_max,
                "min_client_version": None,
            },
        )

    @staticmethod
    def _check_key(request: httpx.Request) -> httpx.Response | None:
        if _BEARER.fullmatch(request.headers.get("Authorization", "")):
            return None
        return _error(401, "unauthenticated", "Invalid or missing API key")

    # ---- request parsing and admission ----

    @staticmethod
    def _parse_body(request: httpx.Request) -> tuple[dict | None, httpx.Response | None]:
        try:
            body = json.loads(request.content)
        except ValueError:
            return None, _validation_error(["body"], "JSON decode error")
        if not isinstance(body, dict):
            return None, _validation_error(["body"], "Input should be an object")
        unknown = sorted(set(body) - REQUEST_FIELDS)
        if unknown:
            return None, _validation_error(
                ["body", "<redacted>"], "Extra inputs are not permitted"
            )
        missing = sorted(REQUEST_FIELDS - set(body))
        if missing:
            return None, _validation_error(["body", missing[0]], "Field required")
        if body["method"] not in ALL_METHODS:
            return None, _validation_error(["body", "method"], "Input should be a method")
        shots = body["shots"]
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 0:
            return None, _validation_error(["body", "shots"], "Input should be >= 0")
        if not isinstance(body["observables"], list) or not body["observables"]:
            return None, _validation_error(["body", "observables"], "At least 1 item")
        return body, None

    def _device_unavailable(self, device: str) -> dict | None:
        """The details of `device_unavailable` for `device`, or None when online."""
        if device in self.stale_devices:
            return {
                "device": device,
                "status": self.device_status.get(device, "online"),
                "stale": True,
            }
        status = self.device_status.get(device, "online")
        if status != "online":
            return {"device": device, "status": status}
        return None

    def _in_flight_jobs(self) -> int:
        return sum(1 for job in self.jobs.values() if job["status"] not in TERMINAL)

    def _admit(
        self, body: dict, *, submit: bool = False
    ) -> tuple[Any, httpx.Response | None]:
        """Device and job checks shared by estimate and submit (section 1, step 4)."""
        device = body["device"]
        if device not in self.devices:
            return None, _error(
                422, "unknown_device", f"Unknown device {device}", {"device": device}
            )
        unavailable = self._device_unavailable(device)
        if submit and unavailable is not None:
            return None, _error(
                422,
                "device_unavailable",
                f"{device} is {unavailable['status']} at the vendor; try again later",
                unavailable,
            )
        if body["shots"] > MAX_SHOTS:
            return None, _error(
                422,
                "shots_invalid",
                "shots outside the device range",
                {"min_shots": 0, "max_shots": MAX_SHOTS},
            )
        weights = np.asarray(body["weights"], dtype=np.float64)
        inputs = np.asarray(body["inputs"], dtype=np.float64)
        if weights.ndim > 1 or inputs.ndim > 1:
            return None, _error(
                422,
                "batch_not_supported",
                "Braket devices run one parameter set per task",
            )
        try:
            spec = circuit_spec_from_json(body["circuit_spec"])
            observables = observables_from_json(body["observables"])
        except Exception as exc:  # any decode failure is circuit_invalid
            return None, _error(
                422,
                "circuit_invalid",
                "circuit_spec or observables cannot be decoded",
                {"reason": type(exc).__name__},
            )
        if weights.shape != (spec.n_weights,) or inputs.shape != (spec.n_inputs,):
            return None, _error(
                422,
                "circuit_invalid",
                "weights or inputs do not match the circuit",
                {"reason": "parameter_count"},
            )
        return (spec, observables, weights, inputs), None

    def _method_rejection(self, body: dict) -> tuple[str, dict] | None:
        method = body["method"]
        ibm = _is_ibm(body["device"])
        supported = self.supported_methods & IBM_METHODS if ibm else self.supported_methods
        if method not in supported:
            return "method_not_supported", {
                "method": method,
                "supported": sorted(supported),
            }
        min_shots = 1 if ibm else METHOD_MIN_SHOTS.get(method, 0)
        if body["shots"] < min_shots:
            return "shots_invalid", {"min_shots": min_shots, "max_shots": MAX_SHOTS}
        return None

    def _ibm_refusal(self, device: str) -> httpx.Response | None:
        """The 403 admission refusal of an IBM job (contract section 1, step 6)."""
        if not _is_ibm(device) or self.ibm_refusal is None:
            return None
        not_connected = self.ibm_refusal == "ibm_account_not_connected"
        details = {"device": device} if not_connected else {}
        return _error(
            403, self.ibm_refusal, IBM_REFUSAL_MESSAGES[self.ibm_refusal], details
        )

    # ---- endpoints ----

    def _estimate(self, request: httpx.Request) -> httpx.Response:
        body, rejected = self._parse_body(request)
        if rejected is not None:
            return rejected
        _, rejected = self._admit(body)
        if rejected is not None:
            return rejected
        reason = self._method_rejection(body)
        would_reject = reason[0] if reason else None
        ibm = _is_ibm(body["device"])
        if self._device_unavailable(body["device"]) is not None:
            would_reject = "device_unavailable"
        if would_reject is None and ibm and self.ibm_refusal is not None:
            would_reject = self.ibm_refusal
        if (
            would_reject is None
            and self.concurrency_limit is not None
            and self._in_flight_jobs() >= self.concurrency_limit
        ):
            would_reject = "concurrency_limit_exceeded"
        if (
            would_reject is None
            and not ibm
            and self.available_usd_micro < self.estimate_usd_micro
        ):
            would_reject = "insufficient_balance"
        if self.would_reject_reason is not None:
            would_reject = self.would_reject_reason
        out = {
            "device": body["device"],
            "method": body["method"],
            "shots": body["shots"],
            "estimate_usd_micro": 0 if ibm else self.estimate_usd_micro,
            "price_version": f"{body['device']}@1",
            # The router asks no balance for an IBM estimate (contract section 10).
            "available_usd_micro": 0 if ibm else self.available_usd_micro,
            "would_reject_reason": would_reject,
        }
        if ibm:
            out["vendor_cost_usd_micro"] = 0
            out["fee_usd_micro"] = 0
        elif self.estimate_breakdown:
            vendor = self.estimate_vendor_cost_usd_micro
            if vendor is None:
                vendor = self.estimate_usd_micro * 5 // 6
            out["vendor_cost_usd_micro"] = vendor
            out["fee_usd_micro"] = self.estimate_usd_micro - vendor
        return _ok(200, out)

    def _submit(self, request: httpx.Request) -> httpx.Response:
        key = request.headers.get("Idempotency-Key")
        if key is None:
            return _error(400, "idempotency_key_required", "Idempotency-Key is required")
        if not _IDEMPOTENCY_KEY.fullmatch(key):
            return _validation_error(["header", "Idempotency-Key"], "Invalid key")
        body, rejected = self._parse_body(request)
        if rejected is not None:
            return rejected
        if key in self._idempotency:
            first_body, job_id = self._idempotency[key]
            if first_body != body:
                return _error(
                    409,
                    "idempotency_conflict",
                    "Idempotency-Key reused with another body",
                )
            return _ok(200, self._job_out(job_id), {"Idempotent-Replayed": "true"})
        admitted, rejected = self._admit(body, submit=True)
        if rejected is not None:
            return rejected
        reason = self._method_rejection(body)
        if reason is not None:
            code, details = reason
            return _error(422, code, f"{code} for this device", details)
        refusal = self._ibm_refusal(body["device"])
        if refusal is not None:
            return refusal
        if self.concurrency_limit is not None:
            in_flight = self._in_flight_jobs()
            if in_flight >= self.concurrency_limit:
                return _error(
                    429,
                    "concurrency_limit_exceeded",
                    f"You have {in_flight} jobs in flight; the limit is "
                    f"{self.concurrency_limit}. Wait for one to finish or cancel one.",
                    {"limit": self.concurrency_limit, "in_flight": in_flight},
                )
        job_id = str(uuid.uuid4())
        created = _now()
        call, _, row = key.rpartition(":")
        self.jobs[job_id] = {
            "call": call if call and row.isdigit() else None,
            "row": int(row) if call and row.isdigit() else None,
            "request": body,
            "result": self._compute(body["method"], body["shots"], *admitted),
            "status": "queued",
            "polls": 0,
            "created_at": created,
            "updated_at": created,
            "completed_at": None,
            "cancel": None,
        }
        self._idempotency[key] = (body, job_id)
        return _ok(202, self._job_out(job_id))

    def _get_job(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if job is None:
            return _error(404, "job_not_found", "Job not found")
        if job["status"] == "cancelled":
            return _ok(200, self._job_out(job_id))
        if job["cancel"] == "vendor":
            # The vendor cancelled the task; the next GET reports it.
            self._set_status(job_id, "cancelled")
            return _ok(200, self._job_out(job_id))
        steps = list(self.status_steps)
        if self._fail_code(job) is not None and steps and steps[-1] == "completed":
            steps[-1] = "failed"
        index = max(0, job["polls"] - self._completion_delay(job))
        status = steps[min(index, len(steps) - 1)]
        job["polls"] += 1
        if status != job["status"]:
            self._set_status(job_id, status)
        return _ok(200, self._job_out(job_id))

    def _set_status(self, job_id: str, status: str) -> None:
        job = self.jobs[job_id]
        job["status"] = status
        job["updated_at"] = _now()
        if status in TERMINAL:
            job["completed_at"] = job["updated_at"]
            self.terminal_order.append(job_id)

    def _cancel(self, job_id: str) -> httpx.Response:
        """POST /v1/jobs/{job_id}/cancel (contract section 6)."""
        job = self.jobs.get(job_id)
        if job is None:
            return _error(404, "job_not_found", "Job not found")
        status = job["status"]
        if status in ("completed", "failed", "rejected"):
            return _error(
                409,
                "job_not_cancellable",
                f"The job is {status}; only a queued or running job can be cancelled",
                {"status": status},
            )
        if status != "cancelled":
            if self.cancel_mode == "immediate":
                self._set_status(job_id, "cancelled")
            elif self.cancel_mode == "vendor":
                job["cancel"] = "vendor"
            # "vendor_finishes": the task runs to the end and completes.
        return _ok(202, self._job_out(job_id))

    def _result(self, job_id: str) -> httpx.Response:
        job = self.jobs.get(job_id)
        if job is None:
            return _error(404, "job_not_found", "Job not found")
        if job["status"] != "completed":
            return _error(
                409,
                "job_not_completed",
                "The job has not completed",
                {"status": job["status"]},
            )
        return _ok(
            200,
            {
                "job_id": job_id,
                "method": job["request"]["method"],
                "bit_order": "qubit0_msb",
                "result": job["result"],
            },
        )

    # ---- per-job behaviour ----

    def _fail_code(self, job: dict[str, Any]) -> str | None:
        if job["row"] is not None and job["row"] in self.fail_rows:
            return self.fail_rows[job["row"]]
        return self.fail_with

    def _completion_delay(self, job: dict[str, Any]) -> int:
        if self.completion_order != "reverse" or job["row"] is None:
            return 0
        last = max(
            other["row"] for other in self.jobs.values() if other["call"] == job["call"]
        )
        return last - job["row"]

    # ---- JobOut and results ----

    def _job_out(self, job_id: str) -> dict[str, Any]:
        job = self.jobs[job_id]
        request = job["request"]
        status = job["status"]
        ibm = _is_ibm(request["device"])
        billing, cost, vendor, error = "held", None, None, None
        if ibm:
            billing = None  # a free IBM job: no hold and no costs, for its whole life
        elif status == "completed":
            billing = self.complete_billing
            if billing == "settled":
                cost, vendor = SETTLED_COST_USD_MICRO, SETTLED_VENDOR_COST_USD_MICRO
        elif status in ("failed", "cancelled"):
            billing, cost, vendor = "released", 0, 0
        if status == "failed":
            code = self._fail_code(job)
            category = self.fail_category or ERROR_CATEGORIES.get(code, "internal")
            error = {
                "code": code,
                "message": f"The job failed: {code}",
                "category": category,
            }
        out = {
            "id": job_id,
            "status": status,
            "billing": billing,
            "device": request["device"],
            "method": request["method"],
            "shots": request["shots"],
            "estimate_usd_micro": 0 if ibm else self.estimate_usd_micro,
            "price_version": f"{request['device']}@1",
            "error": error,
            "actual_cost_usd_micro": cost,
            "vendor_cost_usd_micro": vendor,
            "fee_usd_micro": None if cost is None else cost - vendor,
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "completed_at": job["completed_at"],
        }
        if self.ibm_job_fields:
            out["credential_source"] = self.credential_source if ibm else None
            out["ibm_usage_seconds"] = (
                self.ibm_usage_seconds if ibm and status == "completed" else None
            )
        return out

    @staticmethod
    def _compute(method, shots, spec, observables, weights, inputs) -> dict[str, Any]:
        compiled = NumpySimulator().compile(spec, observables)
        if method == "run":
            values = np.asarray(compiled.run(weights, inputs), dtype=np.float64)
        elif method == "zero_state_probability":
            values = np.asarray(
                compiled.zero_state_probability(weights, inputs), dtype=np.float64
            )
        else:
            probabilities = np.asarray(
                compiled.probabilities(weights, inputs), dtype=np.float64
            )
            if method == "sample":
                return {
                    "kind": "counts",
                    "n_qubits": spec.n_qubits,
                    "data": largest_remainder_counts(probabilities, shots, spec.n_qubits),
                }
            values = probabilities
        return {
            "kind": "array",
            "dtype": "float64",
            "shape": list(values.shape),
            "data": values.tolist(),
        }
