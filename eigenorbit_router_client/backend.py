"""The kit provider that runs circuits through EigenOrbit Router."""

from __future__ import annotations

import os
import sys
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Sequence

import eigenkit
import numpy as np
from eigenkit.core.backends.qiskit_runtime_backend import QiskitRuntimeBackend
from eigenkit.core.ir import ALL_GATES, CircuitSpec, Observable
from eigenkit.core.provider import (
    Capabilities,
    CompiledCircuit,
    CostWarning,
    _batch_pair,
)
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

from . import _client
from ._client import POLL_DELAYS, TERMINAL_STATUSES, RouterHttpClient
from ._decode import decode_result
from .config import RouterConfig
from .errors import RouterError, refusal_hint
from .handle import RouterJobHandle


_BRAKET_UNSUPPORTED_GATES = frozenset(
    {"AmplitudeEncoding", "CPHASE", "ControlledQubitUnitary"}
)

ESTIMATE_KEYS = (
    "estimate_usd_micro",
    "price_version",
    "available_usd_micro",
    "would_reject_reason",
)
# The vendor/fee split of the estimate; both are None when the router does not
# send them.
BREAKDOWN_KEYS = ("vendor_cost_usd_micro", "fee_usd_micro")

_UNAVAILABLE_STATUSES = frozenset({502, 503, 504})

_INTERNAL_DIRS = tuple(
    os.path.dirname(os.path.abspath(path)) + os.sep
    for path in (eigenkit.__file__, __file__)
)

_BATCH_REFUSALS = frozenset(
    {
        "method_not_supported",
        "shots_invalid",
        "negative_balance",
        # IBM admission refusals: every row would get the same 403.
        "ibm_account_not_connected",
        "ibm_connection_invalid",
    }
)

_MICRO_PER_USD = Decimal(1_000_000)
_SIX_PLACES = Decimal("0.000001")

_NO_DEVICE_MESSAGE = (
    "provider='router' needs a device: pass provider=RouterConfig(device='braket/sv1') "
    "from eigenorbit_router_client (install eigenkit[router])"
)


def _usd(micro: int) -> str:
    """Integer micro-USD as an exact dollar string: 90000 -> "$0.090000"."""
    amount = (Decimal(micro) / _MICRO_PER_USD).quantize(_SIX_PLACES)
    sign = "-" if amount < 0 else ""
    return f"{sign}${format(abs(amount), 'f')}"


def _usd_short(micro: int) -> str:
    """Integer micro-USD as an exact dollar string with trailing zeros dropped
    (at least two decimals): 377400 -> "$0.3774", 90000 -> "$0.09"."""
    amount = (Decimal(micro) / _MICRO_PER_USD).quantize(_SIX_PLACES)
    sign = "-" if amount < 0 else ""
    whole, _, fraction = format(abs(amount), "f").partition(".")
    fraction = fraction.rstrip("0").ljust(2, "0")
    return f"{sign}${whole}.{fraction}"


def _caller_stacklevel() -> int:
    """The warning stacklevel of the first frame outside the kit and this package."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and os.path.abspath(
        frame.f_code.co_filename
    ).startswith(_INTERNAL_DIRS):
        frame = frame.f_back
        level += 1
    return level


def _gradient_method(config: RouterConfig) -> str:
    return "none" if config.is_qpu else "parameter_shift"


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _checked_estimate(estimate: dict[str, Any]) -> dict[str, Any]:
    """Refuse an EstimateOut whose money fields are not the contract's integers."""
    for key in ("estimate_usd_micro", "available_usd_micro"):
        if not _is_int(estimate.get(key)):
            raise RouterError(
                "invalid_response",
                f"the router's estimate has a non-integer {key}; amounts must be "
                f"integer micro-USD",
            )
    if not isinstance(estimate.get("price_version"), str):
        raise RouterError(
            "invalid_response", "the router's estimate has no price_version string"
        )
    reason = estimate.get("would_reject_reason")
    if reason is not None and not isinstance(reason, str):
        raise RouterError(
            "invalid_response", "the router's estimate has a non-string would_reject_reason"
        )
    parts = [estimate.get(key) for key in BREAKDOWN_KEYS]
    if parts != [None, None]:
        vendor, fee = parts
        if not (_is_int(vendor) and _is_int(fee) and vendor >= 0 and fee >= 0) or (
            vendor + fee != estimate["estimate_usd_micro"]
        ):
            raise RouterError(
                "invalid_response",
                "the router's estimate has a vendor_cost_usd_micro and fee_usd_micro "
                "that are not non-negative integers summing to estimate_usd_micro",
            )
    return estimate


def _has_breakdown(estimate: dict[str, Any]) -> bool:
    return all(_is_int(estimate.get(key)) for key in BREAKDOWN_KEYS)


def _estimate_unavailable(error: RouterError) -> bool:
    """Whether a failed compile-time estimate still lets the circuit compile."""
    if error.http_status in _UNAVAILABLE_STATUSES or error.code == "router_unreachable":
        return True
    return error.code == "unauthenticated" and error.http_status is None


def _gate_counts(circuit: CircuitSpec) -> dict[str, int]:
    counts: dict[str, int] = {}
    for gate in circuit.gates:
        counts[gate.name] = counts.get(gate.name, 0) + 1
    return counts


def _greedy_depth(circuit: CircuitSpec) -> int:
    """Greedy layer depth of the circuit."""
    level: dict[int, int] = {}
    depth = 0
    for gate in circuit.gates:
        layer = max((level.get(q, 0) for q in gate.qubits), default=0) + 1
        for q in gate.qubits:
            level[q] = layer
        depth = max(depth, layer)
    return depth


def _submitted_ids(jobs: list[dict[str, Any] | None]) -> list[str]:
    return [job["id"] for job in jobs if job is not None]


def _map_rows(pool, fn, indexes, jobs: list[dict[str, Any] | None]) -> list[Any]:
    """Map `fn` over `indexes`; a failure names every job of the batch."""
    try:
        return list(pool.map(fn, indexes))
    except RouterError as error:
        job_ids = _submitted_ids(jobs)
        raise RouterError(
            error.code,
            f"{error.message}. This happened while waiting for a batch of "
            f"{len(jobs)} router jobs, which keep running and are billed: "
            f"{', '.join(job_ids)}",
            {**error.details, "job_ids": job_ids},
            http_status=error.http_status,
            job_id=error.job_id,
        ) from None


class RouterBackend:
    """Kit provider for EigenOrbit Router, registered as `"router"`.

    Raises:
        ValueError: `config` is None, as for a bare `provider="router"`.
    """

    def __init__(self, config: RouterConfig | None = None, name: str = "router"):
        if config is None:
            raise ValueError(_NO_DEVICE_MESSAGE)
        self.name = name
        self.config = config

    @property
    def gradient_method(self) -> str:
        """`"none"` on a QPU, `"parameter_shift"` on a simulator."""
        return _gradient_method(self.config)

    @property
    def capabilities(self) -> Capabilities:
        """The vendor's gate set: Braket's on Braket devices, the kit IBM adapter's on
        IBM devices, every gate otherwise."""
        gates = ALL_GATES
        if self.config.vendor == "braket":
            gates = ALL_GATES - _BRAKET_UNSUPPORTED_GATES
        elif self.config.vendor == "ibm":
            # The same object the own-credentials IBM path validates with, so
            # the two gate sets never drift and a circuit that path refuses is
            # refused here too, before any router call.
            gates = QiskitRuntimeBackend().capabilities.supported_gates
        return Capabilities(
            supported_gates=gates,
            supports_batching=False,
            supports_shots=True,
            max_qubits=None,
        )

    def compile(
        self, circuit: CircuitSpec, observables: Sequence[Observable]
    ) -> RouterCompiled:
        """Prepare `circuit` for repeated evaluation of `observables`."""
        return RouterCompiled(circuit, observables, self.config)


class RouterCompiled(CompiledCircuit):
    """A circuit compiled for the router; every call is one paid router job.

    Raises:
        RouterError: The compile-time estimate was refused (for example 401, 426 or 422).
    """

    submit_methods: frozenset[str] = frozenset(
        {"run", "sample", "probabilities", "zero_state_probability"}
    )

    def __init__(
        self,
        circuit: CircuitSpec,
        observables: Sequence[Observable],
        config: RouterConfig,
    ):
        self._config = config
        self._circuit_json = circuit_spec_to_json(circuit)
        self._observables_json = observables_to_json(list(observables))
        self.n_qubits = circuit.n_qubits
        self.n_weights = circuit.n_weights
        self.n_inputs = circuit.n_inputs
        self.n_observables = len(self._observables_json)
        self._gate_counts = _gate_counts(circuit)
        self._depth = _greedy_depth(circuit)
        self._http = RouterHttpClient(config)
        self._estimate: dict[str, Any] | None = None
        self._estimate_error: RouterError | None = None
        try:
            self._estimate = self._fetch_estimate("run")
        except RouterError as error:
            if not _estimate_unavailable(error):
                self._http.close()
                raise
            self._estimate_error = error
        self._emit_cost_warning()

    @property
    def estimate(self) -> dict[str, Any] | None:
        """The EstimateOut for one `run` job from compile, or None if unavailable."""
        return None if self._estimate is None else dict(self._estimate)

    def _fetch_estimate(self, method: str) -> dict[str, Any]:
        body = self.build_request(method, np.zeros(self.n_weights), np.zeros(self.n_inputs))
        return _checked_estimate(self._http.estimate(body))

    def _emit_cost_warning(self) -> None:
        """Issue this compile's one CostWarning."""
        if self._config.vendor == "ibm":
            parts = self._ibm_cost_parts()
        else:
            parts = self._balance_cost_parts()
        parts.append("Use compiled.dry_run_info() to inspect before submitting.")
        warnings.warn(" ".join(parts), CostWarning, stacklevel=_caller_stacklevel())

    def _ibm_cost_parts(self) -> list[str]:
        """The CostWarning of an IBM device: no router charge, IBM bills quantum time."""
        device = self._config.device
        parts = [
            f"Router jobs on {device} carry no router charge: IBM bills the IBM "
            f"Quantum instance the job runs on for the quantum time it uses.",
        ]
        if self._estimate is None:
            code = self._estimate_error.code if self._estimate_error else "unknown"
            parts.append(
                f"The router could not be asked at compile ({code}), so whether it "
                f"would accept this job is unknown."
            )
        elif self._estimate.get("would_reject_reason"):
            parts.append(
                f"The router would reject this job now: "
                f"{self._estimate['would_reject_reason']}."
            )
        parts.append(
            "On this device each row of a 2-D batch is a separate IBM job that uses "
            "quantum time; RouterJobHandle.ibm_usage_seconds shows how much."
        )
        return parts

    def _balance_cost_parts(self) -> list[str]:
        """The CostWarning of a device billed to the EigenOrbit balance."""
        device = self._config.device
        if self._estimate is None:
            code = self._estimate_error.code if self._estimate_error else "unknown"
            parts = [
                f"Router jobs on {device} are billed to your EigenOrbit balance, "
                f"but the cost estimate is unavailable ({code}): the router could not be "
                f"asked at compile.",
                "Every submit is still gated by the router's hold on your balance, so "
                "no job runs that the balance cannot cover.",
            ]
        else:
            estimate = self._estimate
            micro = estimate["estimate_usd_micro"]
            if _has_breakdown(estimate):
                price = (
                    f"estimated {_usd_short(micro)} (vendor "
                    f"{_usd_short(estimate['vendor_cost_usd_micro'])} + EigenOrbit "
                    f"fee {_usd_short(estimate['fee_usd_micro'])}) per job"
                )
            else:
                price = f"{_usd(micro)} per job"
            parts = [
                f"Router jobs on {device} are billed to your EigenOrbit balance: "
                f"{price} (estimate_usd_micro={micro}, price_version "
                f"{estimate['price_version']}).",
                f"Available balance: {_usd(estimate['available_usd_micro'])}.",
            ]
            if estimate.get("would_reject_reason"):
                parts.append(
                    f"The router would reject this job now: "
                    f"{estimate['would_reject_reason']}."
                )
        if _gradient_method(self._config) == "parameter_shift":
            parts.append(
                "Gradients use parameter-shift here: each gradient evaluation and "
                "each row of a 2-D batch is a separate paid router job."
            )
        else:
            parts.append(
                "On this device each row of a 2-D batch is a separate paid router job."
            )
        return parts

    def dry_run_info(self) -> dict[str, Any]:
        """The IBM adapter's dry-run keys plus the router estimate; sends nothing.

        `vendor_cost_usd_micro` and `fee_usd_micro` are added when the router's
        estimate carries them.
        """
        estimate = self._estimate or {}
        breakdown = (
            {key: estimate[key] for key in BREAKDOWN_KEYS}
            if _has_breakdown(estimate)
            else {}
        )
        return {
            "backend_name": self._config.device,
            "is_fake": False,
            "primitive": None,
            "shots": self._config.shots,
            "optimization_level": None,
            "seed_transpiler": None,
            "n_logical_qubits": self.n_qubits,
            "n_physical_qubits": None,
            "transpiled_gate_counts": dict(self._gate_counts),
            "transpiled_depth": self._depth,
            "n_observables": self.n_observables,
            "n_measurement_groups": None,
            "n_circuit_runs_per_call": 1,
            **{key: estimate.get(key) for key in ESTIMATE_KEYS},
            **breakdown,
        }

    def build_request(self, method: str, weights, inputs) -> dict[str, Any]:
        """The contract `JobRequest` for one call."""
        return {
            "device": self._config.device,
            "method": method,
            "shots": self._config.shots,
            "circuit_spec": self._circuit_json,
            "observables": self._observables_json,
            "weights": np.asarray(weights, dtype=np.float64).tolist(),
            "inputs": np.asarray(inputs, dtype=np.float64).tolist(),
        }

    def run(self, weights, inputs) -> np.ndarray:
        """Evaluate the observables: `(n_obs,)`, or `(B, n_obs)` for a 2-D batch."""
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if _batch_pair(weights, inputs)[0] is None:
            return self._run_one(weights, inputs)
        return self._fan_out("run", weights, inputs)

    def zero_state_probabilities(self, weights, inputs) -> np.ndarray:
        """All-zeros probability for each row of a batch, `(B,)` float64."""
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if _batch_pair(weights, inputs)[0] is None:
            raise ValueError(
                f"zero_state_probabilities expects a batch of shape (B, n_inputs); "
                f"got inputs of shape {inputs.shape}. Use zero_state_probability() "
                f"for one sample."
            )
        return self._fan_out("zero_state_probability", weights, inputs)

    def _fan_out(self, method: str, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Run one router job per batch row and stack the results in row order."""
        n, weights_2d, inputs_2d = _batch_pair(weights, inputs)
        self._check_batch_shapes(weights, inputs)
        if n == 0:
            shape = (0, self.n_observables) if method == "run" else (0,)
            return np.empty(shape, dtype=np.float64)
        rows = [
            (weights[i] if weights_2d else weights, inputs[i] if inputs_2d else inputs)
            for i in range(n)
        ]
        self._check_batch_affordable(method, n)
        # Each row's Idempotency-Key "<call_id>:<row>" is reused on retries.
        call_id = str(uuid.uuid4())

        local = threading.local()
        opened: list[RouterHttpClient] = []
        opened_lock = threading.Lock()

        def http() -> RouterHttpClient:
            client = getattr(local, "http", None)
            if client is None:
                client = RouterHttpClient(self._config)
                local.http = client
                with opened_lock:
                    opened.append(client)
            return client

        jobs: list[dict[str, Any] | None] = [None] * n
        refused: list[RouterError | None] = [None] * n

        def submit_row(i: int) -> None:
            body = self.build_request(method, *rows[i])
            try:
                jobs[i] = http().submit(body, idempotency_key=f"{call_id}:{i}")
            except RouterError as error:
                refused[i] = error

        def poll_row(i: int) -> None:
            jobs[i] = http().get_job(jobs[i]["id"])

        def result_row(i: int) -> np.ndarray:
            job_id = jobs[i]["id"]
            value = decode_result(
                http().get_result(job_id),
                method=method,
                n_qubits=self.n_qubits,
                n_observables=self.n_observables,
                job_id=job_id,
            )
            return np.asarray(value, dtype=np.float64)

        pool = ThreadPoolExecutor(
            max_workers=min(self._config.max_concurrent_submits, n),
            thread_name_prefix="eo-router-batch",
        )
        try:
            list(pool.map(submit_row, range(n)))
            self._wait_for_rows(pool, poll_row, jobs)
            self._raise_for_failed_rows(jobs, refused)
            values = _map_rows(pool, result_row, range(n), jobs)
        except KeyboardInterrupt:
            # Never cancel on Ctrl-C: name every submitted job, which keeps running.
            pool.shutdown(wait=True, cancel_futures=True)
            job_ids = _submitted_ids(jobs)
            interrupt = KeyboardInterrupt(
                f"interrupted during a batch of {n} router jobs. Submitted jobs keep "
                f"running and are billed (no auto-cancel): {', '.join(job_ids) or 'none'}"
            )
            interrupt.job_ids = job_ids
            raise interrupt from None
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for client in opened:
                client.close()
        return np.stack(values)

    def _check_batch_shapes(self, weights: np.ndarray, inputs: np.ndarray) -> None:
        for name, array, length in (
            ("weights", weights, self.n_weights),
            ("inputs", inputs, self.n_inputs),
        ):
            if array.ndim not in (1, 2) or array.shape[-1] != length:
                raise ValueError(
                    f"this circuit takes {length} {name} per row; got {name} of shape "
                    f"{array.shape}"
                )

    def _check_batch_affordable(self, method: str, n: int) -> None:
        """Refuse a batch the estimate says would be rejected or is unaffordable."""
        try:
            estimate = self._fetch_estimate(method)
        except RouterError as error:
            if _estimate_unavailable(error):
                return
            raise
        reason = estimate.get("would_reject_reason")
        if reason in _BATCH_REFUSALS:
            message = (
                f"the router would reject every {method} job of this batch: {reason}. "
                f"Nothing was submitted."
            )
            hint = refusal_hint(reason, {"device": self._config.device})
            if hint:
                message = f"{message} {hint}"
            raise RouterError(
                reason, message, {"would_reject_reason": reason, "n_jobs": n}
            )
        # Exact integer micro-USD, never floats.
        per_job = estimate["estimate_usd_micro"]
        available = estimate["available_usd_micro"]
        required = n * per_job
        if required > available:
            raise RouterError(
                "insufficient_balance",
                f"this batch of {n} router jobs needs about {_usd(required)} "
                f"({n} x {_usd(per_job)}), but the available balance is "
                f"{_usd(available)}. Nothing was submitted. This check is best "
                f"effort; the router's hold on each job is the real gate.",
                {
                    "required_usd_micro": required,
                    "available_usd_micro": available,
                    "n_jobs": n,
                },
            )

    def _wait_for_rows(self, pool, poll_row, jobs: list[dict[str, Any] | None]) -> None:
        """Poll the submitted rows until every one is terminal."""
        timeout = self._config.poll_timeout_seconds
        deadline = _client._monotonic() + timeout
        polls = 0
        pending = [
            i
            for i, job in enumerate(jobs)
            if job is not None and job.get("status") not in TERMINAL_STATUSES
        ]
        while pending:
            _map_rows(pool, poll_row, pending, jobs)
            pending = [i for i in pending if jobs[i].get("status") not in TERMINAL_STATUSES]
            if not pending:
                return
            remaining = deadline - _client._monotonic()
            if remaining <= 0:
                job_ids = [jobs[i]["id"] for i in pending]
                raise RouterError(
                    "poll_timeout",
                    f"{len(pending)} of {len(jobs)} router jobs in this batch are still "
                    f"not finished after poll_timeout_seconds={timeout:g}: "
                    f"{', '.join(job_ids)}. They keep running and are billed; there is "
                    f"no auto-cancel.",
                    {"pending": job_ids, "poll_timeout_seconds": timeout},
                )
            _client._sleep(min(POLL_DELAYS[min(polls, len(POLL_DELAYS) - 1)], remaining))
            polls += 1

    @staticmethod
    def _raise_for_failed_rows(
        jobs: list[dict[str, Any] | None], refused: list[RouterError | None]
    ) -> None:
        """Raise once for every row without a result, in row order."""
        failed = []
        first: RouterError | None = None
        for row, (job, error) in enumerate(zip(jobs, refused)):
            if error is not None:
                job_id = error.job_id or error.details.get("job_id")
                entry = {"row": row, "job_id": job_id, "code": error.code, "status": None}
            elif job is not None and job.get("status") != "completed":
                error = RouterError.from_job(job)
                entry = {
                    "row": row,
                    "job_id": job.get("id"),
                    "code": error.code,
                    "status": job.get("status"),
                }
            else:
                continue
            failed.append(entry)
            if first is None:
                first = error
        if first is None:
            return
        listed = "; ".join(
            f"row {f['row']}: job {f['job_id'] or '(not created)'} {f['code']}"
            for f in failed
        )
        raise RouterError(
            first.code,
            f"{len(failed)} of {len(jobs)} router jobs in this batch failed ({listed}). "
            f"Completed rows are billed; no partial result is returned and nothing "
            f"was cancelled.",
            {"failed": failed, "n_jobs": len(jobs)},
            http_status=first.http_status,
            job_id=failed[0]["job_id"],
        )

    def _run_one(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return self.submit(weights, inputs, method="run").result()

    def submit(
        self,
        weights,
        inputs,
        *,
        method: str = "run",
        idempotency_key: str | None = None,
    ) -> RouterJobHandle:
        """Submit one router job and return its handle without waiting.

        Args:
            idempotency_key: None mints a fresh key, so every call is a new job.
                Pass the `details["idempotency_key"]` of a `router_unreachable`
                error to resubmit the same call: the router replays the job the
                lost submit created instead of creating (and billing) a second.

        Raises:
            ValueError: `method` is not in `submit_methods`, or an argument is not 1-D.
        """
        if method not in self.submit_methods:
            raise ValueError(
                f"method {method!r} cannot be submitted to the router; "
                f"submit_methods = {sorted(self.submit_methods)}"
            )
        weights = np.asarray(weights, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        if weights.ndim != 1 or inputs.ndim != 1:
            raise ValueError(
                f"submit() takes one 1-D weights vector and one 1-D inputs vector; "
                f"got weights of shape {weights.shape} and inputs of shape "
                f"{inputs.shape}. Submit a batch one row at a time."
            )
        body = self.build_request(method, weights, inputs)
        if idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        job = self._http.submit(body, idempotency_key=idempotency_key)
        return RouterJobHandle(
            self._http,
            job,
            method=method,
            n_qubits=self.n_qubits,
            n_observables=self.n_observables,
            poll_timeout_seconds=self._config.poll_timeout_seconds,
        )

    def zero_state_probability(self, weights, inputs) -> float:
        """Probability of the all-zeros state, from one router job."""
        return float(self.submit(weights, inputs, method="zero_state_probability").result())

    def probabilities(self, weights, inputs) -> np.ndarray:
        """Basis-state distribution (qubit-0-MSB), from one router job."""
        return self.submit(weights, inputs, method="probabilities").result()

    def sample(self, weights, inputs) -> dict[str, int]:
        """Measurement counts `{bitstring: count}` (qubit-0-MSB), from one router job."""
        return dict(self.submit(weights, inputs, method="sample").result())

    def close(self) -> None:
        """Close the HTTP connections this handle opened."""
        self._http.close()
