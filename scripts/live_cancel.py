"""Live cancel E2E: submit one long job through the router and cancel it.

Compiles a long circuit with `RouterConfig(device="braket/sv1", shots=100)`,
submits it, waits until `status()` is `running`, calls `handle.cancel()` and
polls the job to a terminal status. It then prints the job ID, the final
status and the router's billing for the job (GET /v1/jobs/{id}).

A cancel only asks: the router cancels a job that has not reached the vendor
at once, and asks the vendor to cancel one that has. Amazon Braket's
simulators accept the request but run a started task to the end, so on SV1
the job usually ends `completed` and is charged like any completed job
(contract section 6). Both endings pass:

- `cancelled`: the vendor (or the router) cancelled it; billed what the vendor
  billed plus the markup, often nothing;
- `completed`: the vendor finished the task first.

A QPU cancel is optional: `--device braket/iqm_garnet --shots 10` submits a
small circuit and cancels it while it waits in the vendor queue (up to $0.38
of balance is held; run it outside the device's availability window so the
task is still queued).

Environment:
    EIGENORBIT_ROUTER_API_KEY      A funded EigenOrbit API key (eo_live_... or eo_test_...).
    EIGENORBIT_ROUTER_URL          The router to test. Required: this script
                                   never falls back to the production default.

Usage:
    uv run python scripts/live_cancel.py [--report cancel.json]
    uv run python scripts/live_cancel.py --device braket/iqm_garnet --shots 10

Exit status: 0 when the cancel was accepted and the job ended cancelled or
completed, 1 otherwise (the job ended before the cancel was sent, the cancel
was refused, or the job failed), 2 on a usage or configuration error, 130 on
Ctrl-C (the job keeps running and is billed; the script prints its ID). The
output and the report never contain the API key.

Cost: one paid vendor task. On SV1 about $0.09 is held and a few cents are
charged at most.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from decimal import Decimal
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np
from eigenkit import Circuit
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import Z

from eigenorbit_router_client import RouterConfig, RouterError
from eigenorbit_router_client._client import RouterHttpClient


API_KEY_ENV = "EIGENORBIT_ROUTER_API_KEY"
URL_ENV = "EIGENORBIT_ROUTER_URL"
DEFAULT_DEVICE = "braket/sv1"
DEFAULT_SHOTS = 100
SIMULATORS = frozenset({"braket/sv1", "braket/dm1"})
# A long simulator task, so it is still running when the cancel is sent.
SIMULATOR_QUBITS, SIMULATOR_LAYERS = 24, 20
# A QPU job waits in the vendor queue anyway; keep its circuit small.
QPU_QUBITS, QPU_LAYERS = 2, 1
TERMINAL = frozenset({"completed", "failed", "cancelled", "rejected"})
PASSING = frozenset({"cancelled", "completed"})
WAIT_DELAY = 1.0
POLL_DELAYS = (1.0, 5.0, 10.0)
JOB_FIELDS = (
    "id",
    "status",
    "billing",
    "device",
    "shots",
    "estimate_usd_micro",
    "actual_cost_usd_micro",
    "vendor_cost_usd_micro",
    "fee_usd_micro",
    "price_version",
    "error",
    "created_at",
    "completed_at",
)

_KEY_PATTERN = re.compile(r"eo_(?:live|test)_[A-Za-z0-9_-]*")
_sleep = time.sleep
_monotonic = time.monotonic


def _redact(text: str) -> str:
    """Remove anything shaped like an API key, and the configured key itself."""
    key = (os.environ.get(API_KEY_ENV) or "").strip()
    if key:
        text = text.replace(key, "[redacted API key]")
    return _KEY_PATTERN.sub("[redacted API key]", text)


def _say(text: str) -> None:
    print(_redact(text), flush=True)


def _usd(micro: Any) -> str:
    if not isinstance(micro, int) or isinstance(micro, bool):
        return "n/a"
    return f"${Decimal(micro) / Decimal(1_000_000):f}"


def build_circuit(qubits: int, layers: int):
    """`layers` of RY on every qubit followed by a CNOT chain, measured with Z0.

    Returns (spec, observables, weights, inputs).
    """
    gates = []
    index = 0
    for _ in range(layers):
        for qubit in range(qubits):
            gates.append(Gate("RY", (qubit,), (WeightRef(index),)))
            index += 1
        for qubit in range(qubits - 1):
            gates.append(Gate("CNOT", (qubit, qubit + 1)))
    spec = CircuitSpec(n_qubits=qubits, n_weights=index, n_inputs=0, gates=tuple(gates))
    weights = np.linspace(-1.0, 1.0, index) if index else np.zeros(0)
    return spec, [Z(0).to_observable()], weights, np.zeros(0)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be an int >= 1")
    return value


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line and fill the per-device defaults."""
    parser = argparse.ArgumentParser(
        prog="live_cancel.py",
        description="Submit one long router job, cancel it, and poll it to the end.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="default: braket/sv1")
    parser.add_argument("--shots", type=_positive_int, default=DEFAULT_SHOTS)
    parser.add_argument("--qubits", type=_positive_int, help="default: 24 (SV1), 2 (QPU)")
    parser.add_argument("--layers", type=_positive_int, help="default: 20 (SV1), 1 (QPU)")
    parser.add_argument(
        "--cancel-at",
        choices=["queued", "running"],
        help="cancel once the job has this status; default running (SV1), queued (QPU)",
    )
    parser.add_argument(
        "--wait-seconds",
        type=_positive_int,
        default=600,
        help="how long to wait for the cancel point before cancelling anyway",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=1800,
        help="how long to poll after the cancel for a terminal status",
    )
    parser.add_argument("--report", metavar="PATH", help="write a JSON report here")
    args = parser.parse_args(argv)
    simulator = args.device in SIMULATORS
    if args.qubits is None:
        args.qubits = SIMULATOR_QUBITS if simulator else QPU_QUBITS
    if args.layers is None:
        args.layers = SIMULATOR_LAYERS if simulator else QPU_LAYERS
    if args.cancel_at is None:
        args.cancel_at = "running" if simulator else "queued"
    return args


def _wait_for_cancel_point(handle, cancel_at: str, wait_seconds: float) -> str:
    """Poll until the job reaches `cancel_at` (queued: any live status) or ends."""
    deadline = _monotonic() + wait_seconds
    last = None
    while True:
        status = handle.status()
        if status != last:
            _say(f"status: {status}")
            last = status
        if status in TERMINAL:
            return status
        if cancel_at == "queued" or status == "running":
            return status
        if _monotonic() >= deadline:
            _say(f"still {status} after {wait_seconds:g} s; cancelling anyway")
            return status
        _sleep(WAIT_DELAY)


def _poll_to_terminal(handle, timeout: float) -> str:
    deadline = _monotonic() + timeout
    polls = 0
    last = None
    while True:
        status = handle.status()
        if status != last:
            _say(f"status: {status}")
            last = status
        if status in TERMINAL:
            return status
        if _monotonic() >= deadline:
            _say(f"still {status} after {timeout:g} s; giving up (the job keeps running)")
            return status
        _sleep(POLL_DELAYS[min(polls, len(POLL_DELAYS) - 1)])
        polls += 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (os.environ.get(API_KEY_ENV) or "").strip():
        print(
            f"error: set {API_KEY_ENV} to a funded EigenOrbit API key (eo_live_... or eo_test_...)",
            file=sys.stderr,
        )
        return 2
    url = (os.environ.get(URL_ENV) or "").strip()
    if not url:
        print(
            f"error: set {URL_ENV} to the URL of the router under test. This script "
            f"never falls back to the production default URL.",
            file=sys.stderr,
        )
        return 2
    try:
        # The key stays in the environment: the config reads it per request.
        config = RouterConfig(device=args.device, shots=args.shots, base_url=url)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    spec, observables, weights, inputs = build_circuit(args.qubits, args.layers)
    _say(
        f"live cancel: {args.device}, shots {args.shots}, {args.qubits} qubits x "
        f"{args.layers} layers ({len(spec.gates)} gates); cancel when {args.cancel_at}"
    )
    record: dict[str, Any] = {
        "tool": "scripts/live_cancel.py",
        "router_host": urlsplit(url).hostname,
        "device": args.device,
        "shots": args.shots,
        "qubits": args.qubits,
        "layers": args.layers,
        "cancel_at": args.cancel_at,
        "job_id": None,
        "status_at_cancel": None,
        "cancel_status": None,
        "cancel_error": None,
        "final_status": None,
        "job": None,
        "passed": False,
        "reason": "",
    }
    kit_circuit = None
    http = RouterHttpClient(config)
    try:
        kit_circuit = Circuit(spec, observables, provider=config)
        handle = kit_circuit.submit(weights, inputs)
        job_id = handle.job_id()
        record["job_id"] = job_id
        _say(f"submitted job {job_id}")
        status = _wait_for_cancel_point(handle, args.cancel_at, args.wait_seconds)
        record["status_at_cancel"] = status
        if status in TERMINAL:
            record["reason"] = (
                f"the job ended {status} before the cancel could be sent; use a longer "
                f"circuit (--layers)"
            )
            _say(record["reason"])
            final = status
        else:
            try:
                record["cancel_status"] = handle.cancel()
                _say(f"cancel requested (202): status {record['cancel_status']}")
            except RouterError as error:
                record["cancel_error"] = error.code
                record["reason"] = f"the cancel was refused: {error.code}: {error.message}"
                _say(record["reason"])
            final = _poll_to_terminal(handle, args.timeout)
        job = http.get_job(job_id)
        record["job"] = {key: job.get(key) for key in JOB_FIELDS}
        final = job.get("status", final)
        record["final_status"] = final
        cancelled_ok = record["cancel_status"] is not None
        record["passed"] = cancelled_ok and final in PASSING
        if cancelled_ok and final not in PASSING:
            record["reason"] = f"the job ended {final}"
        _say(f"job id: {job_id}")
        _say(f"final status: {final}")
        _say(f"billing: {job.get('billing')}")
        _say(
            f"charged: {_usd(job.get('actual_cost_usd_micro'))} "
            f"(actual_cost_usd_micro={job.get('actual_cost_usd_micro')}, "
            f"vendor_cost_usd_micro={job.get('vendor_cost_usd_micro')}, "
            f"fee_usd_micro={job.get('fee_usd_micro')}; "
            f"estimate_usd_micro={job.get('estimate_usd_micro')})"
        )
        if final == "completed" and cancelled_ok:
            _say(
                "note: the vendor finished the task first; it is charged like "
                "any completed job"
            )
    except KeyboardInterrupt:
        _say(
            f"interrupted: job {record['job_id'] or '(not submitted)'} keeps running and "
            f"is billed; check it with GET /v1/jobs/{record['job_id'] or '<id>'}"
        )
        return 130
    except RouterError as error:
        record["reason"] = f"{error.code}: {error.message}"
        _say(f"error: {record['reason']}")
    finally:
        if kit_circuit is not None:
            kit_circuit.compiled.close()
        http.close()

    _say(f"{'PASS' if record['passed'] else 'FAIL'}: {record['reason'] or 'cancel E2E'}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(_redact(json.dumps(record, indent=2)) + "\n")
        _say(f"report written to {args.report}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
