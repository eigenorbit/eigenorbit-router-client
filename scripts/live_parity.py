"""Live parity: a vendor device through your own credentials vs through the router.

Runs the same kit circuits twice, once with your own vendor credentials and
once with `RouterConfig(device=...)` through EigenOrbit Router, and
checks that the two legs agree. `--device` picks the router device (default
braket/sv1).

A router worker moves to a new eigenkit version only after a live parity run
of this script with that version on both legs, on the devices the router
serves.

Braket: the own leg is `BraketConfig` on the device's Braket ARN (your AWS
profile, in the device's region). The router's Braket QPUs are
braket/iqm_garnet and braket/rigetti_cepheus_1_108q. The ARNs, regions and
shot ranges in DEVICES are the router's catalog seed.

IBM Quantum (`--device ibm/<backend>`): the own leg is
`QiskitRuntimeConfig(backend_name=<backend>, shots=N, instance=$IBM_INSTANCE_CRN)`
with the token the kit reads from QISKIT_IBM_TOKEN, and every other setting the
kit's default, as the router's worker uses it. The router leg runs on the IBM
account you connected to the router account behind EIGENORBIT_ROUTER_API_KEY:
connect the same token and instance first. Every router job must report
`credential_source` "own". A job that reports "organization" ran on
an IBM account EigenOrbit provides, and fails its case.

IBM cases per circuit, at shots N (default 100): `run`, `sample` and
`probabilities` (IBM does not offer zero_state_probability, and a QPU has no
shots 0). The QPU rule below applies; for `sample` the key width (n_qubits
characters of 0/1), the shot total and the qubit-0-MSB order are checked
exactly and the distributions loosely.

On a simulator (braket/sv1, braket/dm1):

- identical result shapes and dtypes;
- at shots 0, |own - router| <= 1e-9;
- at shots N, |own - router| <= k * sqrt(sigma_own^2 + sigma_router^2), where
  sigma comes from exact local term expectations (kit numpy simulator):
  sum_i |c_i| * sqrt((1 - <P_i>^2) / N) for an observable, sqrt(p (1 - p) / N)
  for zero_state_probability, floored at 1/N. Default k = 5.

On a QPU the shapes, dtypes and bit ordering must still match exactly, but each
value only has to lie within max(k * shot-noise sigma, QPU_ABS_TOL), with
QPU_ABS_TOL = 0.15: two runs on real hardware differ by drift and readout error
as well as shot noise. A QPU has no exact mode, so there are no shots-0 cases.

Cases per circuit: run at shots 0 (simulators only), run at shots N,
zero_state_probability at shots N (Braket refuses zero_state_probability at
shots 0).

Circuits (`--circuits`):

- `ry`: the contract's RY(0.5) with Z0;
- `ansatz2`: AngleEncoding + BasicVariational on 2 qubits with Z on each;
- `cnot3`: a 3-qubit CNOT chain measured with 0.5 Z0 Z1 + 0.25 X2;
- `mc3`: MCZ, CRY and MCX on 3 qubits with Z on each (all_z(3), one qubit-wise
  commuting group, so one task per case on a Braket QPU). On Braket these gates
  lower to CCNot, CNot, H and Ry, which IQM Garnet accepts.

`all` (the default) is ry, ansatz2 and cnot3. `garnet` is ansatz2 and mc3: one
QPU run that covers MCZ, MCX and CRY next to the encoding ansatz.

`--router-only` needs no AWS or IBM credentials at all: it runs every case
through the router only, compares with the exact local reference, and also
drives one `submit()` through `job_id()`, `status()` and `result()`.

A Braket QPU run costs real money on both legs. It asks the router's estimate
for every job first and prints the total (vendor cost + EigenOrbit fee), and
it submits nothing unless `--i-accept-cost` is given. An IBM run carries no
router charge but spends quantum seconds of the IBM instance each leg runs on;
it prints that first and also needs `--i-accept-cost`.

Environment:
    EIGENORBIT_ROUTER_API_KEY      A funded EigenOrbit API key (eo_live_... or eo_test_...).
    EIGENORBIT_ROUTER_URL          The router to test. Required: this script
                                   never falls back to the production default.
    AWS_PROFILE                    Braket own leg only. The region is the
                                   device's (us-east-1 for the simulators when
                                   AWS_REGION is not set).
    QISKIT_IBM_TOKEN               IBM own leg only: the IBM Quantum API token,
                                   read by the kit, never printed or reported.
    IBM_INSTANCE_CRN               IBM own leg only: the CRN of the IBM instance.

Usage:
    uv run --group live python scripts/live_parity.py --report parity.json
    uv run python scripts/live_parity.py --router-only --report router-only.json
    uv run --group live python scripts/live_parity.py --device braket/iqm_garnet \\
        --circuits garnet --shots 100 --i-accept-cost --report parity-garnet.json
    uv run --group live python scripts/live_parity.py --device ibm/ibm_fez \\
        --circuits ansatz2 --i-accept-cost --report parity-ibm.json

Exit status: 0 when every case passes, 1 when any case fails, 2 on a usage or
configuration error (including a job the router would reject), 3 when a QPU
run was refused because --i-accept-cost is missing. The report never contains
the API key or the IBM token.

Cost: one full SV1 run is 9 SV1 tasks on each leg (the router leg holds about
$0.09 per job, so it needs about $0.81 of available balance). A Braket QPU run
is 2 tasks per circuit on each leg; the script prints the router's estimate
first. `--circuits garnet` on braket/iqm_garnet at 100 shots is 4 tasks per leg,
about $4 for both legs at the router's seeded prices. An IBM run is 3 IBM jobs
per circuit on each leg (`--circuits ansatz2`: 6 jobs at 100 shots, a few
quantum seconds).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from decimal import Decimal
from importlib import metadata
from typing import Any, Callable, Sequence
from urllib.parse import urlsplit

import numpy as np
from eigenkit import Circuit
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.ir import CircuitSpec, Gate, Observable, PauliTerm, WeightRef
from eigenkit.core.observables import X, Z, all_z
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

import eigenorbit_router_client
from eigenorbit_router_client import RouterConfig, RouterError, _client


API_KEY_ENV = "EIGENORBIT_ROUTER_API_KEY"
URL_ENV = "EIGENORBIT_ROUTER_URL"
ROUTER_DEVICE = "braket/sv1"  # the default --device
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
OWN_REGION = "us-east-1"
ATOL_EXACT = 1e-9
DEFAULT_SHOTS = 1000
# A few shots on one cheap IBM device: the IBM default of --shots.
IBM_DEFAULT_SHOTS = 100
IBM_MAX_SHOTS = 100_000
# The IBM devices in the router's catalog seed; any ibm/<backend> is accepted.
IBM_SEEDED = ("ibm/ibm_fez", "ibm/ibm_kingston", "ibm/ibm_marrakesh")
IBM_TOKEN_ENV = "QISKIT_IBM_TOKEN"
IBM_INSTANCE_ENV = "IBM_INSTANCE_CRN"
# The credential_source every router job of an IBM run must report: the run
# compares your own IBM credentials with the IBM account you connected.
IBM_CREDENTIAL_SOURCE = "own"
IBM_COST_STATEMENT = (
    "IBM cost: no router charge. This run spends quantum seconds of the IBM "
    "instance each leg runs on."
)
DEFAULT_K = 5.0
# The absolute floor of the QPU tolerance, for expectation values and
# probabilities alike: two runs on one QPU differ by hardware drift and readout
# error on top of shot noise.
QPU_ABS_TOL = 0.15
EXIT_COST_NOT_ACCEPTED = 3

_KEY_PATTERN = re.compile(r"eo_(?:live|test)_[A-Za-z0-9_-]*")
_MICRO_PER_USD = Decimal(1_000_000)
_SIX_PLACES = Decimal("0.000001")


# ---- devices -----------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """A router device and the vendor device the own-credentials leg runs on.

    `min_shots`..`max_shots` is the catalog's shot range for the sampled cases.
    `arn` and `region` are the Braket device's; an IBM device has neither.
    """

    id: str
    arn: str
    region: str
    kind: str  # "simulator" or "qpu"
    min_shots: int
    max_shots: int

    @property
    def is_qpu(self) -> bool:
        return self.kind == "qpu"

    @property
    def vendor(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def is_ibm(self) -> bool:
        return self.vendor == "ibm"

    @property
    def backend_name(self) -> str:
        """The vendor's own device name: `ibm_fez` for ibm/ibm_fez."""
        return self.id.split("/", 1)[1]


def ibm_device(device_id: str) -> Device:
    """An IBM device `ibm/<backend>`: a QPU, shots 1..100000 (contract section 4)."""
    return Device(device_id, "", "", "qpu", 1, IBM_MAX_SHOTS)


# The router's catalog seed.
DEVICES: dict[str, Device] = {
    device.id: device
    for device in (
        Device("braket/sv1", SV1_ARN, OWN_REGION, "simulator", 0, 100_000),
        Device("braket/dm1", DM1_ARN, OWN_REGION, "simulator", 0, 100_000),
        Device(
            "braket/iqm_garnet",
            "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
            "eu-north-1",
            "qpu",
            1,
            20_000,
        ),
        Device(
            "braket/rigetti_cepheus_1_108q",
            "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
            "us-west-1",
            "qpu",
            10,
            50_000,
        ),
    )
}


def acceptance_rule(device: Device, k: float) -> str:
    """The rule a case must meet on `device`, as printed and reported."""
    if device.is_qpu:
        rule = (
            f"shapes, dtypes and bit ordering must match exactly; each value must lie "
            f"within max({k:g} x shot-noise sigma, QPU_ABS_TOL = {QPU_ABS_TOL:g}) "
            f"(hardware drift and readout error between two QPU runs)"
        )
        if device.is_ibm:
            rule += (
                "; sample: key width, shot total and qubit-0-MSB order exact, the "
                "count distributions within the same per-state rule"
            )
        return rule
    return (
        f"shapes and dtypes must match exactly; |diff| <= {ATOL_EXACT:g} at shots 0 and "
        f"<= {k:g} x shot-noise sigma at shots N"
    )


# ---- circuits and cases ------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ParityCircuit:
    """A kit circuit with fixed parameters, measured with a list of observables."""

    name: str
    spec: CircuitSpec
    observables: list[Observable]
    weights: np.ndarray
    inputs: np.ndarray


@dataclass(frozen=True, eq=False)
class Case:
    """One call to compare: a circuit, a method and a shot count."""

    circuit: ParityCircuit
    method: str  # "run", "zero_state_probability", or on IBM "sample"/"probabilities"
    shots: int

    @property
    def label(self) -> str:
        return f"{self.circuit.name}/{self.method}/shots={self.shots}"


def _ry() -> ParityCircuit:
    # The contract's job_submit_request example: RY(w0) on one qubit, Z0.
    spec = CircuitSpec(
        n_qubits=1, n_weights=1, n_inputs=0, gates=(Gate("RY", (0,), (WeightRef(0),)),)
    )
    return ParityCircuit("ry", spec, [Z(0).to_observable()], np.array([0.5]), np.array([]))


def _ansatz2() -> ParityCircuit:
    spec = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
    weights = np.linspace(-0.9, 1.1, spec.n_weights)
    return ParityCircuit("ansatz2", spec, all_z(2), weights, np.array([0.2, 0.6]))


def _cnot3() -> ParityCircuit:
    spec = CircuitSpec(
        n_qubits=3,
        n_weights=3,
        n_inputs=0,
        gates=(
            Gate("RY", (0,), (WeightRef(0),)),
            Gate("CNOT", (0, 1)),
            Gate("CNOT", (1, 2)),
            Gate("RY", (1,), (WeightRef(1),)),
            Gate("RY", (2,), (WeightRef(2),)),
        ),
    )
    observable = (0.5 * Z(0) * Z(1) + 0.25 * X(2)).to_observable()
    return ParityCircuit("cnot3", spec, [observable], np.array([0.7, -0.4, 0.9]), np.array([]))


def _mc3() -> ParityCircuit:
    # MCZ, CRY and MCX on 3 qubits. On Braket they lower to CCNot, CNot, H and
    # Ry, gates every Braket QPU accepts.
    spec = CircuitSpec(
        n_qubits=3,
        n_weights=4,
        n_inputs=0,
        gates=(
            Gate("RY", (0,), (WeightRef(0),)),
            Gate("RY", (1,), (WeightRef(1),)),
            Gate("H", (2,)),
            Gate("MCZ", (0, 1, 2)),
            Gate("CRY", (0, 2), (WeightRef(2),)),
            Gate("MCX", (1, 2, 0)),
            Gate("RY", (2,), (WeightRef(3),)),
        ),
    )
    # all_z(3) is one qubit-wise commuting group, so a Braket QPU runs each case
    # as one task.
    return ParityCircuit("mc3", spec, all_z(3), np.array([0.9, -0.7, 1.2, 0.4]), np.array([]))


_CIRCUITS: dict[str, Callable[[], ParityCircuit]] = {
    "ry": _ry,
    "ansatz2": _ansatz2,
    "cnot3": _cnot3,
    "mc3": _mc3,
}

# Named circuit sets for `--circuits`. `garnet` covers MCZ, MCX and CRY on a QPU
# in one run (4 tasks per leg).
_CIRCUIT_GROUPS: dict[str, tuple[str, ...]] = {
    "all": ("ry", "ansatz2", "cnot3"),
    "garnet": ("ansatz2", "mc3"),
}


def select_circuits(which: str) -> list[ParityCircuit]:
    """The circuits named by `--circuits`: a set from `_CIRCUIT_GROUPS` or one circuit."""
    names = _CIRCUIT_GROUPS.get(which, (which,))
    return [_CIRCUITS[name]() for name in names]


def build_cases(
    circuits: Sequence[ParityCircuit], shots: int, *, exact: bool = True
) -> list[Case]:
    """run at shots 0 (when `exact`), run at `shots` and zero_state_probability at `shots`.

    zero_state_probability at shots 0 is left out: Braket refuses the
    probability result type without shots. A QPU has no exact mode, so QPU
    runs pass `exact=False` and have no shots-0 case.
    """
    cases = []
    for circuit in circuits:
        if exact:
            cases.append(Case(circuit, "run", 0))
        cases.append(Case(circuit, "run", shots))
        cases.append(Case(circuit, "zero_state_probability", shots))
    return cases


IBM_METHODS = ("run", "sample", "probabilities")


def build_ibm_cases(circuits: Sequence[ParityCircuit], shots: int) -> list[Case]:
    """run, sample and probabilities at `shots` for every circuit.

    IBM offers no zero_state_probability (the kit IBM path raises for it) and a
    QPU has no exact mode, so there is no zero_state_probability and no shots-0
    case.
    """
    return [Case(circuit, method, shots) for circuit in circuits for method in IBM_METHODS]


def cases_for(device: Device, circuits: Sequence[ParityCircuit], shots: int) -> list[Case]:
    """The cases a run on `device` compares."""
    if device.is_ibm:
        return build_ibm_cases(circuits, shots)
    return build_cases(circuits, shots, exact=not device.is_qpu)


# ---- exact references and tolerances -----------------------------------------------


def _numpy_circuit(circuit: ParityCircuit, observables=None):
    return Circuit(
        circuit.spec,
        circuit.observables if observables is None else observables,
        provider="numpy",
    )


def exact_terms(
    spec: CircuitSpec, observable: Observable, weights, inputs
) -> list[tuple[float, float]]:
    """(coefficient, exact <P_i>) for every Pauli term of `observable`.

    Computed with the kit numpy simulator; each term is measured with
    coefficient 1.
    """
    terms = []
    for term in observable.terms:
        unit = Observable(terms=(PauliTerm(coeff=1.0, paulis=term.paulis),))
        value = Circuit(spec, [unit], provider="numpy")(
            np.asarray(weights, dtype=np.float64), np.asarray(inputs, dtype=np.float64)
        )
        terms.append((float(term.coeff), float(value[0])))
    return terms


DISTRIBUTION_METHODS = frozenset({"sample", "probabilities"})


def exact_value(case: Case) -> np.ndarray | float:
    """The exact result of `case` on the kit numpy simulator, in the kit's shape.

    For `sample` it is the exact distribution the counts sample: the
    `(2**n_qubits,)` probabilities in qubit-0-MSB order.
    """
    circuit = case.circuit
    numpy_circuit = _numpy_circuit(circuit)
    if case.method == "run":
        return numpy_circuit(circuit.weights, circuit.inputs)
    if case.method in DISTRIBUTION_METHODS:
        return numpy_circuit.probabilities(circuit.weights, circuit.inputs)
    return numpy_circuit.zero_state_probability(circuit.weights, circuit.inputs)


def tolerance(
    shots: int,
    *,
    coeffs: Sequence[float] | None = None,
    term_expectations: Sequence[float] | None = None,
    p: float | None = None,
    k: float = DEFAULT_K,
    legs: int = 2,
) -> float:
    """The allowed |difference| for one value.

    Args:
        shots: 0 for exact results, otherwise the shot count N of each leg.
        coeffs, term_expectations: The observable's Pauli coefficients and
            their exact expectations (for `run`).
        p: The exact all-zeros probability (for `zero_state_probability`).
        k: How many sigmas to allow.
        legs: How many legs sample: 2 when comparing two sampled legs, 1 when
            comparing one sampled leg with the exact reference.

    Returns:
        1e-9 at shots 0, otherwise k * sqrt(legs) * sigma with sigma floored
        at 1/N.
    """
    if shots == 0:
        return ATOL_EXACT
    if p is not None:
        sigma = math.sqrt(max(p * (1.0 - p), 0.0) / shots)
    else:
        if coeffs is None or term_expectations is None:
            raise ValueError("pass coeffs and term_expectations, or p")
        sigma = sum(
            abs(c) * math.sqrt(max(1.0 - e * e, 0.0) / shots)
            for c, e in zip(coeffs, term_expectations, strict=True)
        )
    sigma = max(sigma, 1.0 / shots)
    return k * math.sqrt(legs) * sigma


def qpu_tolerance(shots: int, **kwargs: Any) -> float:
    """The allowed |difference| for one value on a QPU.

    max(tolerance(shots, ...), QPU_ABS_TOL): the shot-noise rule, never
    tighter than the hardware floor. Takes the same arguments as `tolerance`.
    """
    return max(tolerance(shots, **kwargs), QPU_ABS_TOL)


def case_tolerance(
    case: Case, k: float, legs: int, *, kind: str = "simulator"
) -> np.ndarray | float:
    """Per-observable tolerances for `run`, one tolerance for ZSP, one per basis
    state for `sample` and `probabilities`.

    `kind` "qpu" applies `qpu_tolerance`; any other kind keeps the
    simulator's shot-noise `tolerance` unchanged.
    """
    rule = qpu_tolerance if kind == "qpu" else tolerance
    circuit = case.circuit
    if case.method == "zero_state_probability":
        p = float(exact_value(case))
        return rule(case.shots, p=p, k=k, legs=legs)
    if case.method in DISTRIBUTION_METHODS:
        # One binomial tolerance per basis state.
        return np.array(
            [rule(case.shots, p=float(p), k=k, legs=legs) for p in exact_value(case)]
        )
    tolerances = []
    for observable in circuit.observables:
        terms = exact_terms(circuit.spec, observable, circuit.weights, circuit.inputs)
        tolerances.append(
            rule(
                case.shots,
                coeffs=[c for c, _ in terms],
                term_expectations=[e for _, e in terms],
                k=k,
                legs=legs,
            )
        )
    return np.array(tolerances)


# ---- comparison --------------------------------------------------------------------


def _full(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(float(array))
    return "[" + ", ".join(repr(float(x)) for x in array.ravel()) + "]"


def _jsonable(value: Any) -> Any:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if array.ndim == 0 else array.tolist()


def compare(own, router, tol, *, case: str = "") -> dict[str, Any]:
    """Compare one case's two values.

    Fails when the shapes or dtypes differ, or when any |own - router|
    exceeds `tol` (a scalar or one tolerance per observable).

    Returns:
        A dict with the case label, both values, the tolerance, the largest
        |difference|, `passed`, `reason` and a one-line `report` naming the
        case and both values.
    """
    own_array, router_array = np.asarray(own), np.asarray(router)
    tol_array = np.asarray(tol, dtype=np.float64)
    result: dict[str, Any] = {
        "case": case,
        "own": own_array.tolist(),
        "router": router_array.tolist(),
        "shape": list(router_array.shape),
        "dtype": str(router_array.dtype),
        "tolerance": tol_array.tolist(),
        "max_abs_diff": None,
        "passed": False,
        "reason": "",
    }
    if own_array.shape != router_array.shape:
        result["reason"] = f"shape {own_array.shape} != {router_array.shape}"
    elif own_array.dtype != router_array.dtype:
        result["reason"] = f"dtype {own_array.dtype} != {router_array.dtype}"
    else:
        diff = np.abs(own_array.astype(np.float64) - router_array.astype(np.float64))
        result["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
        if np.all(diff <= tol_array):
            result["passed"] = True
        else:
            result["reason"] = f"|diff| {_full(diff)} > tolerance {_full(tol_array)}"
    verdict = "PASS" if result["passed"] else "FAIL"
    result["report"] = (
        f"{verdict} {case}: own={_full(own_array)} router={_full(router_array)}"
        + (f" ({result['reason']})" if result["reason"] else "")
    )
    return result


def counts_problem(counts: Any, n_qubits: int, shots: int) -> str | None:
    """Why `counts` is not a sample of `shots` shots on `n_qubits`, or None.

    Exact: a dict of `n_qubits`-character 0/1 keys to non-negative int
    counts that sum to `shots`.
    """
    if not isinstance(counts, dict):
        return f"not a counts dict ({type(counts).__name__})"
    for key, count in counts.items():
        if not isinstance(key, str) or len(key) != n_qubits or set(key) - {"0", "1"}:
            return f"key {key!r} is not {n_qubits} characters of 0/1"
        if not _is_int(count) or count < 0:
            return f"count {count!r} of {key!r} is not a non-negative int"
    total = sum(counts.values())
    if total != shots:
        return f"counts sum to {total}, not {shots} shots"
    return None


def counts_to_probabilities(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """A qubit-0-MSB counts dict as a `(2**n_qubits,)` float64 distribution."""
    probabilities = np.zeros(2**n_qubits, dtype=np.float64)
    total = sum(counts.values())
    for key, count in counts.items():
        probabilities[int(key, 2)] = count / total if total else 0.0
    return probabilities


def _bit_reversed(probabilities: np.ndarray, n_qubits: int) -> np.ndarray:
    """The distribution with each basis index's bits reversed (qubit 0 as the LSB)."""
    reversed_ = np.empty_like(probabilities)
    for index, p in enumerate(probabilities):
        reversed_[int(format(index, f"0{n_qubits}b")[::-1], 2)] = p
    return reversed_


def bit_order_problem(
    probabilities: np.ndarray, exact: np.ndarray, n_qubits: int
) -> str | None:
    """Why `probabilities` looks bit-reversed against the exact qubit-0-MSB
    distribution, or None.

    Decisive only when the exact distribution and its bit reversal are far
    apart (L1 >= 2 x QPU_ABS_TOL); otherwise hardware noise could mimic a
    reversal, and the per-state comparison is the check.
    """
    flipped = _bit_reversed(np.asarray(exact, dtype=np.float64), n_qubits)
    if float(np.abs(flipped - exact).sum()) < 2 * QPU_ABS_TOL:
        return None
    to_msb = float(np.abs(probabilities - exact).sum())
    to_lsb = float(np.abs(probabilities - flipped).sum())
    if to_lsb < to_msb:
        return (
            f"closer to the bit-reversed distribution (L1 {to_lsb:.3f}) than to the "
            f"qubit-0-MSB one (L1 {to_msb:.3f})"
        )
    return None


def compare_distribution(
    own, router, tol, *, case: Case, exact: np.ndarray
) -> dict[str, Any]:
    """Compare one `sample` or `probabilities` case.

    `sample` values are counts dicts (the router-only reference is the exact
    distribution): the key width, the shot total and the qubit-0-MSB order
    are exact, then the two distributions go through `compare` with one
    tolerance per basis state. `probabilities` values go through `compare`
    (shape and dtype exact) after the same bit-order check.
    """
    n_qubits = case.circuit.spec.n_qubits
    problems = []
    vectors = []
    counts: dict[str, Any] = {}
    for leg, value in (("own", own), ("router", router)):
        if case.method == "sample" and isinstance(value, dict):
            problem = counts_problem(value, n_qubits, case.shots)
            counts[f"{leg}_counts"] = value
            if problem is not None:
                problems.append(f"{leg} counts: {problem}")
                vectors.append(None)
                continue
            value = counts_to_probabilities(value, n_qubits)
        elif case.method == "sample" and leg == "router":
            problems.append(f"router sample is not a counts dict ({type(value).__name__})")
            vectors.append(None)
            continue
        value = np.asarray(value)
        if value.shape == exact.shape:
            order = bit_order_problem(value.astype(np.float64), exact, n_qubits)
            if order is not None:
                problems.append(f"{leg} bit order: {order}")
        vectors.append(value)
    if problems or any(v is None for v in vectors):
        reason = "; ".join(problems)
        result = {
            "case": case.label,
            "own": None if vectors[0] is None else vectors[0].tolist(),
            "router": None if vectors[1] is None else vectors[1].tolist(),
            "shape": None,
            "dtype": None,
            "tolerance": np.asarray(tol, dtype=np.float64).tolist(),
            "max_abs_diff": None,
            "passed": False,
            "reason": reason,
            "report": f"FAIL {case.label}: {reason}",
        }
    else:
        result = compare(vectors[0], vectors[1], tol, case=case.label)
    result.update(counts)
    return result


@dataclass(frozen=True)
class LegFailure:
    """A leg could not produce a value for a case."""

    leg: str
    error: str

    def __str__(self) -> str:
        return f"{self.leg} leg failed: {self.error}"


def _failed(case: Case, failure: LegFailure, tol) -> dict[str, Any]:
    reason = _redact(str(failure))
    return {
        "case": case.label,
        "own": None,
        "router": None,
        "shape": None,
        "dtype": None,
        "tolerance": np.asarray(tol, dtype=np.float64).tolist(),
        "max_abs_diff": None,
        "passed": False,
        "reason": reason,
        "report": f"FAIL {case.label}: {reason}",
    }


def _redact(text: str) -> str:
    """Remove anything shaped like an API key, the configured key itself and the
    IBM token the own leg's kit reads."""
    for env, label in (
        (API_KEY_ENV, "[redacted API key]"),
        (IBM_TOKEN_ENV, "[redacted IBM token]"),
    ):
        secret = (os.environ.get(env) or "").strip()
        if secret:
            text = text.replace(secret, label)
    return _KEY_PATTERN.sub("[redacted API key]", text)


# ---- cost --------------------------------------------------------------------------


def _usd(micro: int | None) -> str:
    """Integer micro-USD as an exact dollar string: 90000 -> "$0.090000"."""
    if micro is None:
        return "-"
    return f"${(Decimal(micro) / _MICRO_PER_USD).quantize(_SIX_PLACES)}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sum_or_none(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(_is_int(v) for v in values) else None


def estimate_cases(cases: Sequence[Case], config: RouterConfig) -> dict[str, Any]:
    """The router's estimate (POST /v1/jobs/estimate) for every case; submits nothing.

    Returns:
        {"cases": one row per case, "jobs", "estimate_usd_micro",
        "vendor_cost_usd_micro", "fee_usd_micro", "available_usd_micro"}. The
        vendor and fee totals are None when the router does not split them.

    Raises:
        RouterError: The router refused an estimate, or answered off-contract.
    """
    http = _client.RouterHttpClient(config)
    rows = []
    try:
        for case in cases:
            circuit = case.circuit
            out = http.estimate(
                {
                    "device": config.device,
                    "method": case.method,
                    "shots": case.shots,
                    "circuit_spec": circuit_spec_to_json(circuit.spec),
                    "observables": observables_to_json(list(circuit.observables)),
                    "weights": np.asarray(circuit.weights, dtype=np.float64).tolist(),
                    "inputs": np.asarray(circuit.inputs, dtype=np.float64).tolist(),
                }
            )
            if not all(_is_int(out.get(k)) for k in ("estimate_usd_micro", "available_usd_micro")):
                raise RouterError(
                    "invalid_response",
                    "the router's estimate has no integer estimate_usd_micro and "
                    "available_usd_micro",
                )
            rows.append(
                {
                    "case": case.label,
                    "estimate_usd_micro": out["estimate_usd_micro"],
                    "vendor_cost_usd_micro": out.get("vendor_cost_usd_micro"),
                    "fee_usd_micro": out.get("fee_usd_micro"),
                    "available_usd_micro": out["available_usd_micro"],
                    "price_version": out.get("price_version"),
                    "would_reject_reason": out.get("would_reject_reason"),
                }
            )
    finally:
        http.close()
    return {
        "cases": rows,
        "jobs": len(rows),
        "estimate_usd_micro": sum(row["estimate_usd_micro"] for row in rows),
        "vendor_cost_usd_micro": _sum_or_none(rows, "vendor_cost_usd_micro"),
        "fee_usd_micro": _sum_or_none(rows, "fee_usd_micro"),
        "available_usd_micro": rows[-1]["available_usd_micro"] if rows else None,
    }


def _print_estimate(estimate: dict[str, Any], device: Device, *, router_only: bool) -> None:
    if device.is_ibm:
        # The router charges nothing for an IBM job (contract section 10).
        print(
            f"router estimate for {estimate['jobs']} IBM jobs on {device.id}: "
            f"{_usd(estimate['estimate_usd_micro'])} (no router charge)"
        )
        return
    print(f"router estimate for {estimate['jobs']} jobs on {device.id}:")
    for row in estimate["cases"]:
        print(f"  {row['case']:<40} {_usd(row['estimate_usd_micro']):>12}")
    split = ""
    if estimate["vendor_cost_usd_micro"] is not None and estimate["fee_usd_micro"] is not None:
        split = (
            f" (vendor {_usd(estimate['vendor_cost_usd_micro'])} + EigenOrbit fee "
            f"{_usd(estimate['fee_usd_micro'])})"
        )
    print(
        f"router leg total: {_usd(estimate['estimate_usd_micro'])}{split}; available "
        f"balance {_usd(estimate['available_usd_micro'])}"
    )
    if not router_only and estimate["vendor_cost_usd_micro"] is not None:
        print(
            f"own leg: about {_usd(estimate['vendor_cost_usd_micro'])} more, billed by AWS "
            f"to your own account ({device.arn} in {device.region})"
        )


def _cost_refusal(estimate: dict[str, Any]) -> str | None:
    """Why the router would not run this set of jobs, or None."""
    rejected = [row for row in estimate["cases"] if row["would_reject_reason"]]
    if rejected:
        reasons = ", ".join(f"{row['case']}: {row['would_reject_reason']}" for row in rejected)
        return f"the router would reject {len(rejected)} of these jobs ({reasons})"
    available = estimate["available_usd_micro"]
    if available is not None and estimate["estimate_usd_micro"] > available:
        return (
            f"the available balance {_usd(available)} does not cover the router leg "
            f"total {_usd(estimate['estimate_usd_micro'])}"
        )
    return None


# ---- legs --------------------------------------------------------------------------


def _close(kit_circuit: Any) -> None:
    """Close the compiled circuit's connections, when its provider has any.

    A RouterCompiled holds an httpx client; leaving it open makes the
    interpreter warn about an unclosed ssl.SSLSocket at exit.
    """
    close = getattr(getattr(kit_circuit, "compiled", None), "close", None)
    if callable(close):
        close()


def _evaluate(kit_circuit: Any, case: Case) -> Any:
    """One case through the kit `Circuit`'s own method for it."""
    circuit = case.circuit
    if case.method == "run":
        return kit_circuit(circuit.weights, circuit.inputs)
    method = getattr(kit_circuit, case.method)  # zero_state_probability, sample, ...
    return method(circuit.weights, circuit.inputs)


def _run_leg(
    leg: str,
    cases: Sequence[Case],
    make_provider: Callable[[int], Any],
    evaluate: Callable[[Any, Case], Any] = _evaluate,
) -> list[Any]:
    """Evaluate every case, compiling once per (circuit, shots).

    Every compiled circuit is closed before this returns, even on failure.

    Returns the value of each case in the kit's shape, or a LegFailure.
    """
    compiled: dict[tuple[str, int], Any] = {}
    values: list[Any] = []
    try:
        for case in cases:
            circuit = case.circuit
            slot = (circuit.name, case.shots)
            try:
                if slot not in compiled:
                    compiled[slot] = Circuit(
                        circuit.spec,
                        circuit.observables,
                        provider=make_provider(case.shots),
                    )
                values.append(evaluate(compiled[slot], case))
            except Exception as error:  # a live run records every failure and goes on
                values.append(
                    LegFailure(leg, _redact(f"{type(error).__name__}: {error}"))
                )
    finally:
        for kit_circuit in compiled.values():
            _close(kit_circuit)
    return values


def run_router_leg(
    cases: Sequence[Case],
    config: RouterConfig,
    *,
    jobs: list[dict[str, Any] | None] | None = None,
) -> list[Any]:
    """Every case through the router, with `config` at each case's shots.

    Returns `run` values as float64 `(n_observables,)` arrays,
    `zero_state_probability` values as floats, `probabilities` as float64
    `(2**n_qubits,)` arrays and `sample` as counts dicts (or a LegFailure per
    case).

    Args:
        jobs: When a list, each case goes through `compiled.submit(method=...)`
            and its handle instead of the blocking call (the same request and
            the same result), and one record per case is appended: the router
            job's `job_id`, `credential_source` and `ibm_usage_seconds`, or
            None when the case failed before a result. The IBM credential
            check reads it.
    """
    evaluate: Callable[[Any, Case], Any] = _evaluate
    if jobs is not None:
        start = len(jobs)
        jobs.extend([None] * len(cases))
        slots = {id(case): start + i for i, case in enumerate(cases)}

        def submit_and_record(kit_circuit: Any, case: Case) -> Any:
            circuit = case.circuit
            handle = kit_circuit.compiled.submit(
                circuit.weights, circuit.inputs, method=case.method
            )
            value = handle.result()
            jobs[slots[id(case)]] = {
                "job_id": handle.job_id(),
                "credential_source": handle.credential_source,
                "ibm_usage_seconds": handle.ibm_usage_seconds,
            }
            if case.method == "sample":
                return dict(value)
            if case.method == "zero_state_probability":
                return float(value)
            return np.asarray(value, dtype=np.float64)

        evaluate = submit_and_record

    return _run_leg(
        "router", cases, lambda shots: dataclasses.replace(config, shots=shots), evaluate
    )


def ibm_own_env_problem() -> str | None:
    """Why the IBM own leg cannot run from this environment, or None.

    Names the variables only; never prints their values.
    """
    missing = [
        env
        for env in (IBM_TOKEN_ENV, IBM_INSTANCE_ENV)
        if not (os.environ.get(env) or "").strip()
    ]
    if not missing:
        return None
    return (
        f"the IBM own leg needs {' and '.join(missing)}: {IBM_TOKEN_ENV} is the IBM "
        f"Quantum API token (read by the kit, never printed) and {IBM_INSTANCE_ENV} "
        f"the CRN of the IBM instance it runs on. --router-only needs neither."
    )


def run_ibm_own_leg(cases: Sequence[Case], device: Device) -> list[Any]:
    """Every case on the IBM device through the caller's own IBM credentials.

    `QiskitRuntimeConfig(backend_name, shots, instance=$IBM_INSTANCE_CRN)` with
    every other setting the kit's default, as the router's worker builds it.
    The token is not passed: the kit reads QISKIT_IBM_TOKEN itself, and this
    script never reads it into a config, prints it or reports it. The kit IBM
    adapter is imported here, never at module import, like the AWS SDK.
    """
    from eigenkit.core.backends.qiskit_runtime_backend import (
        QiskitRuntimeConfig,
    )

    instance = os.environ[IBM_INSTANCE_ENV].strip()
    return _run_leg(
        "own",
        cases,
        lambda shots: QiskitRuntimeConfig(
            backend_name=device.backend_name, shots=shots, instance=instance
        ),
    )


def run_own_leg(cases: Sequence[Case], device: Device | None = None) -> list[Any]:
    """Every case on `device` through the caller's own vendor credentials.

    An IBM device goes to `run_ibm_own_leg`. A Braket device runs on its ARN:
    the AWS SDK is imported here, never at module import, so `--router-only`
    and the offline tests run without it. Credentials come from the standard
    boto3 chain (AWS_PROFILE); the script never reads or prints them. A QPU
    runs in its own region, so AWS_REGION and AWS_DEFAULT_REGION are set to
    it; a simulator keeps the caller's region (us-east-1 when none is set).
    """
    device = device if device is not None else DEVICES[ROUTER_DEVICE]
    if device.is_ibm:
        return run_ibm_own_leg(cases, device)

    from eigenkit.configs import BraketConfig

    if device.is_qpu:
        for var in ("AWS_REGION", "AWS_DEFAULT_REGION"):
            current = os.environ.get(var)
            if current and current != device.region:
                print(f"note: {var} is {current}; using {device.region} for {device.id}")
            os.environ[var] = device.region
    elif not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        print(f"note: AWS_REGION is not set; using {OWN_REGION} for the own leg")
        os.environ["AWS_DEFAULT_REGION"] = OWN_REGION
    return _run_leg(
        "own", cases, lambda shots: BraketConfig(device_arn=device.arn, shots=shots)
    )


def run_submit_check(
    circuit: ParityCircuit,
    config: RouterConfig,
    *,
    kind: str = "simulator",
    expected_source: str | None = None,
) -> dict[str, Any]:
    """One `submit()` through the router: job_id(), status(), result(), status().

    The result is compared with the exact reference at `config.shots`, under
    the rule for `kind`. With `expected_source` (an IBM run) the job's
    `credential_source` must equal it.
    """
    record: dict[str, Any] = {
        "circuit": circuit.name,
        "shots": config.shots,
        "job_id": None,
        "status_before_result": None,
        "status_after_result": None,
        "value": None,
        "shape": None,
        "dtype": None,
        "credential_source": None,
        "ibm_usage_seconds": None,
        "passed": False,
        "reason": "",
    }
    kit_circuit = None
    try:
        kit_circuit = Circuit(circuit.spec, circuit.observables, provider=config)
        handle = kit_circuit.submit(circuit.weights, circuit.inputs)
        record["job_id"] = handle.job_id()
        record["status_before_result"] = handle.status()
        value = np.asarray(handle.result())
        record["status_after_result"] = handle.status()
        record["value"] = value.tolist()
        record["shape"] = list(value.shape)
        record["dtype"] = str(value.dtype)
        record["credential_source"] = getattr(handle, "credential_source", None)
        record["ibm_usage_seconds"] = getattr(handle, "ibm_usage_seconds", None)
        case = Case(circuit, "run", config.shots)
        check = compare(
            exact_value(case), value, case_tolerance(case, DEFAULT_K, 1, kind=kind)
        )
        record["passed"] = check["passed"] and record["status_after_result"] == "completed"
        record["reason"] = check["reason"]
        source_problem = credential_problem(record, expected_source)
        if source_problem is not None:
            record["passed"] = False
            record["reason"] = "; ".join(r for r in (record["reason"], source_problem) if r)
    except Exception as error:  # recorded, never raised: the report says what broke
        record["reason"] = _redact(f"{type(error).__name__}: {error}")
    finally:
        if kit_circuit is not None:
            _close(kit_circuit)
    return record


def credential_problem(job: dict[str, Any] | None, expected: str | None) -> str | None:
    """Why a router job did not run on the expected IBM account, or None.

    `expected` None (not an IBM run) checks nothing.
    """
    if expected is None or job is None:
        return None
    source = job.get("credential_source")
    if source == expected:
        return None
    return (
        f"router job {job.get('job_id')} ran with credential_source {source!r}, "
        f"expected {expected!r}: the router must run it on the IBM account you connected"
    )


def apply_credential_check(
    results: list[dict[str, Any]],
    jobs: Sequence[dict[str, Any] | None],
    expected: str,
) -> None:
    """Record each router job's IBM fields on its case and fail the case when the
    job did not report the `expected` credential_source."""
    for result, job in zip(results, jobs, strict=True):
        job = job or {}
        result["router_job_id"] = job.get("job_id")
        result["credential_source"] = job.get("credential_source")
        result["ibm_usage_seconds"] = job.get("ibm_usage_seconds")
        problem = credential_problem(job or None, expected)
        if problem is None:
            continue
        result["passed"] = False
        result["reason"] = "; ".join(r for r in (result.get("reason"), problem) if r)
        result["report"] = f"FAIL {result['case']}: {result['reason']}"


# ---- command line ------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be an int >= 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="live_parity.py",
        description=(
            "Compare a Braket or IBM Quantum device through your own credentials with "
            "the same device through EigenOrbit Router."
        ),
    )
    parser.add_argument(
        "--device",
        default=ROUTER_DEVICE,
        help=(
            f"router device id, one of {', '.join(DEVICES)} or ibm/<backend> (seeded: "
            f"{', '.join(IBM_SEEDED)}) (default: {ROUTER_DEVICE})"
        ),
    )
    parser.add_argument(
        "--circuits",
        choices=[*_CIRCUIT_GROUPS, *_CIRCUITS],
        default="all",
        help=(
            "all: ry, ansatz2 and cnot3; garnet: ansatz2 and mc3 (MCZ, MCX and CRY on "
            "3 qubits), the set for IQM Garnet; or one circuit (default: all)"
        ),
    )
    parser.add_argument(
        "--shots",
        type=_positive_int,
        default=None,
        help=(
            f"N for the sampled cases (default: {IBM_DEFAULT_SHOTS} on an IBM device, "
            f"{DEFAULT_SHOTS} otherwise)"
        ),
    )
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="sigmas allowed")
    parser.add_argument(
        "--router-only",
        action="store_true",
        help="no own-credentials leg: compare the router with the exact reference",
    )
    parser.add_argument(
        "--i-accept-cost",
        action="store_true",
        help=(
            "required for a QPU: submit the paid jobs after printing the router's "
            "estimate (on IBM, the jobs that spend the instance's quantum seconds)"
        ),
    )
    parser.add_argument("--report", metavar="PATH", help="write a JSON report here")
    return parser


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return f"{float(array):.6f}"
    return "[" + ", ".join(f"{x:.6f}" for x in array.ravel()) + "]"


def _print_table(results: list[dict[str, Any]], reference: str) -> None:
    header = f"{'case':<40} {reference:>26} {'router':>26} {'max|diff|':>10} {'result':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        diff = "-" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.2e}"
        print(
            f"{r['case']:<40} {_fmt(r['own']):>26} {_fmt(r['router']):>26} "
            f"{diff:>10} {'PASS' if r['passed'] else 'FAIL':>6}"
        )
        if not r["passed"]:
            print(f"    {r['reason']}")


def _versions() -> tuple[str, str]:
    try:
        kit = metadata.version("eigenkit")
    except metadata.PackageNotFoundError:
        kit = "unknown"
    return kit, eigenorbit_router_client.__version__


_IBM_DEVICE = re.compile(r"ibm/[a-z0-9_-]+")


def _select_device(args: argparse.Namespace) -> Device | None:
    """The --device entry, or None after printing why it cannot be used.

    Fills in `args.shots` when it was not given.
    """
    device = DEVICES.get(args.device)
    if device is None and _IBM_DEVICE.fullmatch(args.device or ""):
        device = ibm_device(args.device)
    if device is None:
        print(
            f"error: unknown --device {args.device!r}; supported: {', '.join(DEVICES)} "
            f"and ibm/<backend> (seeded: {', '.join(IBM_SEEDED)})",
            file=sys.stderr,
        )
        return None
    if args.shots is None:
        args.shots = IBM_DEFAULT_SHOTS if device.is_ibm else DEFAULT_SHOTS
    if not device.min_shots <= args.shots <= device.max_shots:
        print(
            f"error: --shots {args.shots} is outside {device.id}'s range "
            f"{device.min_shots}..{device.max_shots}",
            file=sys.stderr,
        )
        return None
    return device


def _cost_gate(
    device: Device,
    cases: Sequence[Case],
    config: RouterConfig,
    args: argparse.Namespace,
) -> tuple[int | None, dict[str, Any] | None]:
    """For a QPU: print the router's estimate and refuse unless the cost is accepted.

    Returns (exit status or None to go on, the estimate or None for a simulator).
    """
    if not device.is_qpu:
        return None, None
    if device.is_ibm:
        print(IBM_COST_STATEMENT)
    try:
        estimate = estimate_cases(cases, config)
    except RouterError as error:
        print(f"error: the router's estimate failed: {_redact(str(error))}", file=sys.stderr)
        return 2, None
    _print_estimate(estimate, device, router_only=args.router_only)
    refusal = _cost_refusal(estimate)
    if refusal is not None:
        print(f"error: {refusal}; nothing was submitted", file=sys.stderr)
        return 2, estimate
    if not args.i_accept_cost:
        cost = (
            "spends quantum seconds of the IBM instance each leg runs on (above)"
            if device.is_ibm
            else "costs real money (the estimate is above)"
        )
        print(
            f"error: {device.id} is a QPU and this run {cost}. Nothing was submitted; "
            f"rerun with --i-accept-cost to run it.",
            file=sys.stderr,
        )
        return EXIT_COST_NOT_ACCEPTED, estimate
    return None, estimate


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device = _select_device(args)
    if device is None:
        return 2

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
        config = RouterConfig(device=device.id, base_url=url)
    except ValueError as error:
        print(f"error: {URL_ENV}: {error}", file=sys.stderr)
        return 2

    if device.is_ibm and not args.router_only:
        problem = ibm_own_env_problem()
        if problem is not None:
            print(f"error: {problem}", file=sys.stderr)
            return 2
    expected_source = IBM_CREDENTIAL_SOURCE if device.is_ibm else None

    circuits = select_circuits(args.circuits)
    cases = cases_for(device, circuits, args.shots)
    mode = "router-only" if args.router_only else "parity"
    rule = acceptance_rule(device, args.k)
    print(
        f"live parity ({mode}) on {device.id}: {len(cases)} cases, shots {args.shots}, "
        f"k {args.k}"
    )
    if expected_source is not None:
        print(
            f"IBM: every router job must report credential_source {expected_source!r} "
            f"(the IBM account you connected)"
        )
    print(f"acceptance rule: {rule}")

    estimated = list(cases)
    if args.router_only:
        estimated.append(Case(circuits[0], "run", args.shots))  # the submit check
    status, estimate = _cost_gate(device, estimated, config, args)
    if status is not None:
        return status

    with warnings.catch_warnings():
        # One CostWarning per router compile would repeat the same estimate.
        warnings.simplefilter("once")
        if args.router_only:
            reference_values = [exact_value(case) for case in cases]
            legs, reference = 1, "exact"
        else:
            reference_values = run_own_leg(cases, device)
            legs, reference = 2, "own"
        jobs: list[dict[str, Any] | None] | None = [] if device.is_ibm else None
        router_values = run_router_leg(cases, config, jobs=jobs)

        results = []
        for case, own, got in zip(cases, reference_values, router_values):
            tol = case_tolerance(case, args.k, legs, kind=device.kind)
            failure = next((v for v in (own, got) if isinstance(v, LegFailure)), None)
            exact = exact_value(case)
            if failure is not None:
                result = _failed(case, failure, tol)
            elif case.method in DISTRIBUTION_METHODS:
                result = compare_distribution(own, got, tol, case=case, exact=exact)
            else:
                result = compare(own, got, tol, case=case.label)
            result["exact"] = _jsonable(exact)
            results.append(result)
        if expected_source is not None:
            apply_credential_check(results, jobs, expected_source)

        submit_check = None
        if args.router_only:
            submit_check = run_submit_check(
                circuits[0],
                dataclasses.replace(config, shots=args.shots),
                kind=device.kind,
                expected_source=expected_source,
            )

    _print_table(results, reference)
    passed = all(r["passed"] for r in results)
    if submit_check is not None:
        print(
            f"submit check: job {submit_check['job_id']} "
            f"{submit_check['status_before_result']} -> "
            f"{submit_check['status_after_result']}: "
            f"{'PASS' if submit_check['passed'] else 'FAIL ' + submit_check['reason']}"
        )
        passed = passed and submit_check["passed"]
    n_passed = sum(r["passed"] for r in results)
    print(f"{'PASS' if passed else 'FAIL'}: {n_passed}/{len(results)} cases")

    if args.report:
        kit_version, client_version = _versions()
        if args.router_only:
            reference_text = "exact kit numpy simulator"
        elif device.is_ibm:
            # The instance CRN and the token stay out of the report.
            reference_text = (
                f"own IBM credentials, QiskitRuntimeConfig(backend_name="
                f"{device.backend_name!r}, instance=${IBM_INSTANCE_ENV}, token from "
                f"${IBM_TOKEN_ENV})"
            )
        else:
            reference_text = f"own AWS credentials, BraketConfig(device_arn={device.arn!r})"
        report = {
            "tool": "scripts/live_parity.py",
            "mode": mode,
            "reference": reference_text,
            "kit_version": kit_version,
            "client_version": client_version,
            "router_host": urlsplit(url).hostname,
            "router_device": device.id,
            "device_kind": device.kind,
            "own_device_arn": device.arn or None,
            "own_region": device.region or None,
            "expected_credential_source": expected_source,
            "acceptance_rule": rule,
            "qpu_abs_tol": QPU_ABS_TOL,
            "estimate": estimate,
            "shots": args.shots,
            "k": args.k,
            "cases": results,
            "submit_check": submit_check,
            "passed": passed,
        }
        text = _redact(json.dumps(report, indent=2))
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"report written to {args.report}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
