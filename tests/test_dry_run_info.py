"""`RouterCompiled.dry_run_info()`: the IBM adapter's keys plus the estimate.

The IBM adapter (kit qiskit_runtime_backend.py) returns 13 keys. The router
path keeps every one of them and adds the four estimate fields of contract
section 10. It never sends a request: the estimate is the one cached at
compile.
"""

import warnings

import httpx
import numpy as np
import pytest
from eigenkit import Circuit, CostWarning
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z

from eigenorbit_router_client import RouterConfig
from fake_router import VALID_KEY


IBM_KEYS = (
    "backend_name",
    "is_fake",
    "primitive",
    "shots",
    "optimization_level",
    "seed_transpiler",
    "n_logical_qubits",
    "n_physical_qubits",
    "transpiled_gate_counts",
    "transpiled_depth",
    "n_observables",
    "n_measurement_groups",
    "n_circuit_runs_per_call",
)
ESTIMATE_KEYS = (
    "estimate_usd_micro",
    "price_version",
    "available_usd_micro",
    "would_reject_reason",
)
# Present when the router's estimate carries them.
BREAKDOWN_KEYS = ("vendor_cost_usd_micro", "fee_usd_micro")
# H(0), CNOT(0,1), CNOT(1,2), RX(w0) on 0: greedy layers 1, 2, 3, 3 -> depth 3.
SPEC = CircuitSpec(
    n_qubits=3,
    n_weights=1,
    n_inputs=0,
    gates=(
        Gate("H", (0,)),
        Gate("CNOT", (0, 1)),
        Gate("CNOT", (1, 2)),
        Gate("RX", (0,), (WeightRef(0),)),
    ),
)
OBSERVABLES = all_z(3)
BASE_URL = "https://router.test"
LEDGER_UNAVAILABLE = {
    "error": {"code": "ledger_unavailable", "message": "Try again", "details": {}}
}


def _config(**overrides) -> RouterConfig:
    settings = {
        "device": "braket/sv1",
        "api_key": VALID_KEY,
        "base_url": BASE_URL,
        "shots": 1000,
    }
    settings.update(overrides)
    return RouterConfig(**settings)


def _compiled(spec=SPEC, observables=OBSERVABLES, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return Circuit(spec, observables, provider=_config(**overrides)).compiled


def test_the_keys_are_the_ibm_keys_plus_the_estimate(fake_router):
    info = _compiled().dry_run_info()

    assert len(IBM_KEYS) == 13
    assert set(info) == set(IBM_KEYS) | set(ESTIMATE_KEYS) | set(BREAKDOWN_KEYS)


def test_the_breakdown_keys_equal_the_estimate_out(fake_router):
    compiled = _compiled()

    info = compiled.dry_run_info()

    assert {key: info[key] for key in BREAKDOWN_KEYS} == {
        "vendor_cost_usd_micro": 75_000,
        "fee_usd_micro": 15_000,
    }
    assert info["vendor_cost_usd_micro"] + info["fee_usd_micro"] == info[
        "estimate_usd_micro"
    ]


def test_an_older_routers_estimate_adds_no_breakdown_keys(fake_router):
    fake_router.estimate_breakdown = False

    info = _compiled().dry_run_info()

    assert set(info) == set(IBM_KEYS) | set(ESTIMATE_KEYS)
    assert info["estimate_usd_micro"] == 90_000


def test_the_ibm_keys_carry_the_router_values(fake_router):
    info = _compiled().dry_run_info()

    assert info["backend_name"] == "braket/sv1"
    assert info["is_fake"] is False
    assert info["shots"] == 1000
    assert info["n_logical_qubits"] == 3
    assert info["n_observables"] == 3
    assert info["n_circuit_runs_per_call"] == 1
    for key in (
        "primitive",
        "optimization_level",
        "seed_transpiler",
        "n_physical_qubits",
        "n_measurement_groups",
    ):
        assert info[key] is None, key


def test_gate_counts_follow_first_appearance_order(fake_router):
    counts = _compiled().dry_run_info()["transpiled_gate_counts"]

    assert list(counts.items()) == [("H", 1), ("CNOT", 2), ("RX", 1)]


def test_depth_is_the_greedy_layer_depth(fake_router):
    assert _compiled().dry_run_info()["transpiled_depth"] == 3


def test_an_empty_circuit_has_depth_0_and_no_counts(fake_router):
    empty = CircuitSpec(n_qubits=1, n_weights=0, n_inputs=0, gates=())

    info = _compiled(spec=empty, observables=all_z(1)).dry_run_info()

    assert info["transpiled_gate_counts"] == {}
    assert info["transpiled_depth"] == 0


def test_the_estimate_keys_equal_the_estimate_out(fake_router):
    compiled = _compiled()

    info = compiled.dry_run_info()

    assert {key: info[key] for key in ESTIMATE_KEYS} == {
        "estimate_usd_micro": 90_000,
        "price_version": "braket/sv1@1",
        "available_usd_micro": 910_000,
        "would_reject_reason": None,
    }
    assert {key: compiled.estimate[key] for key in ESTIMATE_KEYS} == {
        key: info[key] for key in ESTIMATE_KEYS
    }


@pytest.mark.parametrize(
    "failure",
    [{"status": 503, "body": LEDGER_UNAVAILABLE}, {"exc": httpx.ConnectError}],
    ids=["persistent-503", "connect-error"],
)
def test_an_unavailable_estimate_leaves_the_four_keys_none(fake_router, failure):
    fake_router.fail_next("POST", "/v1/jobs/estimate", times=4, **failure)

    info = _compiled().dry_run_info()

    assert set(info) == set(IBM_KEYS) | set(ESTIMATE_KEYS)
    for key in ESTIMATE_KEYS:
        assert info[key] is None, key


def test_dry_run_info_sends_no_request(fake_router):
    compiled = _compiled()
    before = len(fake_router.requests)

    compiled.dry_run_info()
    compiled.dry_run_info()

    assert len(fake_router.requests) == before


def test_a_single_observable_and_no_inputs(fake_router):
    info = _compiled(observables=all_z(3)[:1]).dry_run_info()

    assert info["n_observables"] == 1
    assert info["n_logical_qubits"] == 3


def test_the_result_is_a_fresh_dict(fake_router):
    compiled = _compiled()

    compiled.dry_run_info()["transpiled_gate_counts"]["H"] = 99

    assert compiled.dry_run_info()["transpiled_gate_counts"]["H"] == 1
    assert np.int64 not in {type(v) for v in compiled.dry_run_info().values()}
