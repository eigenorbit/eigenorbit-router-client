"""One router estimate per compile, shown through one CostWarning.

The estimate is requested once, when the kit `Circuit` compiles, and cached on
the compiled circuit. The warning must point at the line that built the
Circuit, show the router's amounts converted exactly from integer micro-USD,
and never present a batch or a parameter-shift gradient as one paid job.
"""

import json
import warnings

import httpx
import numpy as np
import pytest
from eigenkit import Circuit, CostWarning
from eigenkit.core.circuit import Circuit as CoreCircuit
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z

from eigenorbit_router_client import RouterConfig, RouterError
from fake_router import VALID_KEY


SPEC = CircuitSpec(
    n_qubits=2,
    n_weights=2,
    n_inputs=0,
    gates=(
        Gate("RY", (0,), (WeightRef(0),)),
        Gate("RY", (1,), (WeightRef(1),)),
        Gate("CNOT", (0, 1)),
    ),
)
OBSERVABLES = all_z(2)
WEIGHTS = np.array([0.7, -1.3])
BASE_URL = "https://router.test"
LEDGER_UNAVAILABLE = {
    "error": {"code": "ledger_unavailable", "message": "Try again", "details": {}}
}


def _config(**overrides) -> RouterConfig:
    settings = {"device": "braket/sv1", "api_key": VALID_KEY, "base_url": BASE_URL}
    settings.update(overrides)
    return RouterConfig(**settings)


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _compile(**overrides):
    """Build the Circuit and return it with the CostWarnings it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit = Circuit(SPEC, OBSERVABLES, provider=_config(**overrides))
    return circuit, [w for w in caught if issubclass(w.category, CostWarning)]


def _message(**overrides) -> str:
    _, [warning] = _compile(**overrides)
    return str(warning.message)


# ---- one estimate per compile ----


def test_compile_sends_exactly_one_estimate_of_a_zero_run(fake_router):
    circuit, _ = _compile()

    [estimate] = _sent(fake_router, "POST", "/v1/jobs/estimate")
    want = circuit.compiled.build_request("run", np.zeros(2), np.zeros(0))
    assert json.loads(estimate.content) == want
    assert json.loads(estimate.content)["inputs"] == []
    assert _sent(fake_router, "POST", "/v1/jobs") == []


def test_later_calls_send_no_further_estimate(fake_router):
    circuit, _ = _compile()

    for _ in range(3):
        circuit(WEIGHTS)
    circuit.compiled.dry_run_info()

    assert len(_sent(fake_router, "POST", "/v1/jobs/estimate")) == 1
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 3


def test_the_estimate_is_cached_on_the_compiled_circuit(fake_router):
    circuit, _ = _compile()

    assert circuit.compiled.estimate == {
        "device": "braket/sv1",
        "method": "run",
        "shots": 0,
        "estimate_usd_micro": 90_000,
        "vendor_cost_usd_micro": 75_000,
        "fee_usd_micro": 15_000,
        "price_version": "braket/sv1@1",
        "available_usd_micro": 910_000,
        "would_reject_reason": None,
    }


# ---- the warning ----


@pytest.mark.parametrize("circuit_class", [Circuit, CoreCircuit], ids=["kit", "core"])
def test_exactly_one_cost_warning_points_at_the_caller(fake_router, circuit_class):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit_class(SPEC, OBSERVABLES, provider=_config())  # the line the warning names
    cost = [w for w in caught if issubclass(w.category, CostWarning)]

    assert len(cost) == 1
    assert cost[0].filename == __file__
    with open(__file__) as source:
        line = source.read().splitlines()[cost[0].lineno - 1]
    assert "the line the warning names" in line


def test_the_warning_shows_price_price_version_and_balance(fake_router):
    message = _message()

    assert "braket/sv1" in message
    assert "estimated $0.09 (vendor $0.075 + EigenOrbit fee $0.015)" in message
    assert "estimate_usd_micro=90000" in message
    assert "braket/sv1@1" in message
    assert "$0.910000" in message
    assert "dry_run_info()" in message
    assert "would reject" not in message


def test_a_would_reject_reason_is_shown(fake_router):
    fake_router.available_usd_micro = 50_000

    message = _message()

    assert "insufficient_balance" in message
    assert "would reject" in message


def test_a_forced_would_reject_reason_is_shown(fake_router):
    fake_router.would_reject_reason = "negative_balance"

    assert "negative_balance" in _message()


def test_a_simulator_warns_that_gradients_and_batch_rows_are_paid_jobs(fake_router):
    message = _message()

    assert "each gradient evaluation" in message
    assert "each row of a 2-D batch" in message
    assert "separate paid router job" in message


def test_a_qpu_has_no_gradient_sentence_but_keeps_the_batch_note(fake_router):
    message = _message(device="braket/ionq_forte", shots=100)

    assert "gradient" not in message
    assert "each row of a 2-D batch" in message
    assert "separate paid router job" in message


def test_a_qpu_warning_shows_the_vendor_and_fee_breakdown(fake_router):
    # contract section 10: braket/iqm_garnet@1 at 10 shots is 377400 = 314500 + 62900.
    fake_router.devices = fake_router.devices | {"braket/iqm_garnet"}
    fake_router.estimate_usd_micro = 377_400

    message = _message(device="braket/iqm_garnet", shots=10)

    assert "estimated $0.3774 (vendor $0.3145 + EigenOrbit fee $0.0629)" in message
    assert "estimate_usd_micro=377400" in message
    assert "braket/iqm_garnet@1" in message


def test_an_estimate_without_the_breakdown_shows_the_total_only(fake_router):
    # The router sends no vendor/fee split: the warning names the total alone.
    fake_router.estimate_breakdown = False

    message = _message()

    assert (
        "$0.090000 per job (estimate_usd_micro=90000, price_version braket/sv1@1)"
        in message
    )
    assert "vendor" not in message
    assert "EigenOrbit fee" not in message


@pytest.mark.parametrize(
    ("micro", "vendor", "shown"),
    [
        (377_400, 314_500, "estimated $0.3774 (vendor $0.3145 + EigenOrbit fee $0.0629)"),
        (1, 0, "estimated $0.000001 (vendor $0.00 + EigenOrbit fee $0.000001)"),
        (123_456_789, 100_000_000, "estimated $123.456789 (vendor $100.00 + "),
    ],
)
def test_the_breakdown_amounts_are_exact(fake_router, micro, vendor, shown):
    fake_router.estimate_usd_micro = micro
    fake_router.estimate_vendor_cost_usd_micro = vendor
    fake_router.available_usd_micro = 10**12

    assert shown in _message()


@pytest.mark.parametrize(
    "breakdown",
    [
        {"vendor_cost_usd_micro": 70_000, "fee_usd_micro": 15_000},
        {"vendor_cost_usd_micro": 0.075, "fee_usd_micro": 15_000},
        {"vendor_cost_usd_micro": 75_000, "fee_usd_micro": None},
    ],
    ids=["does-not-sum", "float", "half-missing"],
)
def test_a_malformed_breakdown_is_refused(fake_router, breakdown):
    body = {
        "device": "braket/sv1",
        "method": "run",
        "shots": 0,
        "estimate_usd_micro": 90_000,
        "price_version": "braket/sv1@1",
        "available_usd_micro": 910_000,
        "would_reject_reason": None,
        **breakdown,
    }
    fake_router.fail_next("POST", "/v1/jobs/estimate", status=200, body=body)

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=_config())

    assert caught.value.code == "invalid_response"


@pytest.mark.parametrize(
    ("micro", "shown"),
    [(1, "$0.000001"), (123_456_789, "$123.456789"), (0, "$0.000000")],
)
def test_amounts_are_exact_conversions_of_micro_usd(fake_router, micro, shown):
    fake_router.estimate_breakdown = False
    fake_router.estimate_usd_micro = micro
    fake_router.available_usd_micro = 10**12

    message = _message()

    assert f"{shown} per job" in message
    assert f"estimate_usd_micro={micro}" in message
    assert "$1000000.000000" in message


# ---- estimate failures ----


@pytest.mark.parametrize(
    ("overrides", "knobs", "code", "status"),
    [
        ({"api_key": "eo_test_short"}, {}, "unauthenticated", 401),
        ({}, {"protocol_min": 2, "protocol_max": 2}, "protocol_unsupported", 426),
        ({"device": "braket/nope"}, {}, "unknown_device", 422),
    ],
)
def test_a_misconfigured_estimate_raises_at_compile(
    fake_router, overrides, knobs, code, status
):
    for name, value in knobs.items():
        setattr(fake_router, name, value)

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=_config(**overrides))

    assert caught.value.code == code
    assert caught.value.http_status == status
    assert len(_sent(fake_router, "POST", "/v1/jobs/estimate")) == 1
    assert fake_router.jobs == {}


@pytest.mark.parametrize(
    "failure",
    [
        {"status": 503, "body": LEDGER_UNAVAILABLE},
        {"exc": httpx.ConnectError},
    ],
    ids=["persistent-503", "connect-error"],
)
def test_an_unavailable_estimate_still_compiles_and_says_so(fake_router, failure):
    fake_router.fail_next("POST", "/v1/jobs/estimate", times=4, **failure)

    circuit, [warning] = _compile()

    message = str(warning.message)
    assert "estimate is unavailable" in message
    assert "hold" in message
    assert "$" not in message
    assert circuit.compiled.estimate is None
    assert len(_sent(fake_router, "POST", "/v1/jobs/estimate")) == 4
    got = circuit(WEIGHTS)
    want = Circuit(SPEC, OBSERVABLES, provider="numpy")(WEIGHTS)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_an_unavailable_estimate_names_its_code(fake_router):
    fake_router.fail_next("POST", "/v1/jobs/estimate", times=4, exc=httpx.ConnectError)

    assert "router_unreachable" in _message()


def test_an_estimate_with_non_integer_amounts_is_refused(fake_router):
    fake_router.estimate_usd_micro = 0.09

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=_config())

    assert caught.value.code == "invalid_response"


def test_a_key_not_set_yet_compiles_and_the_warning_says_why(fake_router):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        circuit = Circuit(
            SPEC, OBSERVABLES, provider=RouterConfig("braket/sv1", base_url=BASE_URL)
        )
    [warning] = [w for w in caught if issubclass(w.category, CostWarning)]

    assert "estimate is unavailable (unauthenticated)" in str(warning.message)
    assert circuit.compiled.estimate is None
    assert fake_router.requests == []


@pytest.mark.parametrize("status", [502, 504])
def test_a_persistent_gateway_error_is_also_unavailable(fake_router, status):
    fake_router.fail_next("POST", "/v1/jobs/estimate", status=status, body="bad", times=4)

    circuit, [warning] = _compile()

    assert "estimate is unavailable" in str(warning.message)
    assert circuit.compiled.estimate is None


def test_a_submit_failure_injection_does_not_hit_the_estimate(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", status=503, body=LEDGER_UNAVAILABLE)

    circuit, [warning] = _compile()

    assert "estimated $0.09 " in str(warning.message)
    circuit(WEIGHTS)
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 2


# ---- IBM devices ----

IBM = {"device": "ibm/ibm_fez", "shots": 100}


def test_an_ibm_warning_says_there_is_no_router_charge_and_ibm_bills(fake_router):
    message = _message(**IBM)

    assert "ibm/ibm_fez" in message
    assert "no router charge" in message
    assert "IBM bills the IBM Quantum instance the job runs on" in message
    assert "quantum time" in message
    assert "EigenOrbit" not in message
    assert "Available balance" not in message
    assert "$" not in message
    assert "gradient" not in message
    assert "dry_run_info()" in message


def test_an_ibm_warning_shows_an_ibm_refusal_the_router_would_give(fake_router):
    fake_router.ibm_refusal = "ibm_account_not_connected"

    message = _message(**IBM)

    assert "would reject" in message
    assert "ibm_account_not_connected" in message


def test_an_ibm_warning_without_an_estimate_still_says_no_router_charge(fake_router):
    fake_router.fail_next("POST", "/v1/jobs/estimate", times=4, exc=httpx.ConnectError)

    message = _message(**IBM)

    assert "no router charge" in message
    assert "router_unreachable" in message
    assert "EigenOrbit balance" not in message


def test_an_ibm_batch_note_says_each_row_spends_quantum_time(fake_router):
    message = _message(**IBM)

    assert "each row of a 2-D batch is a separate IBM job" in message
    assert "paid router job" not in message


def test_dry_run_info_keys_are_the_same_on_ibm_and_braket(fake_router):
    ibm, _ = _compile(**IBM)
    braket, _ = _compile()

    assert list(ibm.compiled.dry_run_info()) == list(braket.compiled.dry_run_info())
    assert ibm.compiled.dry_run_info()["estimate_usd_micro"] == 0
