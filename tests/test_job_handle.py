"""submit(method=...), RouterJobHandle and the four contract methods.

The far end is the in-process FakeRouter. Request assertions filter by method
and path, so the estimate request a compile sends does not break them.
"""

import numpy as np
import pytest
from eigenkit import Circuit
from eigenkit.core.backends import numpy_backend
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z

import eigenorbit_router_client as client
from eigenorbit_router_client import RouterConfig, RouterError
from fake_router import VALID_KEY


# A 2-qubit RY/CNOT circuit: RY(w0) on 0, RY(w1) on 1, CNOT 0 -> 1.
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
# X on qubit 0 of 2: the state |10>, qubit 0 leftmost.
X_ON_QUBIT_0 = CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),))
OBSERVABLES = all_z(2)
WEIGHTS = np.array([0.7, -1.3])
NO_INPUTS = np.zeros(0)
BASE_URL = "https://router.test"


def _config(**overrides) -> RouterConfig:
    settings = {
        "device": "braket/sv1",
        "api_key": VALID_KEY,
        "base_url": BASE_URL,
        "shots": 1000,
    }
    settings.update(overrides)
    return RouterConfig(**settings)


def _router_circuit(spec=SPEC, **overrides) -> Circuit:
    return Circuit(spec, OBSERVABLES, provider=_config(**overrides))


def _numpy_circuit(spec=SPEC) -> Circuit:
    return Circuit(spec, OBSERVABLES, provider="numpy")


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _submit(router_circuit, method="run"):
    return router_circuit.compiled.submit(WEIGHTS, NO_INPUTS, method=method)


# ---- results through submit ----


def test_submit_run_result_equals_the_numpy_run(fake_router):
    handle = _submit(_router_circuit(), "run")

    got = handle.result()

    want = _numpy_circuit()(WEIGHTS)
    assert isinstance(handle, client.RouterJobHandle)
    assert got.shape == (2,)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_submit_zero_state_probability_result_is_a_0d_float64_array(fake_router):
    handle = _submit(_router_circuit(), "zero_state_probability")

    got = handle.result()

    want = _numpy_circuit().zero_state_probability(WEIGHTS)
    assert isinstance(got, np.ndarray)
    assert got.shape == ()
    assert got.dtype == np.float64
    assert float(got) == pytest.approx(want, abs=1e-12)


def test_submit_probabilities_result_is_4_float64_in_qubit0_msb_order(fake_router):
    handle = _submit(_router_circuit(), "probabilities")

    got = handle.result()

    want = _numpy_circuit().probabilities(WEIGHTS)
    assert got.shape == (4,)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_circuit_probabilities_and_zsp_equal_the_numpy_circuit(fake_router):
    router_circuit = _router_circuit()

    probabilities = router_circuit.probabilities(WEIGHTS)
    zsp = router_circuit.zero_state_probability(WEIGHTS)

    assert probabilities.shape == (4,)
    assert probabilities.dtype == np.float64
    np.testing.assert_allclose(
        probabilities, _numpy_circuit().probabilities(WEIGHTS), rtol=0.0, atol=1e-12
    )
    assert isinstance(zsp, float)
    assert zsp == pytest.approx(_numpy_circuit().zero_state_probability(WEIGHTS), abs=1e-12)


def test_circuit_sample_puts_qubit_0_leftmost(fake_router):
    counts = _router_circuit(X_ON_QUBIT_0, shots=1000).sample(np.zeros(0))

    assert counts == {"10": 1000}
    assert type(counts) is dict
    [submit] = _sent(fake_router, "POST", "/v1/jobs")
    assert b'"method":"sample"' in submit.content.replace(b" ", b"")


# ---- the handle surface ----


def test_handle_reports_job_id_status_billing_and_method(fake_router):
    handle = _submit(_router_circuit(), "probabilities")
    [job_id] = fake_router.jobs

    assert handle.job_id() == job_id
    assert handle.method == "probabilities"
    assert [handle.status(), handle.status(), handle.status()] == [
        "queued",
        "running",
        "completed",
    ]
    assert handle.billing() == "settled"


def test_handle_method_is_read_only(fake_router):
    handle = _submit(_router_circuit(), "run")

    with pytest.raises(AttributeError):
        handle.method = "sample"


def test_a_second_result_call_is_served_from_the_cache(fake_router):
    handle = _submit(_router_circuit(), "run")
    first = handle.result()

    second = handle.result()

    job_id = handle.job_id()
    assert len(_sent(fake_router, "GET", f"/v1/jobs/{job_id}/result")) == 1
    np.testing.assert_array_equal(first, second)


def test_a_completed_job_with_billing_held_returns_its_result_at_once(fake_router):
    fake_router.status_steps = ("completed",)
    fake_router.complete_billing = "held"
    handle = _submit(_router_circuit(), "run")

    got = handle.result()

    assert fake_router.sleeps == []
    np.testing.assert_allclose(got, _numpy_circuit()(WEIGHTS), rtol=0.0, atol=1e-12)
    assert handle.status() == "completed"
    assert handle.billing() == "held"


def test_submit_refuses_a_method_outside_submit_methods(fake_router):
    compiled = _router_circuit().compiled

    with pytest.raises(ValueError, match="submit_methods"):
        compiled.submit(WEIGHTS, NO_INPUTS, method="statevector")

    assert "statevector" not in compiled.submit_methods
    assert compiled.submit_methods == frozenset(
        {"run", "sample", "probabilities", "zero_state_probability"}
    )
    assert _sent(fake_router, "POST", "/v1/jobs") == []


def test_submit_refuses_2d_inputs(fake_router):
    compiled = _router_circuit().compiled

    with pytest.raises(ValueError):
        compiled.submit(np.vstack([WEIGHTS, WEIGHTS]), NO_INPUTS)
    with pytest.raises(ValueError):
        compiled.submit(WEIGHTS, np.zeros((2, 0)))

    assert _sent(fake_router, "POST", "/v1/jobs") == []


@pytest.mark.parametrize("method", ["sample", "probabilities"])
def test_an_unsupported_method_raises_without_a_local_fallback(
    fake_router, monkeypatch, method
):
    fake_router.supported_methods = frozenset({"run", "zero_state_probability"})
    router_circuit = _router_circuit()

    def no_local_simulation(*args, **kwargs):
        raise AssertionError("the client fell back to a local simulation")

    monkeypatch.setattr(numpy_backend.NumpySimulator, "compile", no_local_simulation)

    with pytest.raises(RouterError) as caught:
        getattr(router_circuit, method)(WEIGHTS)

    assert caught.value.code == "method_not_supported"
    assert caught.value.http_status == 422
    [refused] = _sent(fake_router, "POST", "/v1/jobs")
    assert refused.headers["Idempotency-Key"]
    assert fake_router.jobs == {}


def test_a_lost_submit_is_recovered_with_its_idempotency_key(fake_router):
    # Resubmitting with the key the error names replays the job instead of creating
    # a second one.
    compiled = _router_circuit().compiled
    fake_router.drop_response_after_create(times=4)
    with pytest.raises(RouterError) as caught:
        compiled.submit(WEIGHTS, NO_INPUTS)
    key = caught.value.details["idempotency_key"]
    [job_id] = fake_router.jobs

    handle = compiled.submit(WEIGHTS, NO_INPUTS, idempotency_key=key)

    assert handle.job_id() == job_id
    assert list(fake_router.jobs) == [job_id]
    submits = _sent(fake_router, "POST", "/v1/jobs")
    assert {r.headers["Idempotency-Key"] for r in submits} == {key}
    np.testing.assert_allclose(
        handle.result(), _numpy_circuit()(WEIGHTS), rtol=0.0, atol=1e-12
    )


def test_submit_without_a_key_mints_a_fresh_one_per_call(fake_router):
    compiled = _router_circuit().compiled

    compiled.submit(WEIGHTS, NO_INPUTS)
    compiled.submit(WEIGHTS, NO_INPUTS)

    keys = [r.headers["Idempotency-Key"] for r in _sent(fake_router, "POST", "/v1/jobs")]
    assert len(set(keys)) == 2
    assert len(fake_router.jobs) == 2


def test_cancel_of_a_job_not_at_the_vendor_returns_cancelled(fake_router):
    handle = _submit(_router_circuit(), "run")
    job_id = handle.job_id()

    status = handle.cancel()

    assert status == "cancelled"
    assert len(_sent(fake_router, "POST", f"/v1/jobs/{job_id}/cancel")) == 1
    with pytest.raises(RouterError) as caught:
        handle.result()
    assert caught.value.code == "job_not_completed"
    assert caught.value.details["status"] == "cancelled"
    assert caught.value.details["billing"] == "released"
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_cancel_at_the_vendor_keeps_the_status_until_the_vendor_cancels(fake_router):
    fake_router.cancel_mode = "vendor"
    handle = _submit(_router_circuit(), "run")
    assert handle.status() == "queued"

    status = handle.cancel()

    assert status == "queued"
    assert handle.status() == "cancelled"
    assert handle.billing() == "released"
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_a_job_the_vendor_finishes_first_still_returns_its_result(fake_router):
    fake_router.cancel_mode = "vendor_finishes"
    handle = _submit(_router_circuit(), "run")

    assert handle.cancel() == "queued"

    got = handle.result()
    np.testing.assert_allclose(got, _numpy_circuit()(WEIGHTS), rtol=0.0, atol=1e-12)
    assert handle.billing() == "settled"


def test_a_second_cancel_answers_202_again(fake_router):
    handle = _submit(_router_circuit(), "run")

    assert handle.cancel() == "cancelled"
    assert handle.cancel() == "cancelled"
    assert len(_sent(fake_router, "POST", f"/v1/jobs/{handle.job_id()}/cancel")) == 2


def test_cancel_of_a_completed_job_raises_job_not_cancellable(fake_router):
    fake_router.status_steps = ("completed",)
    handle = _submit(_router_circuit(), "run")
    assert handle.status() == "completed"

    with pytest.raises(RouterError) as caught:
        handle.cancel()

    err = caught.value
    assert (err.code, err.http_status) == ("job_not_cancellable", 409)
    assert err.details == {"status": "completed"}
    assert err.job_id == handle.job_id()
    assert "already completed" in err.message
    assert len(_sent(fake_router, "POST", f"/v1/jobs/{handle.job_id()}/cancel")) == 1
    assert fake_router.sleeps == []
    # The job is untouched and its result is still served.
    np.testing.assert_allclose(
        handle.result(), _numpy_circuit()(WEIGHTS), rtol=0.0, atol=1e-12
    )


def test_neither_a_poll_timeout_nor_ctrl_c_cancels_the_job(fake_router, monkeypatch):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(client._client, "_sleep", sleep)
    monkeypatch.setattr(client._client, "_monotonic", lambda: now[0])
    fake_router.status_steps = ("queued", "running")
    handle = _submit(_router_circuit(poll_timeout_seconds=10), "run")

    with pytest.raises(RouterError) as timed_out:
        handle.result()

    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(client._client, "_sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        handle.result()

    assert timed_out.value.code == "poll_timeout"
    assert _sent(fake_router, "POST", f"/v1/jobs/{handle.job_id()}/cancel") == []
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_cancel_surfaces_an_older_routers_501(fake_router):
    handle = _submit(_router_circuit(), "run")
    job_id = handle.job_id()
    fake_router.fail_next(
        "POST",
        f"/v1/jobs/{job_id}/cancel",
        status=501,
        body={"error": {"code": "not_implemented", "message": "not yet", "details": {}}},
    )

    with pytest.raises(RouterError) as caught:
        handle.cancel()

    assert len(_sent(fake_router, "POST", f"/v1/jobs/{job_id}/cancel")) == 1
    assert caught.value.code == "not_implemented"
    assert caught.value.http_status == 501
    assert caught.value.job_id == job_id


def test_a_failed_job_raises_router_error_from_result(fake_router):
    fake_router.fail_with = "vendor_failed"
    handle = _submit(_router_circuit(), "run")

    with pytest.raises(RouterError) as caught:
        handle.result()

    assert caught.value.code == "vendor_failed"
    assert isinstance(caught.value, RuntimeError)
    assert caught.value.job_id == handle.job_id()
    assert caught.value.http_status is None
    assert caught.value.details == {"status": "failed", "billing": "released"}
    assert _sent(fake_router, "GET", f"/v1/jobs/{handle.job_id()}/result") == []


def test_blocking_run_goes_through_submit_with_one_key_per_call(fake_router):
    router_circuit = _router_circuit()

    router_circuit(WEIGHTS)
    router_circuit(WEIGHTS)

    keys = [r.headers["Idempotency-Key"] for r in _sent(fake_router, "POST", "/v1/jobs")]
    assert len(keys) == 2
    assert keys[0] != keys[1]


# ---- whose IBM account ran the job: the user's own (`own`) or an IBM account
# EigenOrbit provides (`organization`) ----


def test_an_ibm_handle_reports_the_organization_source_and_its_usage(fake_router):
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "sample")

    handle.result()

    assert handle.credential_source == "organization"
    assert handle.ibm_usage_seconds == 2.315
    assert handle.billing() is None


def test_an_ibm_handle_reports_the_users_own_account(fake_router):
    fake_router.credential_source = "own"
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "probabilities")

    handle.result()

    assert handle.credential_source == "own"


def test_the_usage_is_none_until_ibm_reports_it(fake_router):
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "run")

    assert handle.credential_source == "organization"
    assert handle.ibm_usage_seconds is None
    assert handle.status() == "queued"
    assert handle.ibm_usage_seconds is None


def test_the_fields_follow_the_last_refresh(fake_router):
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "run")

    statuses = [handle.status() for _ in range(3)]

    assert statuses[-1] == "completed"
    assert handle.ibm_usage_seconds == 2.315


def test_a_braket_handle_has_no_credential_source_or_usage(fake_router):
    handle = _submit(_router_circuit(), "run")

    handle.result()

    assert handle.credential_source is None
    assert handle.ibm_usage_seconds is None


def test_a_router_without_the_ibm_fields_leaves_both_none(fake_router):
    fake_router.ibm_job_fields = False
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "run")

    handle.result()

    assert handle.credential_source is None
    assert handle.ibm_usage_seconds is None


@pytest.mark.parametrize(
    ("source", "usage"),
    [("someone", True), (7, "2.3"), (None, None)],
)
def test_malformed_fields_read_as_none(fake_router, source, usage):
    fake_router.credential_source = source
    fake_router.ibm_usage_seconds = usage
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "run")

    handle.result()

    assert handle.credential_source is None
    assert handle.ibm_usage_seconds is None


def test_an_integer_usage_reads_as_float_seconds(fake_router):
    fake_router.ibm_usage_seconds = 3
    handle = _submit(_router_circuit(device="ibm/ibm_fez", shots=100), "run")

    handle.result()

    assert handle.ibm_usage_seconds == 3.0
    assert isinstance(handle.ibm_usage_seconds, float)
