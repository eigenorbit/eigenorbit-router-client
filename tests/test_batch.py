"""2-D batches fan out into concurrent router jobs.

A batch of N rows makes one fresh estimate, refuses up front when N x the
estimate clearly exceeds the balance, then submits N jobs through a bounded
pool with the Idempotency-Keys "<one call UUID>:<row>". Rows come back in
input order, or the call raises after every row is terminal, listing every
failed row. Nothing is cancelled, and no partial result is returned.
"""

import json
import uuid
import warnings

import numpy as np
import pytest
from eigenkit import Circuit, CostWarning
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.observables import all_z

from eigenorbit_router_client import RouterConfig, RouterError, _client
from fake_router import VALID_KEY


SPEC = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
OBSERVABLES = all_z(2)
BASE_URL = "https://router.test"
TERMINAL = {"completed", "failed", "cancelled", "rejected"}


def _envelope(code, message="refused", details=None):
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _config(**overrides) -> RouterConfig:
    settings = {"device": "braket/sv1", "api_key": VALID_KEY, "base_url": BASE_URL}
    settings.update(overrides)
    return RouterConfig(**settings)


def _router_circuit(**overrides) -> Circuit:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return Circuit(SPEC, OBSERVABLES, provider=_config(**overrides))


def _numpy_circuit() -> Circuit:
    return Circuit(SPEC, OBSERVABLES, provider="numpy")


def _weights(rows=None):
    rng = np.random.default_rng(11)
    shape = (SPEC.n_weights,) if rows is None else (rows, SPEC.n_weights)
    return rng.uniform(-np.pi, np.pi, shape)


def _inputs(rows=None):
    rng = np.random.default_rng(5)
    shape = (SPEC.n_inputs,) if rows is None else (rows, SPEC.n_inputs)
    return rng.uniform(0.0, np.pi, shape)


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _submits(router):
    return _sent(router, "POST", "/v1/jobs")


def _job_ids_by_row(router):
    return {job["row"]: job_id for job_id, job in router.jobs.items()}


@pytest.fixture
def clock(fake_router, monkeypatch):
    """A monotonic clock that only moves when the client sleeps."""
    now = [0.0]

    def sleep(seconds):
        fake_router.sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(_client, "_sleep", sleep)
    monkeypatch.setattr(_client, "_monotonic", lambda: now[0])
    return now


# ---- fan-out and ordering ----


def test_batch_keys_share_one_call_id(fake_router):
    circuit = _router_circuit()
    weights, inputs = _weights(), _inputs(3)
    before = len(fake_router.requests)

    got = circuit(weights, inputs)

    batch = fake_router.requests[before:]
    estimates = [
        i for i, r in enumerate(batch) if r.url.path == "/v1/jobs/estimate"
    ]
    posts = [i for i, r in enumerate(batch) if r.method == "POST" and r.url.path == "/v1/jobs"]
    assert len(estimates) == 1 and len(posts) == 3
    assert estimates[0] < min(posts)
    assert json.loads(batch[estimates[0]].content)["method"] == "run"
    keys = sorted(batch[i].headers["Idempotency-Key"] for i in posts)
    call_id = keys[0].rsplit(":", 1)[0]
    uuid.UUID(call_id)
    assert keys == [f"{call_id}:{row}" for row in range(3)]
    want = _numpy_circuit()(weights, inputs)
    assert got.shape == (3, 2)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_each_row_carries_its_own_parameters(fake_router):
    circuit = _router_circuit()
    weights, inputs = _weights(), _inputs(3)

    circuit(weights, inputs)

    by_row = {
        int(r.headers["Idempotency-Key"].rsplit(":", 1)[1]): json.loads(r.content)
        for r in _submits(fake_router)
    }
    for row in range(3):
        assert by_row[row]["inputs"] == inputs[row].tolist()
        assert by_row[row]["weights"] == weights.tolist()


def test_two_batch_calls_use_two_call_ids(fake_router):
    circuit = _router_circuit()

    circuit(_weights(), _inputs(2))
    circuit(_weights(), _inputs(2))

    call_ids = {r.headers["Idempotency-Key"].rsplit(":", 1)[0] for r in _submits(fake_router)}
    assert len(call_ids) == 2


def test_2d_weights_with_1d_inputs(fake_router):
    weights, inputs = _weights(3), _inputs()

    got = _router_circuit().compiled.run(weights, inputs)

    numpy = _numpy_circuit()
    want = np.stack([numpy(w, inputs) for w in weights])
    assert got.shape == (3, 2)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_both_2d_with_equal_batch_sizes(fake_router):
    weights, inputs = _weights(3), _inputs(3)

    got = _router_circuit().compiled.run(weights, inputs)

    numpy = _numpy_circuit()
    want = np.stack([numpy(w, x) for w, x in zip(weights, inputs)])
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_unequal_batch_sizes_raise_before_any_request(fake_router):
    compiled = _router_circuit().compiled
    before = len(fake_router.requests)

    with pytest.raises(ValueError, match="cannot broadcast"):
        compiled.run(_weights(2), _inputs(3))

    assert len(fake_router.requests) == before


def test_a_row_of_the_wrong_length_raises_before_any_request(fake_router):
    compiled = _router_circuit().compiled
    before = len(fake_router.requests)

    with pytest.raises(ValueError, match="inputs"):
        compiled.run(_weights(), np.zeros((3, SPEC.n_inputs + 1)))

    assert len(fake_router.requests) == before


def test_rows_stay_in_input_order_when_completing_in_reverse(fake_router):
    fake_router.completion_order = "reverse"
    weights, inputs = _weights(), _inputs(4)

    got = _router_circuit()(weights, inputs)

    rows = [fake_router.jobs[job_id]["row"] for job_id in fake_router.terminal_order]
    assert rows == [3, 2, 1, 0]
    np.testing.assert_allclose(
        got, _numpy_circuit()(weights, inputs), rtol=0.0, atol=1e-12
    )


def test_zero_state_probabilities_fans_out(fake_router):
    circuit = _router_circuit(shots=100)
    weights, inputs = _weights(), _inputs(3)
    before = len(fake_router.requests)

    got = circuit.zero_state_probabilities(weights, inputs)

    [estimate] = _sent(fake_router, "POST", "/v1/jobs/estimate")[1:]
    assert json.loads(estimate.content)["method"] == "zero_state_probability"
    methods = {json.loads(r.content)["method"] for r in _submits(fake_router)}
    assert methods == {"zero_state_probability"}
    assert len(fake_router.requests[before:]) > 3
    want = _numpy_circuit().zero_state_probabilities(weights, inputs)
    assert got.shape == (3,)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_zero_state_probabilities_needs_a_batch(fake_router):
    compiled = _router_circuit(shots=100).compiled

    with pytest.raises(ValueError, match="batch"):
        compiled.zero_state_probabilities(_weights(), _inputs())


# ---- balance pre-check ----


def test_precheck_allows_exact_balance(fake_router):
    fake_router.estimate_usd_micro = 90_000
    fake_router.available_usd_micro = 270_000

    got = _router_circuit()(_weights(), _inputs(3))

    assert got.shape == (3, 2)
    assert len(_submits(fake_router)) == 3


def test_precheck_refuses_one_micro_over(fake_router):
    fake_router.estimate_usd_micro = 90_000
    fake_router.available_usd_micro = 269_999
    circuit = _router_circuit()

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    err = caught.value
    assert _submits(fake_router) == []
    assert err.code == "insufficient_balance"
    assert "$0.270000" in err.message
    assert "$0.269999" in err.message
    assert err.details == {
        "required_usd_micro": 270_000,
        "available_usd_micro": 269_999,
        "n_jobs": 3,
    }


def test_precheck_uses_exact_integer_arithmetic(fake_router):
    # 0.1 + 0.2 style float drift would let this through or refuse it wrongly.
    fake_router.estimate_usd_micro = 425
    fake_router.available_usd_micro = 425 * 7 - 1

    with pytest.raises(RouterError) as caught:
        _router_circuit()(_weights(), _inputs(7))

    assert caught.value.details["required_usd_micro"] == 2975
    assert _submits(fake_router) == []


def test_would_reject_method_not_supported_raises_before_any_submit(fake_router):
    fake_router.supported_methods = frozenset({"zero_state_probability"})
    circuit = _router_circuit()

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    assert caught.value.code == "method_not_supported"
    assert _submits(fake_router) == []


@pytest.mark.parametrize("reason", ["shots_invalid", "negative_balance"])
def test_other_would_reject_reasons_raise_before_any_submit(fake_router, reason):
    circuit = _router_circuit()
    fake_router.would_reject_reason = reason

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(2))

    assert caught.value.code == reason
    assert _submits(fake_router) == []


def test_an_unavailable_batch_estimate_skips_the_precheck(fake_router):
    circuit = _router_circuit()
    fake_router.available_usd_micro = 0
    fake_router.fail_next(
        "POST", "/v1/jobs/estimate", status=503, body=_envelope("ledger_unavailable"), times=4
    )

    got = circuit(_weights(), _inputs(3))

    assert got.shape == (3, 2)
    assert len(_submits(fake_router)) == 3


# ---- failures and retries ----


def test_failed_row_raises_after_all_terminal(fake_router):
    # Reverse completion: row 2 completes, row 1 fails, and row 0 is still
    # running at that point; the call must wait for row 0 too.
    fake_router.completion_order = "reverse"
    fake_router.fail_rows = {1: "vendor_failed"}
    circuit = _router_circuit()

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    err = caught.value
    ids = _job_ids_by_row(fake_router)
    assert [fake_router.jobs[ids[row]]["status"] for row in range(3)] == [
        "completed",
        "failed",
        "completed",
    ]
    assert fake_router.terminal_order[-1] == ids[0]
    assert err.code == "vendor_failed"
    assert err.job_id == ids[1]
    assert err.details == {
        "failed": [
            {"row": 1, "job_id": ids[1], "code": "vendor_failed", "status": "failed"}
        ],
        "n_jobs": 3,
    }
    assert ids[1] in err.message
    assert "no partial result" in err.message
    assert [r for r in fake_router.requests if r.url.path.endswith("/result")] == []
    assert [r for r in fake_router.requests if r.url.path.endswith("/cancel")] == []


def test_every_failed_row_is_listed_in_row_order(fake_router):
    fake_router.completion_order = "reverse"
    fake_router.fail_rows = {2: "vendor_failed", 0: "device_offline"}

    with pytest.raises(RouterError) as caught:
        _router_circuit()(_weights(), _inputs(4))

    ids = _job_ids_by_row(fake_router)
    err = caught.value
    assert err.code == "device_offline"
    assert err.job_id == ids[0]
    assert [f["row"] for f in err.details["failed"]] == [0, 2]
    assert [f["job_id"] for f in err.details["failed"]] == [ids[0], ids[2]]
    assert ids[0] in err.message and ids[2] in err.message


def test_a_refused_row_submit_is_listed_after_the_others_finish(fake_router):
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=402,
        body=_envelope("insufficient_balance", "Top up"),
        key_suffix=":1",
    )

    with pytest.raises(RouterError) as caught:
        _router_circuit()(_weights(), _inputs(3))

    err = caught.value
    ids = _job_ids_by_row(fake_router)
    assert sorted(ids) == [0, 2]
    assert all(fake_router.jobs[ids[row]]["status"] == "completed" for row in (0, 2))
    assert err.code == "insufficient_balance"
    assert err.http_status == 402
    assert err.job_id is None
    assert err.details["failed"] == [
        {"row": 1, "job_id": None, "code": "insufficient_balance", "status": None}
    ]


def test_a_503_on_a_row_submit_is_retried_with_the_same_key(fake_router):
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=503,
        body=_envelope("ledger_unavailable"),
        key_suffix=":1",
    )
    weights, inputs = _weights(), _inputs(3)

    got = _router_circuit()(weights, inputs)

    row_1 = [
        r.headers["Idempotency-Key"]
        for r in _submits(fake_router)
        if r.headers["Idempotency-Key"].endswith(":1")
    ]
    assert len(row_1) == 2 and row_1[0] == row_1[1]
    assert len(fake_router.jobs) == 3
    np.testing.assert_allclose(
        got, _numpy_circuit()(weights, inputs), rtol=0.0, atol=1e-12
    )


# ---- concurrency, interrupt, timeout ----


def test_pool_is_bounded(fake_router):
    fake_router.submit_delay_seconds = 0.05
    compiled = _router_circuit(max_concurrent_submits=2).compiled

    got = compiled.run(_weights(), _inputs(5))

    assert got.shape == (5, 2)
    assert fake_router.max_in_flight == 2


def test_a_single_worker_submits_one_at_a_time(fake_router):
    fake_router.submit_delay_seconds = 0.01
    compiled = _router_circuit(max_concurrent_submits=1).compiled

    compiled.run(_weights(), _inputs(3))

    assert fake_router.max_in_flight == 1


def test_ctrl_c_lists_every_submitted_job(fake_router, monkeypatch):
    circuit = _router_circuit()

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        circuit(_weights(), _inputs(3))

    ids = _job_ids_by_row(fake_router)
    assert len(ids) == 3
    for job_id in ids.values():
        assert job_id in str(caught.value)
    assert caught.value.job_ids == [ids[row] for row in range(3)]
    assert [r for r in fake_router.requests if r.url.path.endswith("/cancel")] == []


def test_a_batch_timeout_lists_the_pending_jobs(fake_router, clock):
    fake_router.status_steps = ("queued", "running")
    circuit = _router_circuit(poll_timeout_seconds=40)

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    ids = _job_ids_by_row(fake_router)
    err = caught.value
    assert err.code == "poll_timeout"
    assert err.details["pending"] == [ids[row] for row in range(3)]
    assert fake_router.sleeps == [1.0, 5.0, 30.0, 4.0]
    assert clock[0] == 40.0
    assert [r for r in fake_router.requests if r.url.path.endswith("/cancel")] == []
    for job_id in ids.values():
        assert fake_router.jobs[job_id]["status"] not in TERMINAL


def test_a_request_failing_while_waiting_names_every_job(fake_router):
    circuit = _router_circuit()
    fake_router.fail_next(
        "GET", "/v1/jobs/", status=404, body=_envelope("job_not_found", "Job not found")
    )

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    ids = _job_ids_by_row(fake_router)
    err = caught.value
    assert err.code == "job_not_found"
    assert err.http_status == 404
    assert err.details["job_ids"] == [ids[row] for row in range(3)]
    assert "keep running" in err.message
    assert [r for r in fake_router.requests if r.url.path.endswith("/cancel")] == []


def test_an_empty_batch_sends_nothing(fake_router):
    compiled = _router_circuit(shots=100).compiled
    before = len(fake_router.requests)

    runs = compiled.run(_weights(), np.zeros((0, SPEC.n_inputs)))
    probabilities = compiled.zero_state_probabilities(
        _weights(), np.zeros((0, SPEC.n_inputs))
    )

    assert runs.shape == (0, 2) and runs.dtype == np.float64
    assert probabilities.shape == (0,) and probabilities.dtype == np.float64
    assert len(fake_router.requests) == before


# ---- IBM devices: the client fans a batch out the same way ----


@pytest.mark.parametrize("reason", ["ibm_account_not_connected", "ibm_connection_invalid"])
def test_an_ibm_batch_stops_at_the_estimate_on_an_ibm_refusal(fake_router, reason):
    circuit = _router_circuit(device="ibm/ibm_fez", shots=100)
    fake_router.ibm_refusal = reason

    with pytest.raises(RouterError) as caught:
        circuit(_weights(), _inputs(3))

    assert caught.value.code == reason
    assert caught.value.details["n_jobs"] == 3
    assert "Nothing was submitted" in caught.value.message
    assert "IBM account page (/ibm) of the router web app" in caught.value.message
    assert _submits(fake_router) == []


def test_an_ibm_batch_is_not_refused_for_its_zero_balance(fake_router):
    # An IBM estimate is 0 with available_usd_micro 0 (contract section 10).
    circuit = _router_circuit(device="ibm/ibm_fez", shots=100)

    got = circuit(_weights(), _inputs(3))

    assert got.shape == (3, 2)
    assert len(_submits(fake_router)) == 3
