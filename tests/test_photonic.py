import json
import logging
import pickle
import sys
import uuid
import warnings

import numpy as np
import pytest
from eigenkit import CostWarning
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.ir import InputRef, WeightRef
from eigenkit.core.observables import all_z
from eigenkit.photonics.executor import Compiled, Executor
from eigenkit.photonics.ir import (
    BeamSplitter,
    FockProbability,
    Permutation,
    PhaseShifter,
    PhotonicCircuitSpec,
    PhotonNumber,
)
from eigenkit.photonics.reservoir import random_reservoir_spec, run_photonic_reservoir
from eigenkit.photonics.wire import photonic_spec_to_json

import eigenorbit_router_client as package
from eigenorbit_router_client import (
    RouterBackend,
    RouterCompiled,
    RouterConfig,
    RouterError,
    RouterPhotonicBackend,
    RouterPhotonicCompiled,
    _client,
)
from fake_router import DEVICE_ACCESS_MESSAGE, VALID_KEY, fock_counts_oracle


SPEC = PhotonicCircuitSpec(
    n_modes=4, n_weights=1, n_inputs=1,
    components=(PhaseShifter(1, InputRef(0)), BeamSplitter((0, 1), WeightRef(0)),
                Permutation((1, 0, 2, 3))),
    input_state=(1, 0, 1, 0),
)
WEIGHTS = np.array([0.3])
ONE = np.array([0.276])
DIST = {(1, 0, 1, 0): 261 / 537, (0, 1, 0, 1): 276 / 537}


def config(device="quandela/sim_slos", shots=537):
    return RouterConfig(device, shots=shots, api_key=VALID_KEY,
                        base_url="https://router.test")


def compiled(**over):
    return RouterPhotonicBackend(config(**over)).compile(SPEC)


def posts(router):
    return [r for r in router.requests if r.method == "POST" and r.url.path == "/v1/jobs"]


def test_the_classes_fit_the_photonic_protocols_and_are_exported(fake_router):
    backend = RouterPhotonicBackend(config())
    assert isinstance(backend, Executor)
    assert isinstance(backend.compile(SPEC), Compiled)
    assert {"RouterPhotonicBackend", "RouterPhotonicCompiled"} <= set(package.__all__)


def test_the_request_body(fake_router):
    assert compiled().build_request(WEIGHTS, ONE) == {
        "device": "quandela/sim_slos", "method": "photonic_sample", "shots": 537,
        "circuit_spec": photonic_spec_to_json(SPEC), "observables": [],
        "weights": [0.3], "inputs": [0.276],
    }


def test_run_is_one_job_with_no_estimate_and_no_warning(fake_router):
    with warnings.catch_warnings():
        warnings.simplefilter("error", CostWarning)
        dist = compiled().run(WEIGHTS, ONE)
    assert dist == DIST
    assert len(posts(fake_router)) == 1
    uuid.UUID(posts(fake_router)[0].headers["Idempotency-Key"])
    assert all(r.url.path != "/v1/jobs/estimate" for r in fake_router.requests)


def test_run_without_a_surviving_shot_raises_a_plain_runtime_error(fake_router):
    fake_router.empty_rows = frozenset({0})
    with pytest.raises(RuntimeError, match="no shots survived") as caught:
        compiled().run(WEIGHTS, ONE)
    assert type(caught.value) is RuntimeError


@pytest.mark.parametrize(
    ("observable", "value"),
    [(FockProbability((0, 1, 0, 1)), 276 / 537), (PhotonNumber(0), 261 / 537)],
)
def test_expectation_comes_from_one_run(fake_router, observable, value):
    assert compiled().expectation(WEIGHTS, ONE, observable) == pytest.approx(value)
    assert len(posts(fake_router)) == 1


@pytest.mark.parametrize(
    "observable",
    [FockProbability((1, 0, 1)), FockProbability((2, 1, 0, 0)), PhotonNumber(4)],
)
def test_a_wrong_observable_is_refused_before_anything_is_sent(fake_router, observable):
    with pytest.raises(ValueError):
        compiled().expectation(WEIGHTS, ONE, observable)
    assert fake_router.requests == []


def test_jacobian_always_raises(fake_router):
    with pytest.raises(NotImplementedError):
        compiled().jacobian(WEIGHTS, ONE, PhotonNumber(0))
    assert fake_router.requests == []


@pytest.mark.parametrize(
    ("weights", "inputs"),
    [(np.zeros(2), ONE), (np.zeros((1, 1)), ONE), (WEIGHTS, np.zeros(2)),
     (WEIGHTS, np.zeros((2, 1)))],
)
def test_run_checks_the_shapes_before_anything_is_sent(fake_router, weights, inputs):
    with pytest.raises(ValueError):
        compiled().run(weights, inputs)
    assert fake_router.requests == []


def test_a_failed_job_raises_its_router_error(fake_router):
    fake_router.fail_with = "vendor_failed"
    with pytest.raises(RouterError) as caught:
        compiled().run(WEIGHTS, ONE)
    assert caught.value.code == "vendor_failed" and caught.value.job_id


def test_an_account_outside_the_preview_gets_the_routers_refusal(fake_router):
    fake_router.quandela_access = False
    with pytest.raises(RouterError) as caught:
        compiled().run(WEIGHTS, ONE)
    error = caught.value
    assert (error.code, error.http_status) == ("device_access_restricted", 403)
    assert error.message == DEVICE_ACCESS_MESSAGE
    assert error.details == {"device": "quandela/sim_slos"}
    assert len(posts(fake_router)) == 1


@pytest.mark.parametrize("device", ["braket/sv1", "ibm/ibm_fez"])
def test_the_photonic_backend_refuses_a_qubit_device(fake_router, device):
    for build in (RouterPhotonicBackend, lambda c: RouterPhotonicCompiled(SPEC, c)):
        with pytest.raises(ValueError, match="quandela/"):
            build(config(device))
    assert fake_router.requests == []


def test_zero_shots_are_refused_before_anything_is_sent(fake_router):
    with pytest.raises(ValueError, match="shots"):
        RouterPhotonicBackend(config(shots=0))
    assert fake_router.requests == []


def test_the_qubit_backend_refuses_a_quandela_device(fake_router):
    qubit_spec = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
    for build in (
        lambda: RouterBackend(config()),
        lambda: config().make_backend(),
        lambda: RouterCompiled(qubit_spec, all_z(2), config()),
    ):
        with pytest.raises(ValueError, match="RouterPhotonicBackend"):
            build()
    assert fake_router.requests == []


def test_an_older_eigenkit_is_named_at_compile(fake_router, monkeypatch):
    monkeypatch.setitem(sys.modules, "eigenkit.photonics.wire", None)
    backend = RouterPhotonicBackend(config())
    with pytest.raises(ImportError, match="eigenkit 1.4.0") as caught:
        backend.compile(SPEC)
    assert isinstance(caught.value.__cause__, ImportError)


def _cancels(router):
    return [r for r in router.requests if r.url.path.endswith("/cancel")]


def spy_on_wait(monkeypatch):
    """Record every exception the client's wait raises, to compare by identity."""
    raised = []
    real_wait = _client.RouterHttpClient.wait

    def wait(self, *args, **kwargs):
        try:
            return real_wait(self, *args, **kwargs)
        except BaseException as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(_client.RouterHttpClient, "wait", wait)
    return raised


def test_an_interrupt_while_the_job_runs_cancels_it(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    raised = spy_on_wait(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as caught:
        compiled().run(WEIGHTS, ONE)
    assert len(raised) == 1 and caught.value is raised[0]
    job_id = fake_router.requests[1].url.path.split("/")[3]
    cancels = _cancels(fake_router)
    assert [r.url.path for r in cancels] == [f"/v1/jobs/{job_id}/cancel"]


def test_a_poll_timeout_cancels_the_job(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")
    now = [0.0]
    monkeypatch.setattr(_client, "_sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(_client, "_monotonic", lambda: now[0])
    raised = spy_on_wait(monkeypatch)
    with pytest.raises(RouterError) as caught:
        RouterPhotonicBackend(
            RouterConfig("quandela/sim_slos", shots=537, api_key=VALID_KEY,
                         base_url="https://router.test", poll_timeout_seconds=5)
        ).compile(SPEC).run(WEIGHTS, ONE)
    assert caught.value.code == "poll_timeout"
    assert len(raised) == 1 and caught.value is raised[0]
    assert len(_cancels(fake_router)) == 1


STALE_ADVICE = ("keeps running", "no auto-cancel", "handle.cancel", "compiled.submit")


def test_an_interrupt_says_the_job_is_being_cancelled(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        compiled().run(WEIGHTS, ONE)
    text = str(caught.value)
    assert text.startswith("interrupted while waiting for router job ")
    assert "is being cancelled" in text
    assert not [phrase for phrase in STALE_ADVICE if phrase in text]


def test_a_poll_timeout_says_the_job_is_being_cancelled(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")
    now = [0.0]
    monkeypatch.setattr(_client, "_sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(_client, "_monotonic", lambda: now[0])
    with pytest.raises(RouterError) as caught:
        RouterPhotonicBackend(
            RouterConfig("quandela/sim_slos", shots=537, api_key=VALID_KEY,
                         base_url="https://router.test", poll_timeout_seconds=5)
        ).compile(SPEC).run(WEIGHTS, ONE)
    text = str(caught.value)
    assert "poll_timeout_seconds=5" in text and "is being cancelled" in text
    assert "raise RouterConfig(poll_timeout_seconds=...) to wait longer" in text
    assert not [phrase for phrase in STALE_ADVICE if phrase in text]
    assert caught.value.code == "poll_timeout"
    assert caught.value.details == {"status": "running", "poll_timeout_seconds": 5}


def test_a_cancel_that_fails_leaves_the_original_exception(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")
    fake_router.fail_next("POST", "/v1/jobs/", status=500, times=10)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    raised = spy_on_wait(monkeypatch)
    with pytest.raises(KeyboardInterrupt) as caught:
        compiled().run(WEIGHTS, ONE)
    job_id = fake_router.requests[1].url.path.split("/")[3]
    assert caught.value is raised[0]
    assert caught.value.job_id == job_id and job_id in str(caught.value)
    # a 500 is not retried: exactly one cancel request, for this job
    assert [r.url.path for r in _cancels(fake_router)] == [f"/v1/jobs/{job_id}/cancel"]


def test_a_cancel_that_fails_is_logged_by_type_and_job_id_only(
    fake_router, monkeypatch, caplog
):
    fake_router.status_steps = ("queued", "running")
    fake_router.fail_next(
        "POST", "/v1/jobs/", status=500,
        body={"error": {"code": "internal_error", "message": "SECRET-DETAIL"}},
    )

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    with caplog.at_level(logging.WARNING, logger="eigenorbit_router_client.photonic"):
        with pytest.raises(KeyboardInterrupt):
            compiled().run(WEIGHTS, ONE)
    job_id = fake_router.requests[1].url.path.split("/")[3]
    records = [r for r in caplog.records if r.name == "eigenorbit_router_client.photonic"]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    text = records[0].getMessage()
    assert job_id in text and "RouterError" in text
    assert "SECRET-DETAIL" not in text and VALID_KEY not in text


def test_the_api_key_is_in_neither_a_repr_nor_a_pickle(fake_router):
    backend = RouterPhotonicBackend(config())
    for text in (repr(backend), repr(backend.compile(SPEC))):
        assert VALID_KEY not in text
    data = pickle.dumps(backend)
    assert VALID_KEY.encode() not in data
    assert b"eo_test_" not in data and b"eo_live_" not in data
    assert b"quandela/sim_slos" in data


def test_a_job_that_failed_at_the_vendor_sends_no_cancel(fake_router):
    fake_router.fail_with = "vendor_failed"
    with pytest.raises(RouterError):
        compiled().run(WEIGHTS, ONE)
    assert _cancels(fake_router) == []


def test_a_refused_submit_sends_no_cancel(fake_router):
    fake_router.quandela_access = False
    with pytest.raises(RouterError):
        compiled().run(WEIGHTS, ONE)
    assert _cancels(fake_router) == []


ROWS = np.array([[0.276], [0.0], [0.1]])
MANY = np.linspace(0.0, 0.39, 40).reshape(40, 1)


def expected(rows, shots=537):
    return [
        {tuple(int(n) for n in key.split(",")): count / shots
         for key, count in fock_counts_oracle([1, 0, 1, 0], list(row), shots).items()}
        for row in rows
    ]


def refusal(details):
    return {"error": {"code": "batch_not_supported", "message": "refused",
                      "details": details}}


def test_a_batch_is_one_router_job_in_row_order(fake_router):
    assert compiled().run_batch(WEIGHTS, ROWS) == expected(ROWS)
    assert len(posts(fake_router)) == 1
    assert json.loads(posts(fake_router)[0].content)["inputs"] == ROWS.tolist()


def test_rows_above_the_limit_go_in_chunks_one_after_another(fake_router):
    dists = compiled(device="quandela/sim_belenos").run_batch(WEIGHTS, MANY)
    sent = posts(fake_router)
    keys = [r.headers["Idempotency-Key"] for r in sent]
    events = [(r.method, r.url.path.rsplit("/", 1)[-1]) for r in fake_router.requests]
    submits = [i for i, event in enumerate(events) if event == ("POST", "jobs")]
    results = [i for i, event in enumerate(events) if event == ("GET", "result")]
    assert dists == expected(MANY)
    assert [len(json.loads(r.content)["inputs"]) for r in sent] == [40, 16, 16, 8]
    assert keys[1:] == [f"{keys[0]}:{chunk}" for chunk in range(3)]
    assert len(results) == 3 and all(r < s for r, s in zip(results, submits[2:]))
    assert fake_router.max_in_flight == 1


@pytest.mark.parametrize(
    "details", [{}, {"max_rows": 0}, {"max_rows": "16"}, {"max_rows": True}, {"max_rows": 3}]
)
def test_a_refusal_without_a_usable_row_limit_is_raised_as_it_came(fake_router, details):
    fake_router.fail_next("POST", "/v1/jobs", status=422, body=refusal(details))
    with pytest.raises(RouterError) as caught:
        compiled().run_batch(WEIGHTS, ROWS)
    assert caught.value.code == "batch_not_supported"
    assert len(posts(fake_router)) == 1


def test_a_chunk_that_is_refused_again_ends_the_call(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", status=422, body=refusal({"max_rows": 4}),
                          key_suffix=":1")
    with pytest.raises(RouterError) as caught:
        compiled(device="quandela/sim_belenos").run_batch(WEIGHTS, MANY)
    assert caught.value.code == "batch_not_supported"
    assert len(posts(fake_router)) == 3


def test_a_failed_chunk_raises_and_the_next_one_is_never_sent(fake_router):
    fake_router.fail_rows = {1: "vendor_failed"}
    with pytest.raises(RouterError) as caught:
        compiled(device="quandela/sim_belenos").run_batch(WEIGHTS, MANY)
    assert caught.value.code == "vendor_failed"
    assert len(posts(fake_router)) == 3
    assert _cancels(fake_router) == []


def test_an_interrupt_in_the_second_chunk_cancels_that_job_only(fake_router, monkeypatch):
    def interrupt_in_chunk_two(seconds):
        if len(posts(fake_router)) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt_in_chunk_two)
    with pytest.raises(KeyboardInterrupt):
        compiled(device="quandela/sim_belenos").run_batch(WEIGHTS, MANY)
    fetched = [r for r in fake_router.requests if r.url.path.endswith("/result")]
    assert len(posts(fake_router)) == 3 and len(fetched) == 1
    first_job = fetched[0].url.path.split("/")[3]
    second_job = next(
        r.url.path.split("/")[3] for r in fake_router.requests
        if r.method == "GET" and r.url.path.split("/")[3] != first_job
    )
    cancels = _cancels(fake_router)
    assert [r.url.path for r in cancels] == [f"/v1/jobs/{second_job}/cancel"]


def test_a_row_without_a_surviving_shot_is_named(fake_router):
    fake_router.empty_rows = frozenset({1})
    with pytest.raises(RuntimeError, match=r"rows \[1\] of 3") as caught:
        compiled().run_batch(WEIGHTS, ROWS)
    assert type(caught.value) is RuntimeError


def test_many_empty_rows_are_listed_up_to_ten_with_the_total(fake_router):
    fake_router.empty_rows = frozenset(range(2, 14))
    rows = np.linspace(0.0, 0.39, 15).reshape(15, 1)
    with pytest.raises(RuntimeError) as caught:
        compiled().run_batch(WEIGHTS, rows)
    message = str(caught.value)
    assert "rows [2, 3, 4, 5, 6, 7, 8, 9, 10, 11] of 15" in message
    assert "12 rows" in message and "12, 13" not in message


def test_an_empty_row_of_a_split_batch_is_named_by_its_index_in_the_batch(fake_router):
    # the fake counts rows within a job: local row 10 is row 10 of the whole batch in
    # chunk 0 and row 26 in chunk 1
    fake_router.empty_rows = frozenset({10})
    with pytest.raises(RuntimeError, match=r"rows \[10, 26\] of 40"):
        compiled(device="quandela/sim_belenos").run_batch(WEIGHTS, MANY)
    assert len(posts(fake_router)) == 4


def test_a_batch_above_the_request_limit_is_refused_after_one_request(fake_router):
    envelope = {"error": {"code": "request_too_large",
                          "message": "The job is larger than the router accepts",
                          "details": {"field": "inputs", "max_items": 100000}}}
    fake_router.fail_next("POST", "/v1/jobs", status=413, body=envelope)
    with pytest.raises(RouterError) as caught:
        compiled().run_batch(WEIGHTS, ROWS)
    assert caught.value.code == "request_too_large"
    assert caught.value.details == {"field": "inputs", "max_items": 100000}
    assert len(posts(fake_router)) == 1
    assert _cancels(fake_router) == []


def test_an_empty_batch_is_an_empty_list_and_sends_nothing(fake_router):
    assert compiled().run_batch(WEIGHTS, np.zeros((0, 1))) == []
    assert fake_router.requests == []


@pytest.mark.parametrize("rows", [np.zeros(1), np.zeros((2, 2)), np.zeros((2, 1, 1))])
def test_a_batch_must_be_two_d_with_the_circuits_inputs(fake_router, rows):
    with pytest.raises(ValueError):
        compiled().run_batch(WEIGHTS, rows)
    assert fake_router.requests == []


def test_the_reservoir_runs_as_one_router_job(fake_router):
    spec, weights = random_reservoir_spec(4, 2, depth=2, input_state=(1, 1, 0, 0),
                                          rng=np.random.default_rng(0))
    sequences = np.random.default_rng(1).uniform(-1, 1, (2, 5, 2))
    executor = RouterPhotonicBackend(config(shots=2000))
    features = run_photonic_reservoir(spec, weights, sequences, executor=executor)
    assert features.shape == (2, 5, 10)
    np.testing.assert_allclose(features.sum(axis=-1), 1.0)
    assert len(posts(fake_router)) == 1
    assert len(json.loads(posts(fake_router)[0].content)["inputs"]) == 10
