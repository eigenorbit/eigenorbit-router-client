import subprocess
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
from fake_router import DEVICE_ACCESS_MESSAGE, VALID_KEY


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
    with pytest.raises(ImportError, match="eigenkit 1.4.0"):
        backend.compile(SPEC)


def test_importing_the_client_loads_no_photonics():
    code = "import sys, eigenorbit_router_client; assert 'eigenkit.photonics' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def _cancels(router):
    return [r for r in router.requests if r.url.path.endswith("/cancel")]


def test_an_interrupt_while_the_job_runs_cancels_it(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        compiled().run(WEIGHTS, ONE)
    job_id = fake_router.requests[1].url.path.split("/")[3]
    cancels = _cancels(fake_router)
    assert [r.url.path for r in cancels] == [f"/v1/jobs/{job_id}/cancel"]


def test_a_poll_timeout_cancels_the_job(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")
    now = [0.0]
    monkeypatch.setattr(_client, "_sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(_client, "_monotonic", lambda: now[0])
    with pytest.raises(RouterError) as caught:
        RouterPhotonicBackend(
            RouterConfig("quandela/sim_slos", shots=537, api_key=VALID_KEY,
                         base_url="https://router.test", poll_timeout_seconds=5)
        ).compile(SPEC).run(WEIGHTS, ONE)
    assert caught.value.code == "poll_timeout"
    assert len(_cancels(fake_router)) == 1


def test_a_cancel_that_fails_leaves_the_original_exception(fake_router, monkeypatch):
    fake_router.status_steps = ("queued", "running")
    fake_router.fail_next("POST", "/v1/jobs/", status=500, times=10)

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        compiled().run(WEIGHTS, ONE)
    assert len(_cancels(fake_router)) >= 1


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
