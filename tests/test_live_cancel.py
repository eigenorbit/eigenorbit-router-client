"""Offline tests for scripts/live_cancel.py.

The live script submits one long job through the router, cancels it once it
runs and polls it to a terminal status. Here it runs only against the
in-process FakeRouter: no socket, no vendor task, no charge. The live run
against a router creates a paid SV1 task and runs by hand.
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import pytest

from eigenorbit_router_client import _client
from fake_router import VALID_KEY


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_cancel.py"
BASE_URL = "https://router.test"
SMALL = ["--qubits", "2", "--layers", "1"]


class _MissingScript:
    """Stands in for the script when it does not exist, so every test fails on
    an assertion naming it instead of erroring in the fixture."""

    def __getattr__(self, name):
        raise AssertionError(f"{SCRIPT} does not exist (looked up {name!r})")


@pytest.fixture(scope="module")
def live_cancel():
    if not SCRIPT.is_file():
        yield _MissingScript()
        return
    spec = importlib.util.spec_from_file_location("live_cancel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_cancel"] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop("live_cancel", None)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)


@pytest.fixture
def no_sleep(live_cancel, monkeypatch):
    """The script's own poll sleeps are recorded, never slept."""
    slept = []
    now = [0.0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(live_cancel, "_sleep", sleep)
    monkeypatch.setattr(live_cancel, "_monotonic", lambda: now[0])
    return slept


@pytest.fixture
def opened_clients(fake_router, monkeypatch):
    make = _client._make_http_client
    opened = []

    def recording(*, base_url, timeout):
        http = make(base_url=base_url, timeout=timeout)
        opened.append(http)
        return http

    monkeypatch.setattr(_client, "_make_http_client", recording)
    return opened


def _cancels(router):
    return [r for r in router.requests if r.method == "POST" and r.url.path.endswith("/cancel")]


def _submits(router):
    return [r for r in router.requests if r.method == "POST" and r.url.path == "/v1/jobs"]


def _run(live_cancel, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return live_cancel.main([*SMALL, *args])


# ---- configuration --------------------------------------------------------------


def test_it_refuses_to_run_without_an_explicit_router_url(live_cancel, monkeypatch, capsys):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)

    assert live_cancel.main([]) == 2
    assert "EIGENORBIT_ROUTER_URL" in capsys.readouterr().err


def test_it_refuses_to_run_without_an_api_key(live_cancel, monkeypatch, capsys):
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)

    assert live_cancel.main([]) == 2
    assert "EIGENORBIT_ROUTER_API_KEY" in capsys.readouterr().err


def test_the_default_circuit_is_long_and_on_sv1_at_100_shots(live_cancel):
    args = live_cancel.parse_args([])

    assert args.device == "braket/sv1"
    assert args.shots == 100
    spec, observables, weights, inputs = live_cancel.build_circuit(args.qubits, args.layers)
    assert spec.n_qubits >= 20
    assert len(spec.gates) >= 200
    assert len(weights) == spec.n_weights
    assert len(observables) >= 1


def test_a_qpu_defaults_to_a_small_circuit_cancelled_while_queued(live_cancel):
    args = live_cancel.parse_args(["--device", "braket/iqm_garnet", "--shots", "10"])

    assert args.cancel_at == "queued"
    assert args.qubits <= 20


# ---- the cancel flow against the double -----------------------------------------


def test_a_job_the_vendor_cancels_ends_cancelled_and_exits_0(
    live_cancel, fake_router, env, no_sleep, opened_clients, capsys
):
    fake_router.cancel_mode = "vendor"

    code = _run(live_cancel)

    out = capsys.readouterr().out
    [job_id] = fake_router.jobs
    assert code == 0
    assert len(_submits(fake_router)) == 1
    assert len(_cancels(fake_router)) == 1
    # The cancel was sent once the job was running.
    cancel_at = fake_router.requests.index(_cancels(fake_router)[0])
    gets = [
        r for r in fake_router.requests[:cancel_at]
        if r.method == "GET" and r.url.path == f"/v1/jobs/{job_id}"
    ]
    assert len(gets) == 2  # queued, then running
    assert job_id in out
    assert "final status: cancelled" in out
    assert "billing: released" in out
    assert all(http.is_closed for http in opened_clients)


def test_a_job_the_vendor_finishes_first_ends_completed_and_exits_0(
    live_cancel, fake_router, env, no_sleep, capsys
):
    fake_router.cancel_mode = "vendor_finishes"

    code = _run(live_cancel)

    out = capsys.readouterr().out
    assert code == 0
    assert "final status: completed" in out
    assert "vendor finished the task first" in out
    assert "billing: settled" in out
    assert "actual_cost_usd_micro=4500" in out
    assert "vendor_cost_usd_micro=3750" in out
    assert "fee_usd_micro=750" in out


def test_a_job_that_ends_before_the_cancel_is_sent_exits_1(
    live_cancel, fake_router, env, no_sleep, capsys
):
    fake_router.status_steps = ("completed",)

    code = _run(live_cancel)

    out = capsys.readouterr().out
    assert code == 1
    assert _cancels(fake_router) == []
    assert "before the cancel" in out


def test_a_failed_job_exits_1(live_cancel, fake_router, env, no_sleep, capsys):
    fake_router.cancel_mode = "vendor_finishes"
    fake_router.fail_with = "vendor_failed"

    code = _run(live_cancel)

    assert code == 1
    assert "final status: failed" in capsys.readouterr().out


def test_a_queued_cancel_on_a_qpu(live_cancel, fake_router, env, no_sleep, capsys):
    fake_router.devices = fake_router.devices | {"braket/iqm_garnet"}

    code = _run(live_cancel, "--device", "braket/iqm_garnet", "--shots", "10")

    out = capsys.readouterr().out
    [job] = fake_router.jobs.values()
    assert code == 0
    assert job["request"]["device"] == "braket/iqm_garnet"
    assert job["request"]["shots"] == 10
    assert "final status: cancelled" in out


def test_a_refused_cancel_exits_1_and_names_the_code(
    live_cancel, fake_router, env, no_sleep, capsys
):
    fake_router.fail_next(
        "POST",
        "/v1/jobs/",
        status=409,
        body={
            "error": {
                "code": "job_not_cancellable",
                "message": "The job is completed",
                "details": {"status": "completed"},
            }
        },
    )

    code = _run(live_cancel)

    assert code == 1
    assert "job_not_cancellable" in capsys.readouterr().out


def test_the_output_and_report_never_carry_the_api_key(
    live_cancel, fake_router, env, no_sleep, tmp_path, capsys
):
    fake_router.cancel_mode = "vendor"
    report = tmp_path / "cancel.json"

    code = _run(live_cancel, "--report", str(report))

    captured = capsys.readouterr()
    assert code == 0
    for text in (captured.out, captured.err, report.read_text()):
        assert VALID_KEY not in text
        assert "eo_test_" not in text
    data = json.loads(report.read_text())
    assert data["final_status"] == "cancelled"
    assert data["job"]["billing"] == "released"
    assert data["passed"] is True
