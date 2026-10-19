"""Bounded retries with a constant Idempotency-Key, and the protocol gate.

These tests drive `RouterHttpClient(config)` directly, not a kit `Circuit`, so
the estimate request a compile sends cannot change their request counts. Every
count filters `FakeRouter.requests` by method and path.
"""

import re

import httpx
import numpy as np
import pytest
from eigenkit.core.backends.numpy_backend import NumpySimulator
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

from eigenorbit_router_client import RouterBackend, RouterConfig, RouterError, _client
from eigenorbit_router_client._client import RouterHttpClient
from eigenorbit_router_client._decode import decode_result
from fake_router import VALID_KEY, upgrade_message


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
WEIGHTS = [0.7, -1.3]
BASE_URL = "https://router.test"
# The 426 example of contract v1 (examples/error_protocol_unsupported.json).
CONTRACT_426 = {
    "error": {
        "code": "protocol_unsupported",
        "message": (
            "This router speaks EO-Protocol-Version 1..1; the request's version is "
            "missing or unsupported. Upgrade eigenorbit-router-client."
        ),
        "details": {
            "client_version": 2,
            "supported_min": 1,
            "supported_max": 1,
            "min_client_version": None,
        },
    }
}
LEDGER_UNAVAILABLE = {
    "error": {
        "code": "ledger_unavailable",
        "message": "The balance service is unavailable. Retry later.",
        "details": {},
    }
}


def _config(**overrides) -> RouterConfig:
    settings = {"device": "braket/sv1", "api_key": VALID_KEY, "base_url": BASE_URL}
    settings.update(overrides)
    return RouterConfig(**settings)


def _body(method="run", shots=0):
    return {
        "device": "braket/sv1",
        "method": method,
        "shots": shots,
        "circuit_spec": circuit_spec_to_json(SPEC),
        "observables": observables_to_json(OBSERVABLES),
        "weights": WEIGHTS,
        "inputs": [],
    }


def _numpy_run():
    compiled = NumpySimulator().compile(SPEC, OBSERVABLES)
    return np.asarray(compiled.run(np.array(WEIGHTS), np.zeros(0)), dtype=np.float64)


def _run_to_result(http: RouterHttpClient, key="call-1"):
    job = http.submit(_body(), idempotency_key=key)
    http.wait(job["id"], timeout_seconds=3600)
    payload = http.get_result(job["id"])
    return decode_result(payload, method="run", n_qubits=2, n_observables=2)


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _envelope(code, message="refused", details=None):
    return {"error": {"code": code, "message": message, "details": details or {}}}


# ---- retries ----


def test_a_503_then_202_resends_the_same_key_and_body(fake_router):
    fake_router.status_steps = ("completed",)
    fake_router.fail_next("POST", "/v1/jobs", status=503, body=LEDGER_UNAVAILABLE)
    http = RouterHttpClient(_config())

    got = _run_to_result(http)

    first, second = _sent(fake_router, "POST", "/v1/jobs")
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"] == "call-1"
    assert first.content == second.content
    assert fake_router.sleeps == [1.0]
    np.testing.assert_allclose(got, _numpy_run(), rtol=0.0, atol=1e-12)


def test_lost_submit_response_replays_the_same_job(fake_router):
    fake_router.status_steps = ("completed",)
    fake_router.drop_response_after_create()
    http = RouterHttpClient(_config())

    got = _run_to_result(http)

    first, second = _sent(fake_router, "POST", "/v1/jobs")
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert first.content == second.content
    assert len(fake_router.jobs) == 1
    np.testing.assert_allclose(got, _numpy_run(), rtol=0.0, atol=1e-12)


def test_a_poll_survives_two_read_timeouts(fake_router):
    http = RouterHttpClient(_config())
    job = http.submit(_body(), idempotency_key="call-1")
    fake_router.fail_next("GET", f"/v1/jobs/{job['id']}", exc=httpx.ReadTimeout, times=2)

    done = http.wait(job["id"], timeout_seconds=3600)

    assert done["status"] == "completed"
    # Two retry delays, then the poll backoff of queued -> running -> completed.
    assert fake_router.sleeps == [1.0, 5.0, 1.0, 5.0]


def test_a_persistent_503_gives_up_after_four_attempts(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", status=503, body=LEDGER_UNAVAILABLE, times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert _client.RETRY_DELAYS == (1.0, 5.0, 30.0)
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 4
    assert fake_router.sleeps == [1.0, 5.0, 30.0]
    assert caught.value.code == "ledger_unavailable"
    assert caught.value.http_status == 503
    assert fake_router.jobs == {}


def test_a_persistent_connect_error_is_router_unreachable(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", exc=httpx.ConnectError, times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 4
    assert fake_router.sleeps == [1.0, 5.0, 30.0]
    assert caught.value.code == "router_unreachable"
    assert caught.value.http_status is None
    assert BASE_URL in caught.value.message
    assert "ConnectError" in caught.value.message


@pytest.mark.parametrize("status", [502, 504])
def test_a_gateway_error_without_an_envelope_is_retried(fake_router, status):
    fake_router.fail_next("POST", "/v1/jobs", status=status, body="<html>Bad Gateway</html>")
    http = RouterHttpClient(_config())

    job = http.submit(_body(), idempotency_key="call-1")

    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 2
    assert fake_router.sleeps == [1.0]
    assert job["id"] in fake_router.jobs


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (500, "internal_error"),
        (422, "validation_error"),
        (402, "insufficient_balance"),
        (409, "idempotency_conflict"),
        (501, "not_implemented"),
    ],
)
def test_a_non_retryable_status_is_raised_after_one_request(fake_router, status, code):
    fake_router.fail_next("POST", "/v1/jobs", status=status, body=_envelope(code))
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []
    assert caught.value.code == code
    assert caught.value.http_status == status


def test_two_run_calls_on_the_same_input_use_two_keys(fake_router):
    compiled = RouterBackend(_config()).compile(SPEC, OBSERVABLES)

    compiled.run(np.array(WEIGHTS), np.zeros(0))
    compiled.run(np.array(WEIGHTS), np.zeros(0))

    keys = [r.headers["Idempotency-Key"] for r in _sent(fake_router, "POST", "/v1/jobs")]
    assert len(keys) == 2
    assert keys[0] != keys[1]


# ---- protocol ----


def test_every_path_sends_protocol_version_exactly_1(fake_router):
    fake_router.status_steps = ("completed",)
    http = RouterHttpClient(_config())

    http.estimate(_body())
    job = http.submit(_body(), idempotency_key="call-1")
    http.get_job(job["id"])
    http.get_result(job["id"])
    with pytest.raises(RouterError):
        http.cancel(job["id"])

    job_id = job["id"]
    paths = {(r.method, r.url.path) for r in fake_router.requests}
    assert paths == {
        ("POST", "/v1/jobs/estimate"),
        ("POST", "/v1/jobs"),
        ("GET", f"/v1/jobs/{job_id}"),
        ("GET", f"/v1/jobs/{job_id}/result"),
        ("POST", f"/v1/jobs/{job_id}/cancel"),
    }
    for request in fake_router.requests:
        assert request.headers.get_list("EO-Protocol-Version") == ["1"]


def test_426_is_raised_once_with_the_router_message(fake_router):
    fake_router.protocol_min = fake_router.protocol_max = 2
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    [only] = fake_router.requests
    assert only.headers["EO-Protocol-Version"] == "1"
    assert fake_router.sleeps == []
    assert caught.value.code == "protocol_unsupported"
    assert caught.value.http_status == 426
    assert caught.value.message == upgrade_message(2, 2)
    assert "Upgrade eigenorbit-router-client" in caught.value.message


def test_the_contract_426_example_message_is_kept_verbatim(fake_router):
    fake_router.fail_next("GET", "/v1/jobs/", status=426, body=CONTRACT_426)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.get_job("0b7e7c9a-1111-4222-8333-444455556666")

    assert len(fake_router.requests) == 1
    assert caught.value.message == CONTRACT_426["error"]["message"]
    assert caught.value.details == CONTRACT_426["error"]["details"]
    assert caught.value.job_id == "0b7e7c9a-1111-4222-8333-444455556666"


def test_the_user_agent_names_client_kit_and_python(fake_router):
    http = RouterHttpClient(_config())

    http.estimate(_body())

    [request] = fake_router.requests
    assert re.fullmatch(
        r"eigenorbit-router-client/\S+ eigenkit/\S+ python/\d+\.\d+\.\d+",
        request.headers["User-Agent"],
    )


def test_a_connection_dropped_before_the_answer_is_retried(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", exc=httpx.RemoteProtocolError)
    http = RouterHttpClient(_config())

    job = http.submit(_body(), idempotency_key="call-1")

    first, second = _sent(fake_router, "POST", "/v1/jobs")
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]
    assert list(fake_router.jobs) == [job["id"]]


def test_any_other_transport_error_is_router_unreachable_at_once(fake_router):
    fake_router.fail_next("POST", "/v1/jobs", exc=httpx.UnsupportedProtocol, times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert caught.value.code == "router_unreachable"
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []
