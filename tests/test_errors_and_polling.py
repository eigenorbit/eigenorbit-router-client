"""The RouterError mapping and blocking waits.

Error-mapping tests drive `RouterHttpClient(config)` directly; polling tests
submit through a compiled circuit and count only GET /v1/jobs/{id} and
POST /v1/jobs/{id}/cancel, so the estimate request a compile sends cannot
change them. A fake monotonic clock advances by every recorded sleep.
"""

import json
import pickle
from pathlib import Path

import httpx
import numpy as np
import pytest
from eigenkit.core.backends.numpy_backend import NumpySimulator
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

from eigenorbit_router_client import RouterBackend, RouterConfig, RouterError, _client
from eigenorbit_router_client._client import RouterHttpClient
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
NO_INPUTS = np.zeros(0)
BASE_URL = "https://router.test"
JOB_ID = "0b7e7c9a-1111-4222-8333-444455556666"


def _config(**overrides) -> RouterConfig:
    settings = {"device": "braket/sv1", "api_key": VALID_KEY, "base_url": BASE_URL}
    settings.update(overrides)
    return RouterConfig(**settings)


def _body(method="run", shots=0, device="braket/sv1"):
    return {
        "device": device,
        "method": method,
        "shots": shots,
        "circuit_spec": circuit_spec_to_json(SPEC),
        "observables": observables_to_json(OBSERVABLES),
        "weights": WEIGHTS.tolist(),
        "inputs": [],
    }


def _envelope(code, message="refused", details=None):
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


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


def _handle(poll_timeout_seconds=40, method="run"):
    compiled = RouterBackend(
        _config(poll_timeout_seconds=poll_timeout_seconds)
    ).compile(SPEC, OBSERVABLES)
    return compiled.submit(WEIGHTS, NO_INPUTS, method=method)


# ---- error mapping ----


def test_an_unknown_code_is_kept_verbatim_and_not_retried(fake_router):
    fake_router.fail_next(
        "POST", "/v1/jobs", status=400, body=_envelope("brand_new_code", "new rule")
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert caught.value.code == "brand_new_code"
    assert caught.value.http_status == 400
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []


def test_a_non_json_502_after_the_retries_is_invalid_response(fake_router):
    fake_router.fail_next(
        "GET", f"/v1/jobs/{JOB_ID}", status=502, body="<html>proxy</html>", times=99
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.get_job(JOB_ID)

    assert caught.value.code == "invalid_response"
    assert caught.value.http_status == 502
    assert caught.value.job_id == JOB_ID
    assert len(_sent(fake_router, "GET", f"/v1/jobs/{JOB_ID}")) == 4
    assert "<html>" not in str(caught.value)


def test_a_401_never_carries_the_api_key(fake_router):
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=401,
        body=_envelope("unauthenticated", "Invalid or missing API key"),
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert err.code == "unauthenticated"
    assert err.http_status == 401
    for text in (str(err), repr(err), repr(err.details), err.message):
        assert VALID_KEY not in text
        assert "Bearer" not in text
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_a_shots_invalid_rejection_tells_the_user_to_set_shots(fake_router):
    # The router refuses zero_state_probability at shots=0 on Braket simulators.
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(method="zero_state_probability", shots=0), idempotency_key="k")

    err = caught.value
    assert err.code == "shots_invalid"
    assert err.http_status == 422
    assert err.details == {"min_shots": 1, "max_shots": 100_000}
    assert err.message.startswith("shots_invalid for this device")
    assert "RouterConfig(shots=" in err.message
    assert "1..100000" in err.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.jobs == {}


def test_default_shots_on_a_qpu_surface_the_router_range(fake_router):
    fake_router.devices = fake_router.devices | {"ibm/ibm_torino"}
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=422,
        body=_envelope(
            "shots_invalid",
            "shots must be within 1..100000 on ibm/ibm_torino",
            {"min_shots": 1, "max_shots": 100_000},
        ),
    )
    compiled = RouterBackend(_config(device="ibm/ibm_torino")).compile(SPEC, OBSERVABLES)

    with pytest.raises(RouterError) as caught:
        compiled.run(WEIGHTS, NO_INPUTS)

    err = caught.value
    assert err.code == "shots_invalid"
    assert err.details == {"min_shots": 1, "max_shots": 100_000}
    assert err.message.startswith("shots must be within 1..100000 on ibm/ibm_torino")
    assert "RouterConfig(shots=" in err.message


def test_from_job_maps_a_failed_job():
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "failed",
            "billing": "released",
            "error": {"code": "vendor_failed", "message": "The task failed"},
        }
    )

    assert err.code == "vendor_failed"
    assert err.message == "The task failed"
    assert err.details == {"status": "failed", "billing": "released"}
    assert err.job_id == JOB_ID
    assert err.http_status is None
    assert isinstance(err, RuntimeError)


def test_from_job_maps_a_rejected_job():
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "rejected",
            "billing": None,
            "error": {"code": "insufficient_balance", "message": "Top up"},
        }
    )

    assert err.code == "insufficient_balance"
    assert err.details == {"status": "rejected", "billing": None}
    assert err.job_id == JOB_ID
    assert err.http_status is None


def test_from_job_maps_a_cancelled_job_without_an_error():
    err = RouterError.from_job(
        {"id": JOB_ID, "status": "cancelled", "billing": "released", "error": None}
    )

    assert err.code == "job_not_completed"
    assert err.details["status"] == "cancelled"
    assert err.job_id == JOB_ID
    assert JOB_ID in err.message
    assert err.http_status is None


def test_client_error_codes_cover_the_client_side_failures():
    from eigenorbit_router_client.errors import CLIENT_ERROR_CODES

    assert CLIENT_ERROR_CODES == {"invalid_response", "poll_timeout", "router_unreachable"}


# ---- lost submits: the error names the idempotency key that replays the job ----


def test_a_submit_whose_every_response_is_lost_names_its_idempotency_key(fake_router):
    fake_router.drop_response_after_create(times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert err.code == "router_unreachable"
    assert err.http_status is None
    assert err.details["idempotency_key"] == "call-1"
    assert "may already exist" in err.message
    assert "same Idempotency-Key" in err.message
    assert "call-1" in err.message
    assert "idempotency_key=" in err.message
    # No list route exists (contract section 5): never point at one.
    assert "GET /v1/jobs," not in err.message
    sent = _sent(fake_router, "POST", "/v1/jobs")
    assert len(sent) == 4
    assert {r.headers["Idempotency-Key"] for r in sent} == {"call-1"}
    # The router created (and holds) exactly one job the client never saw.
    assert len(fake_router.jobs) == 1


def test_a_lost_submit_error_never_carries_the_api_key(fake_router):
    fake_router.drop_response_after_create(times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    for text in (str(err), repr(err), repr(err.details), err.message):
        assert VALID_KEY not in text
        assert "Bearer" not in text


def test_an_unreachable_get_carries_the_job_id_and_no_idempotency_key(fake_router):
    fake_router.fail_next("GET", f"/v1/jobs/{JOB_ID}", exc=httpx.ReadTimeout, times=99)
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.get_job(JOB_ID)

    err = caught.value
    assert err.code == "router_unreachable"
    assert err.job_id == JOB_ID
    assert "idempotency_key" not in err.details
    assert "Idempotency-Key" not in err.message


# ---- job error codes and categories ----


def test_from_job_reads_the_error_category():
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "failed",
            "billing": "released",
            "error": {
                "code": "vendor_rejected",
                "message": "The circuit could not be built",
                "category": "circuit",
            },
        }
    )

    assert err.code == "vendor_rejected"
    assert err.category == "circuit"


def test_from_job_without_a_category_has_category_none():
    # A job error without a category: the router does not send one.
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "failed",
            "billing": "released",
            "error": {"code": "vendor_failed", "message": "The task failed"},
        }
    )

    assert err.category is None


def test_an_http_error_has_no_category():
    err = RouterError.from_response(httpx.Response(401, json=_envelope("unauthenticated")))

    assert err.category is None


def test_the_category_survives_pickling():
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "failed",
            "billing": "released",
            "error": {"code": "vendor_failed", "message": "failed", "category": "vendor"},
        }
    )

    copy = pickle.loads(pickle.dumps(err))

    assert (copy.code, copy.category, copy.job_id) == ("vendor_failed", "vendor", JOB_ID)


def test_a_failed_job_from_the_double_carries_its_category(fake_router):
    fake_router.fail_with = "vendor_failed"
    fake_router.status_steps = ("completed",)
    handle = _handle()

    with pytest.raises(RouterError) as caught:
        handle.result()

    assert caught.value.code == "vendor_failed"
    assert caught.value.category == "vendor"


def test_device_unavailable_names_the_device_and_its_status(fake_router):
    fake_router.device_status = {"braket/sv1": "offline"}
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert (err.code, err.http_status) == ("device_unavailable", 422)
    assert err.details == {"device": "braket/sv1", "status": "offline"}
    assert "braket/sv1" in err.message
    assert "offline" in err.message
    assert "GET /v1/devices" in err.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []
    assert fake_router.jobs == {}


def test_a_stale_device_status_asks_to_retry_in_a_few_minutes(fake_router):
    fake_router.stale_devices = frozenset({"braket/sv1"})
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert err.code == "device_unavailable"
    assert err.details["stale"] is True
    assert "has not confirmed this device's status recently" in err.message
    assert "retry in a few minutes" in err.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_concurrency_limit_exceeded_gives_both_numbers(fake_router):
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=429,
        body=_envelope("concurrency_limit_exceeded", "refused", {"limit": 100, "in_flight": 90}),
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert (err.code, err.http_status) == ("concurrency_limit_exceeded", 429)
    assert err.details == {"limit": 100, "in_flight": 90}
    assert "90 jobs in flight" in err.message
    assert "limit is 100" in err.message
    assert "wait for a job to finish or cancel one" in err.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []


def test_the_double_enforces_its_concurrency_cap(fake_router):
    fake_router.concurrency_limit = 1
    http = RouterHttpClient(_config())
    http.submit(_body(), idempotency_key="call-1")

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-2")

    assert caught.value.code == "concurrency_limit_exceeded"
    assert caught.value.details == {"limit": 1, "in_flight": 1}
    assert len(fake_router.jobs) == 1


def test_job_not_cancellable_names_the_status(fake_router):
    fake_router.status_steps = ("completed",)
    http = RouterHttpClient(_config())
    job = http.submit(_body(), idempotency_key="call-1")
    http.get_job(job["id"])

    with pytest.raises(RouterError) as caught:
        http.cancel(job["id"])

    err = caught.value
    assert (err.code, err.http_status) == ("job_not_cancellable", 409)
    assert err.details == {"status": "completed"}
    assert err.job_id == job["id"]
    assert "already completed" in err.message
    assert len(_sent(fake_router, "POST", f"/v1/jobs/{job['id']}/cancel")) == 1
    assert fake_router.sleeps == []


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (422, "device_unavailable"),
        (429, "concurrency_limit_exceeded"),
        (409, "job_not_cancellable"),
    ],
)
def test_the_new_codes_keep_the_router_message_without_details(fake_router, status, code):
    fake_router.fail_next("POST", "/v1/jobs", status=status, body=_envelope(code, "nope"))
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    assert caught.value.code == code
    assert caught.value.message.startswith("nope")
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


# ---- refusals: region_blocked, access_restricted, user_blocked, result_expired ----

REGION_BLOCKED_MESSAGE = "EigenOrbit Router is not available in your region"
EXPIRED_AT = "2026-12-23T14:45:03.214000Z"


def test_region_blocked_names_the_sanctions_rules_and_is_not_retried(fake_router):
    fake_router.fail_next(
        "POST", "/v1/jobs", status=451, body=_envelope("region_blocked", REGION_BLOCKED_MESSAGE)
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert (err.code, err.http_status) == ("region_blocked", 451)
    assert err.message == (
        "EigenOrbit Router is not available in your region "
        "(sanctions and export rules)."
    )
    assert err.details == {}
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []


def test_access_restricted_says_the_account_is_not_approved_yet():
    response = httpx.Response(
        403,
        json=_envelope(
            "access_restricted",
            "Access to EigenOrbit Router is restricted to approved accounts",
        ),
    )

    err = RouterError.from_response(response)

    assert (err.code, err.http_status) == ("access_restricted", 403)
    assert err.message.startswith(
        "Access to EigenOrbit Router is restricted to approved accounts"
    )
    assert "not approved for the router yet" in err.message
    assert "support@eigenorbit.com" in err.message


def test_access_restricted_is_not_retried(fake_router):
    fake_router.fail_next(
        "GET",
        f"/v1/jobs/{JOB_ID}",
        status=403,
        body=_envelope("access_restricted", "restricted"),
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.get_job(JOB_ID)

    assert caught.value.code == "access_restricted"
    assert len(_sent(fake_router, "GET", f"/v1/jobs/{JOB_ID}")) == 1
    assert fake_router.sleeps == []


def test_user_blocked_on_a_job_route_says_the_account_is_blocked(fake_router):
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=403,
        body=_envelope("user_blocked", "This EigenOrbit account is blocked"),
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.submit(_body(), idempotency_key="call-1")

    err = caught.value
    assert (err.code, err.http_status) == ("user_blocked", 403)
    assert "account is blocked" in err.message
    assert "will not help" in err.message
    assert "support@eigenorbit.com" in err.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []


def test_result_expired_names_expired_at_and_the_retention(fake_router):
    fake_router.fail_next(
        "GET",
        f"/v1/jobs/{JOB_ID}/result",
        status=410,
        body=_envelope(
            "result_expired",
            "This job's result was deleted after the 90-day retention period",
            {"expired_at": EXPIRED_AT},
        ),
    )
    http = RouterHttpClient(_config())

    with pytest.raises(RouterError) as caught:
        http.get_result(JOB_ID)

    err = caught.value
    assert (err.code, err.http_status) == ("result_expired", 410)
    assert err.details == {"expired_at": EXPIRED_AT}
    assert err.job_id == JOB_ID
    assert EXPIRED_AT in err.message
    assert "kept for 90 days" in err.message
    assert len(_sent(fake_router, "GET", f"/v1/jobs/{JOB_ID}/result")) == 1
    assert fake_router.sleeps == []


@pytest.mark.parametrize("details", [{}, {"expired_at": 1234}, {"expired_at": ""}])
def test_result_expired_without_a_usable_expired_at_still_explains_the_retention(details):
    response = httpx.Response(
        410, json=_envelope("result_expired", "This job's result was deleted", details)
    )

    err = RouterError.from_response(response)

    assert err.code == "result_expired"
    assert err.message.startswith("This job's result was deleted")
    assert "kept for 90 days" in err.message
    assert "1234" not in err.message


# ---- polling ----


def test_poll_timeout_names_the_job(fake_router, clock):
    fake_router.status_steps = ("queued", "running")
    handle = _handle(poll_timeout_seconds=40)
    job_id = handle.job_id()

    with pytest.raises(RouterError) as caught:
        handle.result()

    err = caught.value
    assert fake_router.sleeps == [1.0, 5.0, 30.0, 4.0]
    assert clock[0] == 40.0
    # One GET after each sleep, the last at the deadline.
    assert len(_sent(fake_router, "GET", f"/v1/jobs/{job_id}")) == 5
    assert err.code == "poll_timeout"
    assert err.job_id == job_id
    assert err.http_status is None
    assert job_id in err.message
    assert "keeps running" in err.message
    assert err.details == {"status": "running", "poll_timeout_seconds": 40}
    # The job is left alone: no cancel, no second submit.
    assert _sent(fake_router, "POST", f"/v1/jobs/{job_id}/cancel") == []
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_a_job_completing_on_the_deadline_get_returns_its_result(fake_router, clock):
    fake_router.status_steps = ("queued", "running", "running", "running", "completed")
    handle = _handle(poll_timeout_seconds=40)

    got = handle.result()

    want = NumpySimulator().compile(SPEC, OBSERVABLES).run(WEIGHTS, NO_INPUTS)
    assert fake_router.sleeps == [1.0, 5.0, 30.0, 4.0]
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_ctrl_c_does_not_cancel(fake_router, monkeypatch):
    handle = _handle()
    job_id = handle.job_id()

    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(_client, "_sleep", interrupted)

    with pytest.raises(KeyboardInterrupt) as caught:
        handle.result()

    message = str(caught.value)
    assert job_id in message
    assert "no auto-cancel" in message
    assert caught.value.job_id == job_id
    assert _sent(fake_router, "POST", f"/v1/jobs/{job_id}/cancel") == []
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


def test_result_can_be_called_again_after_a_timeout(fake_router, clock):
    fake_router.status_steps = ("queued", "running")
    handle = _handle(poll_timeout_seconds=40)
    with pytest.raises(RouterError, match="keeps running"):
        handle.result()

    fake_router.status_steps = ("completed",)
    got = handle.result()

    want = NumpySimulator().compile(SPEC, OBSERVABLES).run(WEIGHTS, NO_INPUTS)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert _sent(fake_router, "POST", f"/v1/jobs/{handle.job_id()}/cancel") == []


# ---- IBM refusals and IBM job codes ----

EXAMPLES = Path(__file__).resolve().parent / "contract" / "v1" / "examples"
IBM_ACCOUNT_PAGE = "IBM account page (/ibm) of the router web app"


def _example(stem: str) -> dict:
    return json.loads((EXAMPLES / f"{stem}.json").read_text())


@pytest.mark.parametrize("code", ["ibm_account_not_connected", "ibm_connection_invalid"])
def test_an_ibm_admission_refusal_names_the_ibm_account_page_and_is_not_retried(
    fake_router, code
):
    example = _example(f"error_{code}")
    fake_router.fail_next("POST", "/v1/jobs", status=403, body=example)
    http = RouterHttpClient(_config(device="ibm/ibm_fez", shots=100))

    with pytest.raises(RouterError) as caught:
        http.submit(_body(device="ibm/ibm_fez", shots=100), idempotency_key="call-1")

    err = caught.value
    assert (err.code, err.http_status) == (code, 403)
    assert err.message.startswith(example["error"]["message"])
    assert IBM_ACCOUNT_PAGE in err.message
    assert err.details == example["error"]["details"]
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
    assert fake_router.sleeps == []


def test_not_connected_names_the_device_and_never_mentions_the_eigenorbit_account():
    err = RouterError.from_response(
        httpx.Response(403, json=_example("error_ibm_account_not_connected"))
    )

    assert "connect your IBM Quantum account" in err.message
    assert "ibm/ibm_fez" in err.message
    assert "Retrying will not help" in err.message
    assert "EigenOrbit" not in err.message


def test_connection_invalid_says_to_reconnect_and_never_mentions_the_eigenorbit_account():
    err = RouterError.from_response(
        httpx.Response(403, json=_example("error_ibm_connection_invalid"))
    )

    assert "Reconnect" in err.message
    assert "until you reconnect" in err.message
    assert "EigenOrbit" not in err.message


def test_the_double_refuses_an_ibm_submit_with_the_contract_403(fake_router):
    fake_router.ibm_refusal = "ibm_account_not_connected"
    http = RouterHttpClient(_config(device="ibm/ibm_fez", shots=100))

    with pytest.raises(RouterError) as caught:
        http.submit(_body(device="ibm/ibm_fez", shots=100), idempotency_key="call-1")

    assert caught.value.code == "ibm_account_not_connected"
    assert caught.value.details == {"device": "ibm/ibm_fez"}
    assert IBM_ACCOUNT_PAGE in caught.value.message
    assert fake_router.jobs == {}
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1


IBM_JOB_HINTS = {
    "ibm_credentials_rejected": IBM_ACCOUNT_PAGE,
    "ibm_quota_exhausted": "quantum time",
    "ibm_target_mismatch": "never resubmits",
    "ibm_device_not_on_instance": IBM_ACCOUNT_PAGE,
}


@pytest.mark.parametrize("code", sorted(IBM_JOB_HINTS))
def test_an_ibm_job_code_is_a_vendor_failure_with_a_hint(code):
    err = RouterError.from_job(
        {
            "id": JOB_ID,
            "status": "failed",
            "billing": None,
            "error": {"code": code, "message": "IBM said no.", "category": "vendor"},
        }
    )

    assert (err.code, err.category) == (code, "vendor")
    assert err.message.startswith("IBM said no.")
    assert IBM_JOB_HINTS[code] in err.message
    assert err.details == {"status": "failed", "billing": None}


def test_the_target_mismatch_example_maps_to_a_vendor_failure_with_its_hint():
    example = _example("job_status_ibm_failed_target_mismatch")

    err = RouterError.from_job(example)

    assert (err.code, err.category) == ("ibm_target_mismatch", "vendor")
    assert err.message.startswith(example["error"]["message"])
    assert "never resubmits" in err.message


def test_an_ibm_job_failing_at_the_vendor_raises_its_code_once(fake_router):
    fake_router.fail_with = "ibm_quota_exhausted"
    fake_router.status_steps = ("completed",)
    compiled = RouterBackend(_config(device="ibm/ibm_fez", shots=100)).compile(
        SPEC, OBSERVABLES
    )
    handle = compiled.submit(WEIGHTS, NO_INPUTS, method="run")

    with pytest.raises(RouterError) as caught:
        handle.result()

    assert (caught.value.code, caught.value.category) == ("ibm_quota_exhausted", "vendor")
    assert "quantum time" in caught.value.message
    assert len(_sent(fake_router, "POST", "/v1/jobs")) == 1
