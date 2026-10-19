"""Tracer: a kit Circuit runs through RouterConfig end to end.

The path under test is the real one: kit `Circuit` -> provider resolution ->
`RouterBackend.compile` -> kit wire encoding -> POST /v1/jobs -> polling ->
GET result -> decoding. Only the far end is the in-process FakeRouter.
Request assertions filter by method and path, so later compile-time requests
(the estimate) do not break them.
"""

import json
import re

import numpy as np
import pytest
from eigenkit import Circuit
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.observables import all_z
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

from eigenorbit_router_client import RouterBackend, RouterConfig, RouterError
from eigenorbit_router_client._decode import decode_result
from fake_router import VALID_KEY


SPEC = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
OBSERVABLES = all_z(2)
BASE_URL = "https://router.test"


def _config(**overrides) -> RouterConfig:
    settings = {"device": "braket/sv1", "api_key": VALID_KEY, "base_url": BASE_URL}
    settings.update(overrides)
    return RouterConfig(**settings)


def _params():
    rng = np.random.default_rng(7)
    weights = rng.uniform(-np.pi, np.pi, SPEC.n_weights)
    inputs = rng.uniform(0.0, np.pi, SPEC.n_inputs)
    return weights, inputs


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def test_run_matches_the_numpy_path(fake_router):
    weights, inputs = _params()

    got = Circuit(SPEC, OBSERVABLES, provider=_config())(weights, inputs)
    want = Circuit(SPEC, OBSERVABLES, provider="numpy")(weights, inputs)

    assert got.shape == want.shape == (2,)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    # queued -> running -> completed: the contract backoff, 1 s then 5 s.
    assert fake_router.sleeps == [1.0, 5.0]


def test_submit_carries_the_contract_headers_and_the_kit_encoding(fake_router):
    weights, inputs = _params()

    Circuit(SPEC, OBSERVABLES, provider=_config())(weights, inputs)

    [submit] = _sent(fake_router, "POST", "/v1/jobs")
    assert submit.headers["Authorization"] == f"Bearer {VALID_KEY}"
    assert submit.headers["EO-Protocol-Version"] == "1"
    assert re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", submit.headers["Idempotency-Key"])
    assert submit.headers["User-Agent"].startswith("eigenorbit-router-client/")
    body = json.loads(submit.content)
    assert body["circuit_spec"] == circuit_spec_to_json(SPEC)
    assert body["observables"] == observables_to_json(OBSERVABLES)
    assert body["device"] == "braket/sv1"
    assert body["method"] == "run"
    assert body["shots"] == 0
    assert body["weights"] == weights.tolist()
    assert body["inputs"] == inputs.tolist()
    for request in fake_router.requests:
        assert request.headers["Authorization"] == f"Bearer {VALID_KEY}"
        assert request.headers["EO-Protocol-Version"] == "1"
        assert request.url.host == "router.test"


def test_a_failed_job_raises_router_error_with_its_code_and_id(fake_router):
    fake_router.fail_with = "vendor_failed"
    weights, inputs = _params()

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=_config())(weights, inputs)

    [submit] = _sent(fake_router, "POST", "/v1/jobs")
    [job_id] = fake_router.jobs
    assert caught.value.code == "vendor_failed"
    assert caught.value.job_id == job_id
    assert caught.value.http_status is None
    assert caught.value.details == {"status": "failed", "billing": "released"}
    assert isinstance(caught.value, RuntimeError)
    assert job_id in str(caught.value)
    assert _sent(fake_router, "GET", f"/v1/jobs/{job_id}/result") == []
    assert submit.headers["Idempotency-Key"]


def test_an_unknown_device_raises_router_error_unknown_device(fake_router):
    weights, inputs = _params()

    with pytest.raises(RouterError) as caught:
        circuit = Circuit(SPEC, OBSERVABLES, provider=_config(device="braket/nope"))
        circuit(weights, inputs)

    assert caught.value.code == "unknown_device"
    assert caught.value.http_status == 422
    assert caught.value.details == {"device": "braket/nope"}
    assert fake_router.jobs == {}


def test_a_bare_router_backend_asks_for_a_device():
    with pytest.raises(ValueError, match=r"RouterConfig\(device='braket/sv1'\)"):
        RouterBackend()


def test_braket_devices_get_the_braket_gate_set():
    braket = RouterBackend(_config()).capabilities.supported_gates
    # A vendor with no kit gate set of its own (IBM has one).
    other = RouterBackend(_config(device="acme/qpu1")).capabilities.supported_gates

    assert "AmplitudeEncoding" not in braket
    assert "AmplitudeEncoding" in other
    assert RouterBackend(_config()).gradient_method == "parameter_shift"
    assert RouterBackend(_config(device="ibm/ibm_torino")).gradient_method == "none"


def _run_payload(**changes):
    payload = {
        "job_id": "j",
        "method": "run",
        "bit_order": "qubit0_msb",
        "result": {
            "kind": "array",
            "dtype": "float64",
            "shape": [2],
            "data": [0.5, -0.25],
        },
    }
    payload.update(changes)
    return payload


def test_decode_result_rebuilds_the_array():
    values = decode_result(_run_payload(), method="run", n_qubits=2, n_observables=2)

    assert values.dtype == np.float64
    np.testing.assert_array_equal(values, [0.5, -0.25])


@pytest.mark.parametrize(
    "changes",
    [
        {"bit_order": "qubit0_lsb"},
        {"method": "probabilities"},
        {
            "result": {
                "kind": "array",
                "dtype": "float64",
                "shape": [3],
                "data": [0, 0, 0],
            }
        },
        {
            "result": {
                "kind": "array",
                "dtype": "float64",
                "shape": [2],
                "data": [[0.5, 0]],
            }
        },
        {"result": {"kind": "counts", "n_qubits": 2, "data": {"00": 1}}},
    ],
)
def test_decode_result_refuses_a_payload_off_contract(changes):
    with pytest.raises(RouterError) as caught:
        decode_result(
            _run_payload(**changes), method="run", n_qubits=2, n_observables=2, job_id="j"
        )

    assert caught.value.code == "invalid_response"
    assert caught.value.job_id == "j"
