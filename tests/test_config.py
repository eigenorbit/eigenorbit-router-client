"""RouterConfig: construction rules, key and URL resolution, transport safety.

Circuit-level tests wrap both construction and the call where they expect an
error, and filter recorded requests by method and path, because a later plan
moves the first router request (the estimate) into compile.
"""

import dataclasses

import httpx
import numpy as np
import pytest
from eigenkit import Circuit
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.observables import all_z

from eigenorbit_router_client import (
    DEFAULT_BASE_URL,
    RouterConfig,
    RouterError,
    _client,
)
from eigenorbit_router_client.config import (
    API_KEY_ENV,
    resolve_api_key,
    resolve_base_url,
)
from fake_router import VALID_KEY


SPEC = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
OBSERVABLES = all_z(2)
BASE_URL = "https://router.test"
KEY_A = "eo_test_" + "a" * 43
KEY_B = "eo_test_" + "b" * 43


def _params():
    rng = np.random.default_rng(11)
    return rng.uniform(-1, 1, SPEC.n_weights), rng.uniform(0, 1, SPEC.n_inputs)


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _submits(router):
    return _sent(router, "POST", "/v1/jobs")


# The environment variable name and keyword-only fields.


def test_the_api_key_environment_variable_is_eigenorbit_router_api_key():
    assert API_KEY_ENV == "EIGENORBIT_ROUTER_API_KEY"


def test_the_key_comes_from_eigenorbit_router_api_key(fake_router, monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", KEY_A)
    config = RouterConfig(device="braket/sv1", base_url=BASE_URL)

    Circuit(SPEC, OBSERVABLES, provider=config)(*_params())

    [submit] = _submits(fake_router)
    assert submit.headers["Authorization"] == f"Bearer {KEY_A}"


def test_the_missing_key_message_names_eigenorbit_router_api_key(monkeypatch):
    monkeypatch.delenv("EIGENORBIT_ROUTER_API_KEY", raising=False)

    with pytest.raises(RouterError) as caught:
        resolve_api_key(RouterConfig(device="braket/sv1"))

    assert "EIGENORBIT_ROUTER_API_KEY" in caught.value.message


def test_device_may_be_positional_and_the_rest_by_keyword():
    assert RouterConfig("braket/sv1").device == "braket/sv1"
    assert RouterConfig(device="braket/sv1", shots=100).shots == 100


def test_fields_after_device_are_keyword_only():
    with pytest.raises(TypeError):
        RouterConfig("braket/sv1", 100)
    kw_only = {f.name: f.kw_only for f in dataclasses.fields(RouterConfig)}
    assert kw_only.pop("device") is False
    assert all(kw_only.values()), kw_only


def test_the_api_key_cannot_be_passed_positionally():
    with pytest.raises(TypeError):
        RouterConfig("braket/sv1", 0, VALID_KEY)


def test_router_config_stays_frozen():
    config = RouterConfig("braket/sv1")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.shots = 5


# Construction rules.


@pytest.mark.parametrize(
    "device",
    ["", "sv1", "Braket/sv1", "braket/SV1", "braket/sv 1", "bräket/sv1", "braket/sv1/x"]
    + ["braket/" + "a" * 58],
)
def test_a_malformed_device_is_refused_at_construction(device):
    with pytest.raises(ValueError, match="device"):
        RouterConfig(device=device)


@pytest.mark.parametrize(
    "device",
    ["braket/sv1", "braket/ionq_forte", "ibm/ibm_torino", "braket/" + "a" * 57],
)
def test_a_well_formed_device_is_accepted(device):
    assert RouterConfig(device=device).device == device


def test_a_non_string_device_is_refused():
    with pytest.raises(ValueError, match="device"):
        RouterConfig(device=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shots", -1),
        ("shots", 1.5),
        ("shots", True),
        ("poll_timeout_seconds", 0),
        ("poll_timeout_seconds", -5),
        ("max_concurrent_submits", 0),
        ("max_concurrent_submits", 2.0),
        ("request_timeout_seconds", 0),
        ("request_timeout_seconds", -0.5),
        ("request_timeout_seconds", False),
    ],
)
def test_an_out_of_range_setting_is_refused_and_named(field, value):
    with pytest.raises(ValueError, match=field):
        RouterConfig(device="braket/sv1", **{field: value})


def test_in_range_settings_are_accepted():
    config = RouterConfig(
        device="braket/sv1",
        shots=100,
        poll_timeout_seconds=1,
        max_concurrent_submits=1,
        request_timeout_seconds=0.5,
    )

    assert (config.shots, config.max_concurrent_submits) == (100, 1)


def test_the_api_key_never_shows_in_repr():
    text = repr(RouterConfig(device="braket/sv1", api_key=VALID_KEY))

    assert "eo_test_" not in text
    assert "braket/sv1" in text


# API key resolution.


def test_the_config_key_beats_the_environment(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", KEY_B)

    assert resolve_api_key(RouterConfig(device="braket/sv1", api_key=KEY_A)) == KEY_A


def test_the_environment_key_is_used_when_the_config_key_is_none(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", KEY_B)

    assert resolve_api_key(RouterConfig(device="braket/sv1")) == KEY_B


def test_surrounding_whitespace_is_stripped_from_the_key(fake_router, monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", f"  {KEY_B}  \n")
    padded = RouterConfig(
        device="braket/sv1", api_key=f"  {KEY_A}  \n", base_url=BASE_URL
    )
    from_env = RouterConfig(device="braket/sv1", base_url=BASE_URL)

    Circuit(SPEC, OBSERVABLES, provider=padded)(*_params())
    Circuit(SPEC, OBSERVABLES, provider=from_env)(*_params())

    assert [r.headers["Authorization"] for r in _submits(fake_router)] == [
        f"Bearer {KEY_A}",
        f"Bearer {KEY_B}",
    ]


@pytest.mark.parametrize(
    ("api_key", "env"),
    [(None, None), ("", None), ("  \n", None), (None, ""), (None, "   ")],
)
def test_a_missing_key_fails_before_any_request(fake_router, monkeypatch, api_key, env):
    if env is not None:
        monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", env)
    config = RouterConfig(device="braket/sv1", api_key=api_key, base_url=BASE_URL)

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=config)(*_params())

    assert caught.value.code == "unauthenticated"
    assert caught.value.http_status is None
    assert fake_router.requests == []
    assert "eo_test_" not in str(caught.value)


# Base URL resolution.


def test_the_config_base_url_beats_the_environment(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", "https://env.test")
    config = RouterConfig(device="braket/sv1", base_url="https://config.test/")

    assert resolve_base_url(config) == "https://config.test"


def test_the_environment_base_url_beats_the_default(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", "https://env.test/")

    assert resolve_base_url(RouterConfig(device="braket/sv1")) == "https://env.test"


def test_the_default_base_url_is_the_package_constant():
    assert resolve_base_url(RouterConfig(device="braket/sv1")) == DEFAULT_BASE_URL
    assert DEFAULT_BASE_URL == "https://router.eigenorbit.com"


@pytest.mark.parametrize(
    "url", ["http://router.example", "ftp://router.example", "router.example", "https://"]
)
def test_a_cleartext_or_malformed_base_url_is_refused(url, monkeypatch):
    with pytest.raises(ValueError, match="base_url"):
        RouterConfig(device="braket/sv1", base_url=url)
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", url)
    with pytest.raises(ValueError, match="EIGENORBIT_ROUTER_URL"):
        resolve_base_url(RouterConfig(device="braket/sv1"))


def test_a_cleartext_environment_url_fails_before_any_request(fake_router, monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", "http://router.example")
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY)

    with pytest.raises(ValueError, match="https"):
        Circuit(SPEC, OBSERVABLES, provider=config)(*_params())

    assert fake_router.requests == []


@pytest.mark.parametrize(
    "url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"]
)
def test_plain_http_is_allowed_for_loopback_only(url):
    assert resolve_base_url(RouterConfig(device="braket/sv1", base_url=url)) == url


# Per-request resolution (concurrency edge).


def test_interleaved_circuits_each_send_their_own_key(fake_router):
    circuit_a = Circuit(
        SPEC,
        OBSERVABLES,
        provider=RouterConfig(device="braket/sv1", api_key=KEY_A, base_url=BASE_URL),
    )
    circuit_b = Circuit(
        SPEC,
        OBSERVABLES,
        provider=RouterConfig(device="braket/sv1", api_key=KEY_B, base_url=BASE_URL),
    )
    weights, inputs = _params()

    circuit_a(weights, inputs)
    circuit_b(weights, inputs)
    circuit_a(weights, inputs)

    assert [r.headers["Authorization"] for r in _submits(fake_router)] == [
        f"Bearer {KEY_A}",
        f"Bearer {KEY_B}",
        f"Bearer {KEY_A}",
    ]


def test_an_environment_change_between_calls_is_picked_up(fake_router, monkeypatch):
    circuit = Circuit(
        SPEC, OBSERVABLES, provider=RouterConfig(device="braket/sv1", base_url=BASE_URL)
    )
    weights, inputs = _params()

    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", KEY_A)
    circuit(weights, inputs)
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", KEY_B)
    circuit(weights, inputs)

    assert [r.headers["Authorization"] for r in _submits(fake_router)] == [
        f"Bearer {KEY_A}",
        f"Bearer {KEY_B}",
    ]


def test_the_key_is_not_stored_on_the_compiled_circuit(fake_router):
    config = RouterConfig(device="braket/sv1", api_key=KEY_A, base_url=BASE_URL)
    circuit = Circuit(SPEC, OBSERVABLES, provider=config)
    circuit(*_params())

    compiled = circuit.compiled
    held = {**vars(compiled), **vars(compiled._http)}
    assert all(KEY_A not in repr(value) for value in held.values())


# Redirects are never followed.


@pytest.mark.parametrize("body", [b"", b'{"error": {"code": "moved", "message": "m"}}'])
def test_a_redirect_is_not_followed(fake_router, monkeypatch, body):
    def redirecting(request):
        if request.url.host == "router.test":
            fake_router.requests.append(request)
            return httpx.Response(
                302,
                headers={"Location": "https://elsewhere.test/v1/jobs"},
                content=body,
            )
        return fake_router.handle(request)

    def make_http_client(*, base_url, timeout):
        return httpx.Client(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=False,
            transport=httpx.MockTransport(redirecting),
        )

    monkeypatch.setattr(_client, "_make_http_client", make_http_client)
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY, base_url=BASE_URL)

    with pytest.raises(RouterError) as caught:
        Circuit(SPEC, OBSERVABLES, provider=config)(*_params())

    assert caught.value.code == "invalid_response"
    assert caught.value.http_status == 302
    assert [r.url.host for r in fake_router.requests] == ["router.test"]


def test_the_real_http_client_does_not_follow_redirects():
    client = _client._make_http_client(base_url=BASE_URL, timeout=1.0)
    try:
        assert client.follow_redirects is False
    finally:
        client.close()


@pytest.mark.parametrize(
    ("device", "is_qpu"),
    [("braket/sv1", False), ("braket/dm1", False), ("braket/tn1", True)],
)
def test_only_catalog_simulators_count_as_simulators(device, is_qpu):
    # TN1 is retired and not in the v1 catalog, so it is not a known simulator.
    assert RouterConfig(device=device).is_qpu is is_qpu
