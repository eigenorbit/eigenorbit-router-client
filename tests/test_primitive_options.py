"""RouterConfig.primitive_options: sent as the job request's `ibm_options`.

The field has the name, type and default of eigenkit's
`QiskitRuntimeConfig.primitive_options`, so the same code runs on
`provider="qiskit-runtime"` and on the router. The client checks only that it
is a mapping and that the device is `ibm/`; the router checks the option names
and values, and its 422 gets a hint that names what it accepts. The far end is
the in-process FakeRouter.
"""

import json
from pathlib import Path
from types import MappingProxyType

import httpx
import numpy as np
import pytest
from eigenkit import Circuit
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z

from eigenorbit_router_client import RouterConfig, RouterError
from eigenorbit_router_client.config import ibm_options
from fake_router import REQUEST_FIELDS, VALID_KEY


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
EXAMPLES = Path(__file__).resolve().parent / "contract" / "v1" / "examples"
IBM = "ibm/ibm_fez"
DD_AND_TWIRLING = {
    "dynamical_decoupling": {"enable": True, "sequence_type": "XpXm"},
    "twirling": {
        "enable_gates": True,
        "num_randomizations": 4,
        "shots_per_randomization": 25,
    },
}
ALLOWED_KEYS = (
    "dynamical_decoupling.enable",
    ".sequence_type",
    "twirling.enable_gates",
    ".enable_measure",
    ".num_randomizations",
    ".shots_per_randomization",
)
SHOTS_RULE = "num_randomizations * shots_per_randomization must equal RouterConfig.shots"


def _config(device=IBM, **overrides) -> RouterConfig:
    settings = {"api_key": VALID_KEY, "base_url": BASE_URL, "shots": 100}
    settings.update(overrides)
    return RouterConfig(device, **settings)


def _circuit(device=IBM, **overrides) -> Circuit:
    return Circuit(SPEC, OBSERVABLES, provider=_config(device, **overrides))


def _sent(router, method, path):
    return [r for r in router.requests if r.method == method and r.url.path == path]


def _bodies(router, path):
    return [json.loads(r.content) for r in _sent(router, "POST", path)]


def _hint_of(details, *, sent=True) -> str:
    response = httpx.Response(
        422,
        json={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": details,
            }
        },
    )
    error = RouterError.from_response(response, sent_ibm_options=sent)
    assert (error.code, error.http_status) == ("validation_error", 422)
    assert error.details == details
    return error.message


def _assert_names_the_options(message: str) -> None:
    assert message.startswith("Request validation failed. ")
    for key in ALLOWED_KEYS:
        assert key in message, key
    assert "'XX', 'XpXm' or 'XY4'" in message
    assert SHOTS_RULE in message


# ---- the field ----


def test_the_default_is_an_empty_dict_per_config():
    first, second = RouterConfig(IBM), RouterConfig(IBM)

    assert first.primitive_options == {}
    assert type(first.primitive_options) is dict
    assert first.primitive_options is not second.primitive_options


def test_options_are_accepted_on_an_ibm_device():
    assert _config(primitive_options=DD_AND_TWIRLING).primitive_options == DD_AND_TWIRLING


@pytest.mark.parametrize("device", ["braket/sv1", "braket/iqm_garnet"])
def test_options_on_a_non_ibm_device_raise_before_any_request(fake_router, device):
    with pytest.raises(ValueError, match="primitive_options .* IBM devices") as caught:
        _circuit(device, primitive_options={"twirling": {"enable_gates": True}})

    assert device in str(caught.value)
    assert fake_router.requests == []


def test_empty_options_on_a_non_ibm_device_are_fine():
    assert RouterConfig("braket/sv1", primitive_options={}).primitive_options == {}


@pytest.mark.parametrize("value", [None, ["twirling"], "twirling"])
def test_options_that_are_not_a_mapping_raise_type_error(value):
    with pytest.raises(TypeError, match="primitive_options must be a mapping"):
        RouterConfig(IBM, primitive_options=value)


def test_options_added_to_a_non_ibm_config_later_raise_before_any_request(fake_router):
    options = {}
    config = _config("braket/sv1", primitive_options=options)
    options["twirling"] = {"enable_gates": True}

    with pytest.raises(ValueError, match="primitive_options"):
        Circuit(SPEC, OBSERVABLES, provider=config)

    assert fake_router.requests == []


# ---- the request body ----


def test_options_are_sent_as_ibm_options_to_the_estimate_and_the_submit(fake_router):
    circuit = _circuit(primitive_options=DD_AND_TWIRLING)

    circuit.compiled.submit(WEIGHTS, NO_INPUTS, method="sample").result()

    [estimate] = _bodies(fake_router, "/v1/jobs/estimate")
    [submit] = _bodies(fake_router, "/v1/jobs")
    for body in (estimate, submit):
        assert body["ibm_options"] == DD_AND_TWIRLING
        assert set(body) == REQUEST_FIELDS | {"ibm_options"}
    assert submit["method"] == "sample"


def test_the_options_do_not_change_the_rest_of_the_request(fake_router):
    plain = _circuit().compiled.build_request("run", WEIGHTS, NO_INPUTS)
    with_options = _circuit(primitive_options=DD_AND_TWIRLING).compiled.build_request(
        "run", WEIGHTS, NO_INPUTS
    )

    assert with_options.pop("ibm_options") == DD_AND_TWIRLING
    assert with_options == plain


@pytest.mark.parametrize("device", [IBM, "braket/sv1"])
def test_empty_options_send_no_ibm_options_field(fake_router, device):
    _circuit(device)(WEIGHTS)

    requests = _sent(fake_router, "POST", "/v1/jobs/estimate") + _sent(
        fake_router, "POST", "/v1/jobs"
    )
    assert len(requests) == 2
    for request in requests:
        assert b"ibm_options" not in request.content
        assert set(json.loads(request.content)) == REQUEST_FIELDS


def test_the_client_does_not_check_the_option_names():
    # The router decides what it accepts; the client forwards the mapping as is.
    options = {"resilience_level": 1, "twirling": {"strategy": "active"}}

    assert ibm_options(RouterConfig(IBM, primitive_options=options)) == options


def test_a_read_only_mapping_is_sent_as_plain_json(fake_router):
    options = MappingProxyType(
        {"dynamical_decoupling": MappingProxyType({"enable": True})}
    )

    _circuit(primitive_options=options)(WEIGHTS)

    [submit] = _bodies(fake_router, "/v1/jobs")
    assert submit["ibm_options"] == {"dynamical_decoupling": {"enable": True}}


def test_every_row_of_a_batch_carries_the_options(fake_router):
    circuit = _circuit(primitive_options=DD_AND_TWIRLING)

    values = circuit.compiled.run(np.vstack([WEIGHTS, -WEIGHTS]), NO_INPUTS)

    assert values.shape == (2, 2)
    submits = _bodies(fake_router, "/v1/jobs")
    assert len(submits) == 2
    assert all(body["ibm_options"] == DD_AND_TWIRLING for body in submits)


# ---- the router's 422 about the options ----


def test_twirling_shots_that_do_not_multiply_to_the_shots_get_the_hint(fake_router):
    options = {"twirling": {"num_randomizations": 4, "shots_per_randomization": 32}}

    with pytest.raises(RouterError) as caught:
        _circuit(primitive_options=options)

    err = caught.value
    assert (err.code, err.http_status) == ("validation_error", 422)
    assert err.details["errors"][0]["loc"] == ["body"]
    _assert_names_the_options(err.message)
    assert _sent(fake_router, "POST", "/v1/jobs") == []


@pytest.mark.parametrize(
    "options",
    [
        {"resilience_level": 1},
        {"twirling": {"strategy": "active"}},
        {"dynamical_decoupling": {"sequence_type": "XYXY"}},
    ],
)
def test_an_option_the_router_refuses_gets_the_hint(fake_router, options):
    with pytest.raises(RouterError) as caught:
        _circuit(primitive_options=options)

    err = caught.value
    assert (err.code, err.http_status) == ("validation_error", 422)
    assert err.details["errors"][0]["loc"][:2] == ["body", "ibm_options"]
    _assert_names_the_options(err.message)


def test_a_router_without_ibm_options_says_so(fake_router):
    fake_router.ibm_options_supported = False

    with pytest.raises(RouterError) as caught:
        _circuit(primitive_options={"dynamical_decoupling": {"enable": True}})

    err = caught.value
    assert err.details["errors"][0] == {
        "loc": ["body", "ibm_options"],
        "msg": "Extra inputs are not permitted",
        "type": "extra_forbidden",
    }
    assert "does not accept RouterConfig.primitive_options" in err.message
    assert "leave primitive_options empty" in err.message


def test_a_submit_refused_for_the_options_gets_the_hint(fake_router):
    compiled = _circuit(primitive_options=DD_AND_TWIRLING).compiled
    fake_router.fail_next(
        "POST",
        "/v1/jobs",
        status=422,
        body={
            "error": {
                "code": "validation_error",
                "message": "ibm_options are not valid for this job",
                "details": {},
            }
        },
    )

    with pytest.raises(RouterError) as caught:
        compiled.submit(WEIGHTS, NO_INPUTS)

    assert caught.value.message.startswith("ibm_options are not valid for this job. ")
    assert SHOTS_RULE in caught.value.message


def test_an_error_naming_ibm_options_gets_the_hint_without_the_request():
    details = {
        "errors": [
            {
                "loc": ["body", "ibm_options", "twirling", "num_randomizations"],
                "msg": "Input should be greater than 0",
                "type": "greater_than",
            }
        ]
    }

    _assert_names_the_options(_hint_of(details, sent=False))


def test_an_error_on_another_field_gets_no_hint_even_with_options():
    details = {"errors": [{"loc": ["body", "shots"], "msg": "bad", "type": "value_error"}]}

    assert _hint_of(details) == "Request validation failed"


def test_a_whole_body_error_without_options_keeps_the_router_message():
    details = {"errors": [{"loc": ["body"], "msg": "bad", "type": "value_error"}]}

    assert _hint_of(details, sent=False) == "Request validation failed"


def test_the_contract_validation_example_gets_no_hint():
    details = json.loads((EXAMPLES / "error_validation.json").read_text())["error"][
        "details"
    ]

    assert _hint_of(details) == "Request validation failed"
