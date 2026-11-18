"""The router's terms notice reaches the user as one `TermsWarning`.

While an account is in the grace period of a new version of the router terms,
the router answers a submit and an estimate with an `EO-Terms-Notice` header:
one sentence with the version to accept, where, and the date new jobs stop.
"""

import warnings

import pytest
from eigenkit.core.ir import CircuitSpec, Gate, WeightRef
from eigenkit.core.observables import all_z
from eigenkit.core.wire import circuit_spec_to_json, observables_to_json

from eigenorbit_router_client import RouterConfig, TermsWarning
from eigenorbit_router_client._client import RouterHttpClient
from fake_router import VALID_KEY


NOTICE = (
    "Accept the EigenOrbit Router terms 2027-01-01 at https://router.eigenorbit.com/ "
    "before 2027-01-08 (UTC). From that date this account cannot run new jobs until "
    "you accept them."
)
SPEC = CircuitSpec(
    n_qubits=1, n_weights=1, n_inputs=0, gates=(Gate("RY", (0,), (WeightRef(0),)),)
)
BODY = {
    "device": "braket/sv1",
    "method": "run",
    "shots": 0,
    "circuit_spec": circuit_spec_to_json(SPEC),
    "observables": observables_to_json(all_z(1)),
    "weights": [0.7],
    "inputs": [],
}


def _http() -> RouterHttpClient:
    return RouterHttpClient(
        RouterConfig(
            device="braket/sv1", api_key=VALID_KEY, base_url="https://router.test"
        )
    )


def test_a_submit_with_the_notice_warns_with_its_text(fake_router):
    fake_router.terms_notice = NOTICE

    with pytest.warns(TermsWarning) as caught:
        job = _http().submit(BODY, idempotency_key="k1")

    assert job["id"]
    assert [str(w.message) for w in caught] == [NOTICE]


def test_an_estimate_with_the_notice_warns_too(fake_router):
    fake_router.terms_notice = NOTICE

    with pytest.warns(TermsWarning, match="before 2027-01-08"):
        _http().estimate(BODY)


def test_a_response_without_the_notice_does_not_warn(fake_router):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _http().submit(BODY, idempotency_key="k1")


def test_a_loop_of_submissions_shows_the_notice_once(fake_router):
    fake_router.terms_notice = NOTICE
    http = _http()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for n in range(3):
            http.submit(BODY, idempotency_key=f"k{n}")

    assert [str(w.message) for w in caught if w.category is TermsWarning] == [NOTICE]


def test_polling_a_job_never_warns(fake_router):
    fake_router.terms_notice = NOTICE
    http = _http()
    with pytest.warns(TermsWarning):
        job = http.submit(BODY, idempotency_key="k1")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        http.get_job(job["id"])
