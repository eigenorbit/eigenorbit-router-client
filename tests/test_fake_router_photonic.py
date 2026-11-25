import httpx
import pytest

from fake_router import DEVICE_ACCESS_MESSAGE, VALID_KEY, FakeRouter, fock_counts_oracle


HEADERS = {"EO-Protocol-Version": "1", "Authorization": f"Bearer {VALID_KEY}"}
CIRCUIT = {
    "n_modes": 4, "n_weights": 1, "n_inputs": 1, "input_state": [1, 0, 1, 0],
    "components": [
        {"c": "PS", "mode": 1, "phi": {"t": "InputRef", "i": 0}},
        {"c": "BS", "modes": [0, 1], "theta": {"t": "WeightRef", "i": 0},
         "phi": {"t": "Constant", "v": 0.0}},
        {"c": "PERM", "perm": [1, 0, 2, 3]},
    ],
}


def body(**over):
    request = {"device": "quandela/sim_slos", "method": "photonic_sample", "shots": 537,
               "circuit_spec": CIRCUIT, "observables": [], "weights": [0.3],
               "inputs": [0.276]}
    request.update(over)
    return request


@pytest.fixture
def double():
    router = FakeRouter()
    with httpx.Client(base_url="https://router.test", transport=router.transport) as http:
        yield router, http


def submit(http, request, key="k"):
    return http.post("/v1/jobs", json=request, headers={**HEADERS, "Idempotency-Key": key})


def finished(http, job_id):
    for _ in range(3):
        job = http.get(f"/v1/jobs/{job_id}", headers=HEADERS).json()
    return job, http.get(f"/v1/jobs/{job_id}/result", headers=HEADERS).json()


def test_the_oracle_gives_the_counts_of_the_spec_example():
    assert fock_counts_oracle([1, 0, 1, 0], [0.276], 537) == {"1,0,1,0": 261, "0,1,0,1": 276}
    assert fock_counts_oracle([1, 0, 1, 0], [0.0], 498) == {"1,0,1,0": 498}


def test_a_photonic_job_is_free_and_returns_fock_counts(double):
    _, http = double
    created = submit(http, body())
    job, result = finished(http, created.json()["id"])
    assert created.status_code == 202
    assert (job["status"], job["method"]) == ("completed", "photonic_sample")
    assert (job["billing"], job["estimate_usd_micro"]) == (None, 0)
    assert (job["credential_source"], job["ibm_usage_seconds"]) == ("organization", None)
    assert result["method"] == "photonic_sample" and result["bit_order"] == "qubit0_msb"
    assert result["result"] == {"kind": "fock_counts", "n_modes": 4,
                                "data": [{"1,0,1,0": 261, "0,1,0,1": 276}]}
    execution = result["execution"]
    assert isinstance(execution["vendor_job_id"], str) and execution["vendor_job_id"]
    assert len(execution) > 1
    assert all(value is None for key, value in execution.items() if key != "vendor_job_id")


def test_two_d_inputs_give_one_object_per_row_in_request_order(double):
    router, http = double
    router.empty_rows = frozenset({2})
    rows = [[0.276], [0.0], [0.1]]
    _, result = finished(http, submit(http, body(inputs=rows)).json()["id"])
    assert result["result"]["data"] == [
        fock_counts_oracle([1, 0, 1, 0], rows[0], 537),
        fock_counts_oracle([1, 0, 1, 0], rows[1], 537),
        {},
    ]


def test_rows_above_the_device_limit_are_refused_with_max_rows(double):
    _, http = double
    refused = submit(http, body(device="quandela/sim_belenos", inputs=[[0.0]] * 17))
    accepted = submit(http, body(device="quandela/sim_belenos", inputs=[[0.0]] * 16), "k2")
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "batch_not_supported"
    assert refused.json()["error"]["details"] == {"max_rows": 16}
    assert accepted.status_code == 202


@pytest.mark.parametrize(
    ("over", "code", "details"),
    [
        ({"observables": [{}]}, "validation_error", None),
        ({"method": "run", "observables": [{}]}, "method_not_supported", None),
        ({"device": "braket/sv1"}, "method_not_supported", None),
        ({"shots": 0}, "shots_invalid", {"min_shots": 1, "max_shots": 1_000_000}),
        ({"device": "quandela/qpu_ascella", "shots": 100_001}, "shots_invalid",
         {"min_shots": 1, "max_shots": 100_000}),
        ({"inputs": [[0.1, 0.2]]}, "circuit_invalid", None),
        ({"circuit_spec": {"n_modes": 4}}, "circuit_invalid", None),
        ({"weights": [[0.3]]}, "batch_not_supported", {}),
    ],
)
def test_a_wrong_photonic_request_is_refused(double, over, code, details):
    _, http = double
    error = submit(http, body(**over)).json()["error"]
    assert error["code"] == code
    assert details is None or error["details"] == details


def test_an_account_outside_the_preview_is_refused(double):
    router, http = double
    router.quandela_access = False
    refused = submit(http, body())
    estimate = http.post("/v1/jobs/estimate", json=body(), headers=HEADERS)
    assert refused.status_code == 403
    assert refused.json()["error"] == {"code": "device_access_restricted",
                                       "message": DEVICE_ACCESS_MESSAGE,
                                       "details": {"device": "quandela/sim_slos"}}
    assert estimate.json()["would_reject_reason"] == "device_access_restricted"
    assert estimate.json()["estimate_usd_micro"] == 0
