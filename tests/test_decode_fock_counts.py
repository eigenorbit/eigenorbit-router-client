import subprocess
import sys

import pytest

from eigenorbit_router_client import RouterError
from eigenorbit_router_client._decode import decode_fock_counts


def payload(data, n_modes=4, **over):
    body = {
        "job_id": "j1",
        "method": "photonic_sample",
        "bit_order": "qubit0_msb",
        "result": {"kind": "fock_counts", "n_modes": n_modes, "data": data},
    }
    body.update(over)
    return body


def test_the_spec_example_decodes_to_tuples_in_row_order():
    data = [{"1,0,1,0": 261, "0,1,0,1": 276}, {"1,0,1,0": 498}]
    rows = decode_fock_counts(payload(data), n_modes=4, n_rows=2)
    assert rows == [{(1, 0, 1, 0): 261, (0, 1, 0, 1): 276}, {(1, 0, 1, 0): 498}]


def test_an_empty_row_stays_empty():
    rows = decode_fock_counts(payload([{}, {"2,0,0,0": 5}]), n_modes=4, n_rows=2)
    assert rows == [{}, {(2, 0, 0, 0): 5}]


def test_a_multi_digit_photon_count_and_a_zero_count_decode():
    rows = decode_fock_counts(payload([{"10,0": 0}], n_modes=2), n_modes=2, n_rows=1)
    assert rows == [{(10, 0): 0}]


def test_bit_order_is_not_read():
    body = payload([{"1,0": 3}], n_modes=2)
    del body["bit_order"]
    assert decode_fock_counts(body, n_modes=2, n_rows=1) == [{(1, 0): 3}]


@pytest.mark.parametrize(
    "body",
    [
        "not an object",
        payload([{"1,0,1,0": 1}], method="sample"),
        payload([], result={"kind": "counts", "n_qubits": 4, "data": {}}),
        payload([{"1,0,1": 1}], n_modes=3),
        payload({"1,0,1,0": 1}),
        payload([{"1,0,1,0": 1}, {}]),
        payload([["1,0,1,0", 1]]),
        payload([{"1,0,1": 1}]),
        payload([{"1,0,x,0": 1}]),
        payload([{"1,0,-1,0": 1}]),
        payload([{"1,0,1,0": 1.5}]),
        payload([{"1,0,1,0": True}]),
        payload([{"1,0,1,0": -1}]),
        # other malformed keys, counts and result objects
        payload([{"1,0,1,0,0": 1}]),
        payload([{"1,0,1,0,": 1}]),
        payload([{"1,,1,0": 1}]),
        payload([{"1, 0,1,0": 1}]),
        payload([{"+1,0,1,0": 1}]),
        payload([{"1,0,1,01": 1}]),
        payload([{"1,0,1,٣": 1}]),
        payload([{"": 1}]),
        payload([{"1,0,1,0": "5"}]),
        payload([{"1,0,1,0": None}]),
        payload([None]),
        payload(None),
        payload([{"1,0,1,0": 1}], result=None),
        payload([{"1,0,1,0": 1}], result={"kind": "fock_counts", "data": [{}]}),
        payload([{"1,0,1,0": 1}], n_modes=4.0),
    ],
)
def test_a_payload_that_breaks_the_contract_is_an_invalid_response(body):
    with pytest.raises(RouterError) as caught:
        decode_fock_counts(body, n_modes=4, n_rows=1, job_id="j1")
    assert caught.value.code == "invalid_response"
    assert caught.value.job_id == "j1"


def test_n_modes_that_differs_from_the_request_is_refused_even_with_matching_keys():
    with pytest.raises(RouterError) as caught:
        decode_fock_counts(payload([{"1,0": 1}], n_modes=2), n_modes=4, n_rows=1)
    assert caught.value.code == "invalid_response"


def test_a_bool_n_modes_in_the_payload_is_refused():
    with pytest.raises(RouterError) as caught:
        decode_fock_counts(payload([{"3": 7}], n_modes=True), n_modes=1, n_rows=1)
    assert caught.value.code == "invalid_response"


def test_one_mode_decodes():
    rows = decode_fock_counts(payload([{"3": 7}], n_modes=1), n_modes=1, n_rows=1)
    assert rows == [{(3,): 7}]


def test_several_rows_and_a_large_count_decode_in_order():
    data = [{"1,0": 10**12}, {}, {"0,1": 2, "1,0": 3}]
    rows = decode_fock_counts(payload(data, n_modes=2), n_modes=2, n_rows=3)
    assert rows == [{(1, 0): 10**12}, {}, {(0, 1): 2, (1, 0): 3}]


@pytest.mark.parametrize("field", ["method", "n_modes"])
def test_a_huge_payload_value_stays_out_of_the_error(field):
    huge = "x" * 100000
    body = payload([{"1,0,1,0": 1}])
    if field == "method":
        body["method"] = huge
    else:
        body["result"]["n_modes"] = huge
    with pytest.raises(RouterError) as caught:
        decode_fock_counts(body, n_modes=4, n_rows=1)
    assert caught.value.code == "invalid_response"
    assert len(str(caught.value)) < 300
    assert huge not in str(caught.value)


def test_importing_the_package_does_not_import_the_photonic_kit():
    code = (
        "import sys, eigenorbit_router_client; "
        "sys.exit('eigenkit.photonics' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
