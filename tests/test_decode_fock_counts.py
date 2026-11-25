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
        # beyond the brief: other malformed keys, counts and result objects
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
