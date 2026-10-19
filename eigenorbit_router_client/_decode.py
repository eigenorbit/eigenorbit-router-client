"""Decode a router `JobResultOut` into what the kit returns locally."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import RouterError


def _array_shape(method: str, n_qubits: int, n_observables: int) -> tuple[int, ...]:
    if method == "run":
        return (n_observables,)
    if method == "zero_state_probability":
        return ()
    if method == "probabilities":
        return (2**n_qubits,)
    raise ValueError(f"no array result for method {method!r}")


def decode_result(
    payload: Any,
    *,
    method: str,
    n_qubits: int,
    n_observables: int,
    job_id: str | None = None,
) -> np.ndarray | dict[str, int]:
    """Turn a JobResultOut into the kit's local return value.

    Returns:
        A float64 array, or a `{bitstring: count}` dict for `sample`.

    Raises:
        RouterError: `invalid_response` when the payload breaks the contract.
    """

    def invalid(reason: str) -> RouterError:
        return RouterError(
            "invalid_response", f"the router result {reason}", job_id=job_id
        )

    if not isinstance(payload, dict):
        raise invalid("is not a JSON object")
    if payload.get("bit_order") != "qubit0_msb":
        raise invalid("does not use bit_order qubit0_msb")
    if payload.get("method") != method:
        raise invalid(f"is for method {payload.get('method')!r}, expected {method!r}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise invalid("has no result object")
    if method == "sample":
        return _decode_counts(result, n_qubits, invalid)
    try:
        expected = _array_shape(method, n_qubits, n_observables)
    except ValueError:
        raise invalid(f"is for unknown method {method!r}") from None
    if result.get("kind") != "array" or result.get("dtype") != "float64":
        raise invalid("is not a float64 array")
    shape = result.get("shape")
    if not isinstance(shape, list) or tuple(shape) != expected:
        raise invalid(f"has shape {shape!r}, expected {list(expected)!r}")
    try:
        values = np.asarray(result.get("data"), dtype=np.float64)
    except (TypeError, ValueError):
        raise invalid("data is not numeric") from None
    if values.shape != expected:
        raise invalid(
            f"data has shape {list(values.shape)!r}, expected {list(expected)!r}"
        )
    return values.reshape(expected)


def _decode_counts(result: dict, n_qubits: int, invalid) -> dict[str, int]:
    if result.get("kind") != "counts":
        raise invalid("is not counts")
    if result.get("n_qubits") != n_qubits:
        raise invalid(f"has n_qubits {result.get('n_qubits')!r}, expected {n_qubits}")
    data = result.get("data")
    if not isinstance(data, dict):
        raise invalid("counts are not an object")
    counts: dict[str, int] = {}
    for key, value in data.items():
        if not (isinstance(key, str) and len(key) == n_qubits and set(key) <= {"0", "1"}):
            raise invalid(f"has a key that is not a {n_qubits}-bit string")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise invalid("has a count that is not a non-negative integer")
        counts[key] = value
    return counts
