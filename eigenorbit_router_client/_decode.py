"""Decode a router `JobResultOut` into what the kit returns locally."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .errors import RouterError


_PHOTON_COUNT = re.compile(r"0|[1-9][0-9]*")


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


def decode_execution(payload: Any) -> dict[str, Any] | None:
    """The `execution` object of a JobResultOut (contract section 8), or None.

    None when the router sends null (a Braket job, or a result stored before
    the router recorded it), leaves the field out (an older router) or sends
    something that is not an object. Its keys, including ones a newer router
    adds, are returned as the router sent them.
    """
    execution = payload.get("execution") if isinstance(payload, dict) else None
    return execution if isinstance(execution, dict) else None


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


def decode_fock_counts(
    payload: Any,
    *,
    n_modes: int,
    n_rows: int,
    job_id: str | None = None,
) -> list[dict[tuple[int, ...], int]]:
    """Turn a `photonic_sample` JobResultOut into one count dict per input row.

    Each dict maps a Fock state, the photon count of every mode as a tuple of
    ints, to its shot count. A row in which no shot passed the photon filter is
    an empty dict. `bit_order` is not read: it applies to bitstring results.

    Raises:
        RouterError: `invalid_response` when the payload breaks the contract.
    """

    def invalid(reason: str) -> RouterError:
        return RouterError(
            "invalid_response", f"the router result {reason}", job_id=job_id
        )

    if not isinstance(payload, dict):
        raise invalid("is not a JSON object")
    if payload.get("method") != "photonic_sample":
        raise invalid("is for a method other than 'photonic_sample'")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise invalid("has no result object")
    if result.get("kind") != "fock_counts":
        raise invalid("is not fock_counts")
    result_modes = result.get("n_modes")
    if type(result_modes) is not int or result_modes != n_modes:
        raise invalid(f"has an n_modes that is not the requested {n_modes}")
    data = result.get("data")
    if not isinstance(data, list) or len(data) != n_rows:
        raise invalid(f"does not hold a list of {n_rows} rows")
    rows: list[dict[tuple[int, ...], int]] = []
    for row in data:
        if not isinstance(row, dict):
            raise invalid("has a row that is not an object")
        counts: dict[tuple[int, ...], int] = {}
        for key, value in row.items():
            parts = key.split(",") if isinstance(key, str) else []
            if len(parts) != n_modes or not all(
                _PHOTON_COUNT.fullmatch(p) for p in parts
            ):
                raise invalid(
                    f"has a key that is not {n_modes} comma-joined photon counts"
                )
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise invalid("has a count that is not a non-negative integer")
            counts[tuple(int(p) for p in parts)] = value
        rows.append(counts)
    return rows
