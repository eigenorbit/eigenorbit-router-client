"""The photonic executor: runs one `eigenkit.photonics` circuit as one router job."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from ._client import RouterHttpClient
from ._decode import decode_fock_counts
from .config import RouterConfig
from .errors import RouterError


if TYPE_CHECKING:
    from eigenkit.photonics.ir import PhotonicCircuitSpec

_logger = logging.getLogger(__name__)
_EMPTY_ROWS_LISTED = 10


def _check_config(config: RouterConfig) -> None:
    if not config.is_photonic:
        raise ValueError(
            f"RouterPhotonicBackend runs on the router's photonic devices "
            f"(quandela/<device>, for example 'quandela/sim_slos'); got "
            f"{config.device!r}. Pass this RouterConfig as provider= to run qubit "
            f"circuits on it."
        )
    if config.shots < 1:
        raise ValueError(
            f"a photonic job is sampled: set RouterConfig(shots=...) to 1 or more, "
            f"got {config.shots}"
        )


def _check_finite(array: np.ndarray, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite: no NaN and no infinity")


class RouterPhotonicBackend:
    """Photonic executor for the router's Quandela devices (limited preview).

    Raises:
        ValueError: `config` is not a photonic device or its `shots` is below 1.
    """

    def __init__(self, config: RouterConfig):
        _check_config(config)
        self.config = config

    def compile(self, spec: PhotonicCircuitSpec) -> RouterPhotonicCompiled:
        """Prepare `spec` for repeated runs; sends nothing."""
        return RouterPhotonicCompiled(spec, self.config)


class RouterPhotonicCompiled:
    """A photonic circuit compiled for the router; every `run` is one router job.

    A run that is interrupted or times out cancels its job.

    Raises:
        ValueError: The config is not photonic or its `shots` is below 1.
        ImportError: The installed eigenkit is older than 1.4.0.
    """

    def __init__(self, spec: PhotonicCircuitSpec, config: RouterConfig):
        _check_config(config)
        try:
            from eigenkit.photonics.wire import photonic_spec_to_json
        except ImportError as exc:
            raise ImportError(
                "RouterPhotonicBackend needs eigenkit 1.4.0 or later "
                "(eigenkit.photonics.wire): upgrade eigenkit."
            ) from exc
        self._spec = spec
        self._config = config
        self._circuit_json = photonic_spec_to_json(spec)
        self._http = RouterHttpClient(config)

    def build_request(self, weights: Any, inputs: Any) -> dict[str, Any]:
        """The `POST /v1/jobs` body for one run (one row per row of 2-D `inputs`)."""
        return {
            "device": self._config.device,
            "method": "photonic_sample",
            "shots": self._config.shots,
            "circuit_spec": self._circuit_json,
            "observables": [],
            "weights": np.asarray(weights, dtype=np.float64).tolist(),
            "inputs": np.asarray(inputs, dtype=np.float64).tolist(),
        }

    def _checked_weights(self, weights: Any) -> np.ndarray:
        array = np.asarray(weights, dtype=np.float64)
        if array.shape != (self._spec.n_weights,):
            raise ValueError(
                f"weights must have shape ({self._spec.n_weights},), got {array.shape}"
            )
        _check_finite(array, "weights")
        return array

    def _run_job(
        self, weights: np.ndarray, inputs: np.ndarray, *, idempotency_key: str
    ) -> list[dict[tuple[int, ...], int]]:
        """Submit one job, wait, fetch and decode: one count dict per input row.

        An exception raised while the job is in flight (an interrupt, the poll
        timeout, a failed fetch) cancels the job and is re-raised unchanged. So does
        an interrupt during the submit itself, which may have left a job behind.
        """
        rows = 1 if np.ndim(inputs) == 1 else len(inputs)
        body = self.build_request(weights, inputs)
        try:
            job = self._http.submit(body, idempotency_key=idempotency_key)
        except RouterError:
            raise
        except BaseException:
            self._cancel_interrupted_submit(body, idempotency_key)
            raise
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RouterError("invalid_response", "the router returned a job without an id")
        try:
            job = self._http.wait(
                job_id,
                timeout_seconds=self._config.poll_timeout_seconds,
                cancels_job=True,
            )
            if job.get("status") == "completed":
                return decode_fock_counts(
                    self._http.get_result(job_id),
                    n_modes=self._spec.n_modes,
                    n_rows=rows,
                    job_id=job_id,
                )
        except BaseException:
            self._cancel(job_id)
            raise
        raise RouterError.from_job(job)

    def _cancel_interrupted_submit(self, body: dict[str, Any], idempotency_key: str) -> None:
        """Cancel the job an interrupted submit may have created.

        The same request goes out once more with the same Idempotency-Key: the router
        answers with the job it already created, or creates it now, and the job is
        cancelled either way. A failure is logged, not raised: by type alone when the
        replay gave no job, with the job id when only the cancel failed.
        """
        try:
            job = self._http.submit(body, idempotency_key=idempotency_key)
            job_id = job.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise RouterError("invalid_response", "the router returned a job without an id")
        except Exception as exc:
            _logger.warning(
                "could not cancel the router job of an interrupted submit (%s); it "
                "may still be running and hold the device",
                type(exc).__name__,
            )
            return
        self._cancel(job_id)

    def _cancel(self, job_id: str) -> None:
        """Ask the router to cancel `job_id`; a failed request is logged, not raised."""
        try:
            self._http.cancel(job_id)
        except Exception as exc:
            if isinstance(exc, RouterError) and exc.code == "job_not_cancellable":
                return  # the job already ended: nothing is left to cancel
            _logger.warning(
                "could not cancel router job %s (%s); it may still be running and "
                "hold the device",
                job_id,
                type(exc).__name__,
            )

    def run(self, weights: Any, inputs: Any) -> dict[tuple[int, ...], float]:
        """Sample the circuit once; the Fock distribution `{state: probability}`.

        Raises:
            ValueError: `weights` or `inputs` has the wrong shape.
            RouterError: The router refused or failed the job.
            RuntimeError: No shot survived post-selection.
        """
        weights = self._checked_weights(weights)
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.shape != (self._spec.n_inputs,):
            raise ValueError(
                f"inputs must have shape ({self._spec.n_inputs},), got {inputs.shape}"
            )
        _check_finite(inputs, "inputs")
        counts = self._run_job(weights, inputs, idempotency_key=str(uuid.uuid4()))[0]
        return _normalized(counts)

    def run_batch(self, weights: Any, rows: Any) -> list[dict[tuple[int, ...], float]]:
        """Sample the circuit for every row of 2-D `rows`: one distribution per row.

        The batch is one router job. When the router refuses it for having more rows
        than the device takes (`batch_not_supported` with a usable `max_rows`), the
        rows go in chunks of that limit, one job after another; an interrupt cancels
        only the chunk in flight. The whole batch must still fit one router request:
        a larger one is refused with `RouterError` code `request_too_large` after
        one request, and the caller splits it.

        Raises:
            ValueError: `weights` or `rows` has the wrong shape.
            RouterError: The router refused or failed a job (a refusal without a
                usable row limit, and a refused chunk, are raised as they came).
            RuntimeError: No shot survived post-selection in some rows.
        """
        weights = self._checked_weights(weights)
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self._spec.n_inputs:
            raise ValueError(
                f"rows must have shape (n, {self._spec.n_inputs}), got {rows.shape}; "
                f"run() takes one sample"
            )
        _check_finite(rows, "rows")
        n = len(rows)
        if n == 0:
            return []
        call_id = str(uuid.uuid4())
        max_rows = None
        try:
            counts = self._run_job(weights, rows, idempotency_key=call_id)
        except RouterError as exc:
            limit = exc.details.get("max_rows")
            if (
                exc.code != "batch_not_supported"
                or isinstance(limit, bool)
                or not isinstance(limit, int)
                or not 1 <= limit < n
            ):
                raise
            max_rows = limit
        if max_rows is not None:
            counts = []
            for chunk, start in enumerate(range(0, n, max_rows)):
                counts += self._run_job(
                    weights,
                    rows[start : start + max_rows],
                    idempotency_key=f"{call_id}:{chunk}",
                )
        empty = [i for i, row in enumerate(counts) if sum(row.values()) == 0]
        if empty:
            listed = empty[:_EMPTY_ROWS_LISTED]
            more = (
                f" (the first {len(listed)} of {len(empty)} rows)"
                if len(empty) > len(listed)
                else ""
            )
            raise RuntimeError(
                f"no shots survived post-selection on the detected photon count in "
                f"rows {listed} of {n}{more}; raise RouterConfig(shots=...)"
            )
        return [_normalized(row) for row in counts]

    def expectation(self, weights: Any, inputs: Any, observable: Any) -> float:
        """The expectation of a Fock-diagonal observable from one sampled run."""
        from eigenkit.photonics.gradients import _check_observable, _expectation

        _check_observable(self._spec, observable)
        return _expectation(observable, self.run(weights, inputs))

    def jacobian(
        self,
        weights: Any,
        inputs: Any,
        observable: Any,
        *,
        method: str = "parameter_shift",
        need_weight_grad: bool = True,
        need_input_grad: bool = True,
    ):
        """Always raises: gradients are unavailable on Quandela devices."""
        raise NotImplementedError(
            "gradients are unavailable on Quandela devices (finite-shot sampling); "
            "use eigenkit's local PercevalBackend for gradients"
        )

    def close(self) -> None:
        """Close the HTTP connections this handle opened."""
        self._http.close()


def _normalized(counts: dict[tuple[int, ...], int]) -> dict[tuple[int, ...], float]:
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError(
            "no shots survived post-selection on the detected photon count; "
            "raise RouterConfig(shots=...)"
        )
    return {state: count / total for state, count in counts.items()}
