"""The photonic executor: runs one `eigenkit.photonics` circuit as one router job."""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from ._client import RouterHttpClient
from ._decode import decode_fock_counts
from .config import RouterConfig
from .errors import RouterError


if TYPE_CHECKING:
    from eigenkit.photonics.ir import PhotonicCircuitSpec


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
        return array

    def _run_job(
        self, weights: np.ndarray, inputs: np.ndarray, *, idempotency_key: str
    ) -> list[dict[tuple[int, ...], int]]:
        """Submit one job, wait, fetch and decode: one count dict per input row.

        An exception raised while the job is in flight (an interrupt, the poll
        timeout, a failed fetch) cancels the job and is re-raised unchanged.
        """
        rows = 1 if np.ndim(inputs) == 1 else len(inputs)
        job = self._http.submit(
            self.build_request(weights, inputs), idempotency_key=idempotency_key
        )
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RouterError("invalid_response", "the router returned a job without an id")
        try:
            job = self._http.wait(job_id, timeout_seconds=self._config.poll_timeout_seconds)
            if job.get("status") == "completed":
                return decode_fock_counts(
                    self._http.get_result(job_id),
                    n_modes=self._spec.n_modes,
                    n_rows=rows,
                    job_id=job_id,
                )
        except BaseException:
            with contextlib.suppress(Exception):
                self._http.cancel(job_id)
            raise
        raise RouterError.from_job(job)

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
        counts = self._run_job(weights, inputs, idempotency_key=str(uuid.uuid4()))[0]
        return _normalized(counts)

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
