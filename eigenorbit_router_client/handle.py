"""`RouterJobHandle`: one submitted router job, as the kit's `JobHandle`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._decode import decode_result
from .errors import RouterError


if TYPE_CHECKING:
    from ._client import RouterHttpClient


# JobOut.credential_source (contract section 5.3).
_CREDENTIAL_SOURCES = frozenset({"own", "organization"})


class RouterJobHandle:
    """A router job returned by `RouterCompiled.submit()`; not built directly."""

    def __init__(
        self,
        http: RouterHttpClient,
        job: dict[str, Any],
        *,
        method: str,
        n_qubits: int,
        n_observables: int,
        poll_timeout_seconds: int,
    ):
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RouterError("invalid_response", "the router returned a job without an id")
        self._http = http
        self._job = dict(job)
        self._job_id = job_id
        self._method = method
        self._n_qubits = n_qubits
        self._n_observables = n_observables
        self._poll_timeout_seconds = poll_timeout_seconds
        self._result: np.ndarray | dict[str, int] | None = None

    @property
    def method(self) -> str:
        """The contract method this job runs, for example `"run"`."""
        return self._method

    def job_id(self) -> str:
        """The router job ID (a UUID)."""
        return self._job_id

    def _refresh(self) -> dict[str, Any]:
        self._job = self._http.get_job(self._job_id)
        return self._job

    def status(self) -> str:
        """The job's current contract status, from one non-blocking GET."""
        return self._refresh().get("status")

    def billing(self) -> str | None:
        """The job's billing state (`held`, `settled`, `released`), or None.

        None for an IBM job, which the router never charges.
        """
        return self._refresh().get("billing")

    @property
    def credential_source(self) -> str | None:
        """Whose IBM account ran the job, from the last job the router returned.

        `"own"` is your own connected IBM Quantum account (IBM bills your
        instance), `"organization"` an IBM account EigenOrbit provides. None
        for a Braket job and when the router does not send it. Sends nothing;
        `status()` or `result()` refreshes it.
        """
        source = self._job.get("credential_source")
        if isinstance(source, str) and source in _CREDENTIAL_SOURCES:
            return source
        return None

    @property
    def ibm_usage_seconds(self) -> float | None:
        """The IBM quantum time the job used, in seconds, from the last job the
        router returned.

        None until IBM reports it (it may never), for a Braket job, and when
        the router does not send it. Sends nothing; `status()` or `result()`
        refreshes it.
        """
        usage = self._job.get("ibm_usage_seconds")
        if isinstance(usage, bool) or not isinstance(usage, (int, float)):
            return None
        return float(usage)

    def result(self) -> np.ndarray | dict[str, int]:
        """Wait up to `poll_timeout_seconds` for the job and return its result.

        Raises:
            RouterError: The job did not complete, or the wait timed out
                (`poll_timeout`; call `result()` again to keep waiting).
        """
        if self._result is not None:
            return self._result
        job = self._http.wait(self._job_id, timeout_seconds=self._poll_timeout_seconds)
        self._job = job
        if job.get("status") != "completed":
            raise RouterError.from_job(job)
        payload = self._http.get_result(self._job_id)
        value = decode_result(
            payload,
            method=self._method,
            n_qubits=self._n_qubits,
            n_observables=self._n_observables,
            job_id=self._job_id,
        )
        if self._method == "zero_state_probability":
            value = np.asarray(value, dtype=np.float64).reshape(())
        self._result = value
        return value

    def cancel(self) -> str | None:
        """Ask the router to cancel the job, and return its status right after.

        A 202 means the cancel was requested, not that the job is cancelled: a
        job that never reached the vendor comes back `cancelled` at once, and a
        job at the vendor keeps its status and ends as the vendor reports it
        (`cancelled`, or `completed` when the vendor finishes the task first,
        with its result). Keep polling `status()` or call `result()` to see
        how it ended. Nothing is resubmitted.

        Raises:
            RouterError: `job_not_cancellable` (409) when the job is already
                completed, failed or rejected; `details["status"]` says which.
        """
        try:
            job = self._http.cancel(self._job_id)
        except RouterError as error:
            status = error.details.get("status")
            if error.code == "job_not_cancellable" and isinstance(status, str):
                self._job = {**self._job, "status": status}
            raise
        self._job = dict(job)
        return self._job.get("status")

    def __repr__(self) -> str:
        return f"RouterJobHandle(job_id={self._job_id!r}, method={self._method!r})"
