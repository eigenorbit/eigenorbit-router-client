"""HTTP client for the router's `/v1/jobs*` API (contract v1)."""

from __future__ import annotations

import logging
import platform
import threading
import time
import warnings
from importlib import metadata
from typing import Any

import httpx

from .config import RouterConfig, resolve_api_key, resolve_base_url
from .errors import RouterError, TermsWarning


PROTOCOL_VERSION = 1
# The router's sentence about new terms, on a successful submit or estimate while the
# account is in the grace period of a new version of the router terms.
TERMS_NOTICE_HEADER = "EO-Terms-Notice"

_logger = logging.getLogger(__name__)
POLL_DELAYS = (1.0, 5.0, 30.0)
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "rejected"})
RETRY_DELAYS = (1.0, 5.0, 30.0)
_RETRY_STATUSES = frozenset({502, 503, 504})
# Safe to retry: a resent submit reuses its Idempotency-Key, so the router replays it.
_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)

_sleep = time.sleep
_monotonic = time.monotonic


def _make_http_client(*, base_url: str, timeout: float) -> httpx.Client:
    # Never follow redirects: the Authorization header must not reach another origin.
    return httpx.Client(base_url=base_url, timeout=timeout, follow_redirects=False)


def _dist_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def user_agent() -> str:
    """The User-Agent: client, kit and Python versions."""
    return (
        f"eigenorbit-router-client/{_dist_version('eigenorbit-router-client')} "
        f"eigenkit/{_dist_version('eigenkit')} "
        f"python/{platform.python_version()}"
    )


def _json(response: httpx.Response, *, job_id: str | None = None) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise RouterError(
            "invalid_response",
            f"the router answered HTTP {response.status_code} with a body that is "
            f"not a JSON object",
            http_status=response.status_code,
            job_id=job_id,
        )
    return body


class RouterHttpClient:
    """Blocking client for the router job API, one per compiled circuit."""

    def __init__(self, config: RouterConfig):
        self._config = config
        self._clients: dict[str, httpx.Client] = {}
        self._lock = threading.Lock()

    def _client_for(self, base_url: str) -> httpx.Client:
        with self._lock:
            client = self._clients.get(base_url)
            if client is None:
                client = _make_http_client(
                    base_url=base_url, timeout=self._config.request_timeout_seconds
                )
                self._clients[base_url] = client
            return client

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> httpx.Response:
        """Send one request; retries resend the same headers and body."""
        base_url = resolve_base_url(self._config)
        fixed_headers = {
            "EO-Protocol-Version": str(PROTOCOL_VERSION),
            "User-Agent": user_agent(),
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }
        client = self._client_for(base_url)
        sent_ibm_options = isinstance(json, dict) and "ibm_options" in json
        attempts = len(RETRY_DELAYS) + 1
        last_response: httpx.Response | None = None
        last_failure = ""
        for attempt in range(attempts):
            if attempt:
                _sleep(RETRY_DELAYS[attempt - 1])
            request_headers = {
                "Authorization": f"Bearer {resolve_api_key(self._config)}",
                **fixed_headers,
            }
            try:
                response = client.request(method, path, json=json, headers=request_headers)
            except _RETRY_EXCEPTIONS as exc:
                last_response, last_failure = None, type(exc).__name__
                continue
            except httpx.TransportError as exc:
                raise RouterError(
                    "router_unreachable",
                    f"could not reach the router at {base_url}: {type(exc).__name__}",
                    job_id=job_id,
                ) from None
            status = response.status_code
            if status in _RETRY_STATUSES:
                last_response = response
                continue
            if 300 <= status < 400:
                # Redirects are never followed: the key stays with the configured router.
                raise RouterError(
                    "invalid_response",
                    f"the router answered HTTP {status} (a redirect); redirects are "
                    f"not followed",
                    http_status=status,
                    job_id=job_id,
                )
            if not 200 <= status < 300:
                raise RouterError.from_response(
                    response, job_id=job_id, sent_ibm_options=sent_ibm_options
                )
            notice = response.headers.get(TERMS_NOTICE_HEADER)
            if notice:
                # The fixed stacklevel is deliberate: Python's default filter then
                # shows the sentence once per calling line, so a loop of submissions
                # is not flooded.
                try:
                    warnings.warn(notice, TermsWarning, stacklevel=2)
                except TermsWarning:
                    # Warnings are errors here (`-W error`). The request succeeded,
                    # and a submit holds funds already: never lose its answer.
                    _logger.warning("%s", notice)
            return response
        if last_response is not None:
            raise RouterError.from_response(
                last_response, job_id=job_id, sent_ibm_options=sent_ibm_options
            )
        idempotency_key = fixed_headers.get("Idempotency-Key")
        if idempotency_key is None:
            raise RouterError(
                "router_unreachable",
                f"could not reach the router at {base_url} after {attempts} attempts: "
                f"{last_failure}",
                job_id=job_id,
            )
        # Every attempt may have reached the router and lost only its answer:
        # the job can exist and hold funds, so name the key
        # that replays it instead of creating a second, separately billed job.
        raise RouterError(
            "router_unreachable",
            f"could not reach the router at {base_url} after {attempts} attempts "
            f"({last_failure}). The job may already exist and be billed: retry with "
            f"the same Idempotency-Key {idempotency_key} (compiled.submit(..., "
            f"idempotency_key=...)) before submitting again; the router then "
            f"replays that job instead of creating a second one.",
            {"idempotency_key": idempotency_key},
            job_id=job_id,
        )

    def estimate(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/jobs/estimate; returns the EstimateOut dict."""
        return _json(self._request("POST", "/v1/jobs/estimate", json=body))

    def submit(self, body: dict[str, Any], *, idempotency_key: str) -> dict[str, Any]:
        """POST /v1/jobs with an Idempotency-Key; returns the JobOut dict."""
        response = self._request(
            "POST", "/v1/jobs", json=body, headers={"Idempotency-Key": idempotency_key}
        )
        return _json(response)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1/jobs/{job_id}; returns the JobOut dict."""
        return _json(
            self._request("GET", f"/v1/jobs/{job_id}", job_id=job_id), job_id=job_id
        )

    def get_result(self, job_id: str) -> dict[str, Any]:
        """GET /v1/jobs/{job_id}/result; returns the JobResultOut dict."""
        response = self._request("GET", f"/v1/jobs/{job_id}/result", job_id=job_id)
        return _json(response, job_id=job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """POST /v1/jobs/{job_id}/cancel; returns the JobOut dict."""
        response = self._request("POST", f"/v1/jobs/{job_id}/cancel", job_id=job_id)
        return _json(response, job_id=job_id)

    def wait(
        self, job_id: str, *, timeout_seconds: float, cancels_job: bool = False
    ) -> dict[str, Any]:
        """Poll the job until its status is terminal and return its JobOut.

        `cancels_job` says whether the caller cancels the job when this raises; the
        messages then tell the user so instead of saying that the job keeps running.

        Raises:
            RouterError: `poll_timeout` when the job is not terminal at the deadline.
            KeyboardInterrupt: On Ctrl-C, with `.job_id` set.
        """
        # This method never cancels the job. The caller decides: the qubit backend
        # leaves it running (and billed), the photonic backend cancels it, and the
        # messages say which.
        deadline = _monotonic() + timeout_seconds
        polls = 0
        try:
            while True:
                job = self.get_job(job_id)
                status = job.get("status")
                if status in TERMINAL_STATUSES:
                    return job
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    after = (
                        "The job is being cancelled; raise "
                        "RouterConfig(poll_timeout_seconds=...) to wait longer."
                        if cancels_job
                        else "It keeps running and is billed; there is no auto-cancel. "
                        "Call result() again on a submit() handle to keep waiting, or "
                        "handle.cancel() to stop it."
                    )
                    raise RouterError(
                        "poll_timeout",
                        f"router job {job_id} is still {status} after "
                        f"poll_timeout_seconds={timeout_seconds:g}. {after}",
                        {"status": status, "poll_timeout_seconds": timeout_seconds},
                        job_id=job_id,
                    )
                _sleep(min(POLL_DELAYS[min(polls, len(POLL_DELAYS) - 1)], remaining))
                polls += 1
        except KeyboardInterrupt:
            after = (
                "The job is being cancelled."
                if cancels_job
                else "The job keeps running and is billed (no auto-cancel). Use "
                "compiled.submit(...) and handle.cancel() to control long jobs."
            )
            interrupt = KeyboardInterrupt(
                f"interrupted while waiting for router job {job_id}. {after}"
            )
            interrupt.job_id = job_id
            raise interrupt from None

    def close(self) -> None:
        """Close every httpx client this instance opened."""
        with self._lock:
            clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            client.close()
