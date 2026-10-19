"""The one exception every router failure raises."""

from __future__ import annotations

from typing import Any

import httpx


# Codes the client raises itself, never received from the router.
CLIENT_ERROR_CODES = frozenset({"invalid_response", "poll_timeout", "router_unreachable"})

SUPPORT_EMAIL = "support@eigenorbit.com"
# How long the router keeps a job's result and request (contract section 8).
RESULT_RETENTION_DAYS = 90


class RouterError(RuntimeError):
    """A router call failed, or a router job ended without a result.

    Attributes:
        code: Stable snake_case code; branch on this, never on the message.
        message: Human-readable text that may change.
        details: Extra structured data for the code, possibly empty.
        http_status: The HTTP status, or None when no response carried the error.
        job_id: The router job ID, when the failure belongs to a job.
        category: Who or what stopped a failed or rejected job (`rejected`,
            `circuit`, `vendor` or `internal`; contract section 9.2), or None
            for HTTP errors, client-side errors and routers that predate it.
            Branch on `category` for display and on `code` for logic.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        *,
        http_status: int | None = None,
        job_id: str | None = None,
        category: str | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details) if details else {}
        self.http_status = http_status
        self.job_id = job_id
        self.category = category

    def __str__(self) -> str:
        where = f"code={self.code}"
        if self.job_id:
            where += f", job_id={self.job_id}"
        return f"{self.message} ({where})"

    def __reduce__(self):
        return (
            _rebuild,
            (
                self.code,
                self.message,
                self.details,
                self.http_status,
                self.job_id,
                self.category,
            ),
        )

    @classmethod
    def from_response(
        cls, response: httpx.Response, *, job_id: str | None = None
    ) -> RouterError:
        """Build the error for a non-2xx router response."""
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if not isinstance(code, str) or not code:
            return cls(
                "invalid_response",
                f"the router answered HTTP {status} without a contract error body",
                http_status=status,
                job_id=job_id,
            )
        message = error.get("message")
        if not isinstance(message, str) or not message:
            message = f"the router answered HTTP {status}"
        details = error.get("details")
        if not isinstance(details, dict):
            details = {}
        extra = refusal_hint(code, details)
        if extra:
            message = _append(message, extra)
        return cls(code, message, details, http_status=status, job_id=job_id)

    @classmethod
    def from_job(cls, job: dict[str, Any]) -> RouterError:
        """Build the error for a terminal job that has no result."""
        job_id = job.get("id") if isinstance(job.get("id"), str) else None
        status = job.get("status")
        error = job.get("error") if isinstance(job.get("error"), dict) else {}
        code = error.get("code")
        if not isinstance(code, str) or not code:
            code = "job_not_completed"
        message = error.get("message")
        if not isinstance(message, str) or not message:
            message = f"router job {job_id} ended {status}"
        category = error.get("category")
        if not isinstance(category, str) or not category:
            category = None  # None when the router does not send it
        hint = _JOB_HINTS.get(code)
        if hint is not None:
            message = _append(message, hint)
        return cls(
            code,
            message,
            {"status": status, "billing": job.get("billing")},
            http_status=None,
            job_id=job_id,
            category=category,
        )


def _shots_hint(details: dict[str, Any]) -> str:
    """The fix hint for a shots_invalid rejection."""
    low, high = details.get("min_shots"), details.get("max_shots")
    if all(isinstance(v, int) and not isinstance(v, bool) for v in (low, high)):
        return f"Set RouterConfig(shots=...) to a value within {low}..{high}."
    return "Set RouterConfig(shots=...) to a value the device and method accept."


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _device_unavailable_hint(details: dict[str, Any]) -> str:
    """The hint for a device_unavailable rejection (422, details {device, status})."""
    device = details.get("device")
    if not isinstance(device, str) or not device:
        return ""
    if details.get("stale") is True:
        return (
            f"The router has not confirmed this device's status recently ({device}); "
            f"retry in a few minutes."
        )
    status = details.get("status")
    if not isinstance(status, str) or not status:
        return ""
    return (
        f"{device} is {status} at the vendor: pick another device (GET /v1/devices "
        f"lists them) or retry later."
    )


def _concurrency_hint(details: dict[str, Any]) -> str:
    """The hint for a concurrency_limit_exceeded rejection (429, {limit, in_flight})."""
    limit, in_flight = details.get("limit"), details.get("in_flight")
    if not (_is_int(limit) and _is_int(in_flight)):
        return ""
    return (
        f"You have {in_flight} jobs in flight and the limit is {limit}: wait for a "
        f"job to finish or cancel one, then submit again."
    )


def _not_cancellable_hint(details: dict[str, Any]) -> str:
    """The hint for a job_not_cancellable refusal (409, details {status})."""
    status = details.get("status")
    if not isinstance(status, str) or not status:
        return ""
    return f"The job is already {status}, so there is nothing to cancel."


def _region_blocked_hint(details: dict[str, Any]) -> str:
    """The note for a region_blocked refusal (451, details {}); never retried."""
    return "(sanctions and export rules)."


def _access_restricted_hint(details: dict[str, Any]) -> str:
    """The hint for an access_restricted refusal (403, details {}); never retried."""
    return (
        f"Your account is not approved for the router yet: contact {SUPPORT_EMAIL} "
        f"to request access. Signing in again will not help."
    )


def _user_blocked_hint(details: dict[str, Any]) -> str:
    """The hint for a user_blocked refusal (403); never retried."""
    return (
        f"Retrying or signing in again will not help: contact {SUPPORT_EMAIL} about "
        f"the block."
    )


def _result_expired_hint(details: dict[str, Any]) -> str:
    """The hint for a result_expired refusal (410, details {expired_at}); never retried."""
    retention = (
        f"Results are kept for {RESULT_RETENTION_DAYS} days after a job finishes; the "
        f"job's status and costs stay readable."
    )
    expired_at = details.get("expired_at")
    if isinstance(expired_at, str) and expired_at:
        return f"The result was deleted at {expired_at}. {retention}"
    return retention


# Where a user connects or reconnects an IBM Quantum account (the router web app's
# "IBM account" section, route /ibm).
IBM_ACCOUNT_PAGE = "the IBM account page (/ibm) of the router web app"


def _ibm_account_not_connected_hint(details: dict[str, Any]) -> str:
    """The hint for ibm_account_not_connected (403, details {device}); never retried."""
    device = details.get("device")
    target = f"run on {device}" if isinstance(device, str) and device else "run IBM jobs"
    return (
        f"To {target}, connect your IBM Quantum account on {IBM_ACCOUNT_PAGE}. "
        f"Retrying will not help until then."
    )


def _ibm_connection_invalid_hint(details: dict[str, Any]) -> str:
    """The hint for ibm_connection_invalid (403, details {}); never retried."""
    return (
        f"Reconnect your IBM Quantum account on {IBM_ACCOUNT_PAGE}: IBM jobs are "
        f"refused until you reconnect, and retrying will not help."
    )


# Per-code hints appended to the router's message; unknown codes keep the
# router's message verbatim (contract section 9.3).
_HINTS = {
    "shots_invalid": _shots_hint,
    "device_unavailable": _device_unavailable_hint,
    "concurrency_limit_exceeded": _concurrency_hint,
    "job_not_cancellable": _not_cancellable_hint,
    "region_blocked": _region_blocked_hint,
    "access_restricted": _access_restricted_hint,
    "user_blocked": _user_blocked_hint,
    "result_expired": _result_expired_hint,
    "ibm_account_not_connected": _ibm_account_not_connected_hint,
    "ibm_connection_invalid": _ibm_connection_invalid_hint,
}

# Hints appended to the message of a failed IBM job (contract section 9.2,
# category vendor). The router never retries these jobs, and neither does the client.
_JOB_HINTS = {
    "ibm_credentials_rejected": (
        f"IBM refused your IBM Quantum token or instance. Reconnect your account on "
        f"{IBM_ACCOUNT_PAGE}; until then IBM jobs are refused with "
        f"ibm_connection_invalid."
    ),
    "ibm_quota_exhausted": (
        "The IBM instance this job ran on has used up its quantum time: submitting "
        "again will not help until IBM renews it."
    ),
    "ibm_target_mismatch": (
        "The router never resubmits it: submit the job again, and it is compiled "
        "for the device's current target."
    ),
    "ibm_device_not_on_instance": (
        f"Your IBM instance cannot use this device: pick one it lists on "
        f"{IBM_ACCOUNT_PAGE}."
    ),
}


def refusal_hint(code: str, details: dict[str, Any]) -> str:
    """The fix hint the client adds to the router's message for `code`, or ""."""
    hint = _HINTS.get(code)
    return hint(details) if hint is not None else ""


def _append(message: str, extra: str) -> str:
    """`message` continued by `extra`: a parenthetical note continues the router's
    sentence, anything else starts a new one."""
    message = message.rstrip()
    if extra.startswith("("):
        return f"{message} {extra}"
    if message.endswith((".", "!", "?")):
        return f"{message} {extra}"
    return f"{message}. {extra}"


def _rebuild(code, message, details, http_status, job_id, category=None) -> RouterError:
    return RouterError(
        code,
        message,
        details,
        http_status=http_status,
        job_id=job_id,
        category=category,
    )
