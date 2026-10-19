"""`RouterConfig`: run eigenkit circuits through EigenOrbit Router."""

from __future__ import annotations

import os
import re
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from eigenkit.core.secret_fields import SecretFieldsMixin

from .errors import RouterError


if TYPE_CHECKING:
    from .backend import RouterBackend


DEFAULT_BASE_URL = "https://router.eigenorbit.com"
API_KEY_ENV = "EIGENORBIT_ROUTER_API_KEY"
BASE_URL_ENV = "EIGENORBIT_ROUTER_URL"

# Every other device is treated as a QPU (no gradients).
_SIMULATORS = frozenset({"braket/sv1", "braket/dm1"})

_DEVICE_PATTERN = r"[a-z0-9]+/[a-z0-9_-]+"
_DEVICE_MAX_LENGTH = 64

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class RouterConfig(SecretFieldsMixin):
    """Router device and connection settings; pass as `provider=`.

    Attributes:
        device: Router device ID `<vendor>/<device>`, e.g. `"braket/sv1"`.
        shots: 0 asks a simulator for exact results; a positive int samples.
        api_key: EigenOrbit API key; None reads `EIGENORBIT_ROUTER_API_KEY`.
            Never saved, pickled or shown in `repr`.
        base_url: Router URL; None reads `EIGENORBIT_ROUTER_URL`, then the default.
        poll_timeout_seconds: How long a blocking call waits for its job.
        max_concurrent_submits: How many jobs a batch submits at once.
        request_timeout_seconds: Timeout of each HTTP request.
    """

    device: str
    _: KW_ONLY
    shots: int = 0
    api_key: str | None = field(default=None, repr=False, metadata={"secret": True})
    base_url: str | None = None
    poll_timeout_seconds: int = 3600
    max_concurrent_submits: int = 4
    request_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _check_device(self.device)
        if not _is_int(self.shots) or self.shots < 0:
            raise ValueError(f"shots must be an int >= 0, got {self.shots!r}")
        if not _is_number(self.poll_timeout_seconds) or self.poll_timeout_seconds <= 0:
            raise ValueError(
                f"poll_timeout_seconds must be a number > 0, "
                f"got {self.poll_timeout_seconds!r}"
            )
        if not _is_int(self.max_concurrent_submits) or self.max_concurrent_submits < 1:
            raise ValueError(
                f"max_concurrent_submits must be an int >= 1, "
                f"got {self.max_concurrent_submits!r}"
            )
        if (
            not _is_number(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError(
                f"request_timeout_seconds must be a number > 0, "
                f"got {self.request_timeout_seconds!r}"
            )
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ValueError("api_key must be a str or None")
        if self.base_url is not None:
            _check_base_url(self.base_url, source="base_url")

    def make_backend(self) -> RouterBackend:
        """Build the backend; called by the kit's provider layer."""
        from .backend import RouterBackend

        return RouterBackend(config=self)

    @property
    def vendor(self) -> str:
        """The vendor part of `device`, e.g. `"braket"`."""
        return self.device.split("/", 1)[0]

    @property
    def is_qpu(self) -> bool:
        """False for the Braket simulators SV1 and DM1, True otherwise."""
        return self.device not in _SIMULATORS


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_device(device: object) -> None:
    if (
        not isinstance(device, str)
        or len(device) > _DEVICE_MAX_LENGTH
        or re.fullmatch(_DEVICE_PATTERN, device) is None
    ):
        raise ValueError(
            f"device must be '<vendor>/<device>' in ASCII lower case (letters, "
            f"digits, '_' and '-' after the '/'), at most {_DEVICE_MAX_LENGTH} "
            f"characters, for example 'braket/sv1'; got {device!r}"
        )


def _check_base_url(url: object, *, source: str) -> str:
    """Return `url` without a trailing `/`; https only, except http to loopback."""
    if not isinstance(url, str):
        raise ValueError(f"{source} must be a str, got {type(url).__name__}")
    url = url.strip().rstrip("/")
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        host = None
    if not host:
        raise ValueError(
            f"{source} must be an absolute https:// URL such as "
            f"'https://router.eigenorbit.com', got {url!r}"
        )
    if parts.scheme == "https":
        return url
    if parts.scheme == "http" and host in _LOOPBACK_HOSTS:
        return url
    raise ValueError(
        f"{source} must use https:// (plain http:// is allowed only for "
        f"localhost, 127.0.0.1 and ::1, so the API key never travels in clear "
        f"text), got {url!r}"
    )


def resolve_api_key(config: RouterConfig) -> str:
    """Return the API key: `config.api_key`, else `EIGENORBIT_ROUTER_API_KEY`.

    Raises:
        RouterError: `unauthenticated` when no key is set.
    """
    key = config.api_key if config.api_key is not None else os.environ.get(API_KEY_ENV)
    key = (key or "").strip()
    if not key:
        raise RouterError(
            "unauthenticated",
            "No EigenOrbit API key: set EIGENORBIT_ROUTER_API_KEY or pass "
            "RouterConfig(api_key=...)",
            http_status=None,
        )
    return key


def resolve_base_url(config: RouterConfig) -> str:
    """Return the base URL: `config.base_url`, the env variable, then the default.

    Raises:
        ValueError: The URL is not https:// (http:// is allowed only for loopback).
    """
    if config.base_url is not None:
        return _check_base_url(config.base_url, source="base_url")
    env_url = (os.environ.get(BASE_URL_ENV) or "").strip()
    if env_url:
        return _check_base_url(env_url, source=BASE_URL_ENV)
    return DEFAULT_BASE_URL
