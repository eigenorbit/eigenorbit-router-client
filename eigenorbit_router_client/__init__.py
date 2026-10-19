"""eigenkit provider for EigenOrbit Router (`RouterConfig`)."""

from importlib import metadata as _metadata

from ._client import PROTOCOL_VERSION
from .backend import RouterBackend, RouterCompiled
from .config import DEFAULT_BASE_URL, RouterConfig
from .errors import CLIENT_ERROR_CODES, RouterError
from .handle import RouterJobHandle


try:
    __version__ = _metadata.version("eigenorbit-router-client")
except _metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "CLIENT_ERROR_CODES",
    "DEFAULT_BASE_URL",
    "PROTOCOL_VERSION",
    "RouterBackend",
    "RouterCompiled",
    "RouterConfig",
    "RouterError",
    "RouterJobHandle",
    "__version__",
]


from eigenkit.core.provider import register_lazy as _register_lazy


def _router_backend() -> RouterBackend:
    """`provider="router"`: raises the ValueError that names RouterConfig(device=...)."""
    return RouterBackend()


# Last, so `RouterBackend` is bound and eigenkit's own providers are registered.
_register_lazy("router", _router_backend)
