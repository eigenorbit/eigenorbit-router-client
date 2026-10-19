import socket

import httpx
import pytest

from eigenorbit_router_client import _client
from fake_router import FakeRouter


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any real socket connection fails loudly, so no test can reach a network."""

    def refuse(*args, **kwargs):
        raise RuntimeError("network access in tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def _clean_router_env(monkeypatch):
    """No test sees the developer's router key or URL from the shell."""
    monkeypatch.delenv("EIGENORBIT_ROUTER_API_KEY", raising=False)
    monkeypatch.delenv("EIGENORBIT_ROUTER_URL", raising=False)


@pytest.fixture
def fake_router(monkeypatch):
    """A FakeRouter wired in as the transport of every client the package builds.

    `router.sleeps` records the poll delays instead of sleeping.
    """
    router = FakeRouter()

    def make_http_client(*, base_url, timeout):
        return httpx.Client(
            base_url=base_url,
            timeout=timeout,
            follow_redirects=False,
            transport=router.transport,
        )

    router.sleeps = []
    monkeypatch.setattr(_client, "_make_http_client", make_http_client)
    monkeypatch.setattr(_client, "_sleep", router.sleeps.append)
    return router
