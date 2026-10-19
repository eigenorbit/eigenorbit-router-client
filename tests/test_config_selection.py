"""A kit user selects the router exactly as with any eigenkit config.

Importing eigenorbit_router_client registers `"router"` with eigenkit, so
`provider="router"` builds `RouterBackend()` (whose ValueError names
`RouterConfig(device=...)`), and a `RouterConfig` runs against the in-process
router double with the key from EIGENORBIT_ROUTER_API_KEY.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
from eigenkit import Circuit
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.observables import all_z

from eigenorbit_router_client import RouterConfig
from fake_router import VALID_KEY


SPEC = assemble([AngleEncoding(in_features=2), BasicVariational(depth=1)])
OBSERVABLES = all_z(2)
BASE_URL = "https://router.test"
NEEDS_A_DEVICE = "RouterConfig(device="


def _params():
    rng = np.random.default_rng(11)
    weights = rng.uniform(-np.pi, np.pi, SPEC.n_weights)
    inputs = rng.uniform(0.0, np.pi, SPEC.n_inputs)
    return weights, inputs


def _run(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter, so no earlier import has registered router."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )


def test_importing_the_client_registers_router():
    result = _run(
        """
        from eigenkit.core import provider

        # An eigenkit with its own router hint registers router itself when the
        # client is installed; only an eigenkit without one starts without it.
        if "router" not in provider._EXTRA_HINT:
            assert "router" not in provider.registered_names()
        import eigenorbit_router_client  # noqa: F401
        assert "router" in provider.registered_names()
        """
    )

    assert result.returncode == 0, result.stderr


def test_a_bare_router_string_asks_for_a_router_config():
    with pytest.raises(ValueError) as excinfo:
        Circuit(SPEC, OBSERVABLES, provider="router")

    assert NEEDS_A_DEVICE in str(excinfo.value)


def test_the_kit_registry_raises_the_same_value_error_not_an_import_error():
    from eigenkit.core import provider

    with pytest.raises(ValueError) as excinfo:
        provider.get("router")

    assert not isinstance(excinfo.value, ImportError)
    assert NEEDS_A_DEVICE in str(excinfo.value)


def test_the_environment_key_alone_runs_a_circuit(fake_router, monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)
    config = RouterConfig(device="braket/sv1", base_url=BASE_URL)
    assert config.api_key is None
    weights, inputs = _params()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = Circuit(SPEC, OBSERVABLES, provider=config)(weights, inputs)
    want = Circuit(SPEC, OBSERVABLES, provider="numpy")(weights, inputs)

    assert got.shape == want.shape
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    submits = [
        r
        for r in fake_router.requests
        if r.method == "POST" and r.url.path == "/v1/jobs"
    ]
    assert len(submits) == 1
    assert submits[0].headers["Authorization"] == f"Bearer {VALID_KEY}"


# ---- the gate set per vendor ----


def _backend(device: str):
    from eigenorbit_router_client import RouterBackend

    return RouterBackend(RouterConfig(device=device, api_key=VALID_KEY, base_url=BASE_URL))


def test_an_ibm_device_gets_the_kit_ibm_adapter_gate_set():
    from eigenkit.core.backends.qiskit_runtime_backend import (
        QiskitRuntimeBackend,
    )

    got = _backend("ibm/test_eagle").capabilities.supported_gates

    assert got == QiskitRuntimeBackend().capabilities.supported_gates
    assert "QubitUnitary" not in got
    assert "CPHASE" not in got


def test_a_braket_device_keeps_the_braket_gate_set():
    from eigenkit.core.ir import ALL_GATES

    got = _backend("braket/sv1").capabilities.supported_gates

    assert got == ALL_GATES - {"AmplitudeEncoding", "CPHASE", "ControlledQubitUnitary"}


def test_a_qubit_unitary_circuit_on_ibm_is_refused_before_any_request(fake_router):
    from eigenkit.core.blocks import QubitUnitary

    spec = assemble([QubitUnitary(np.array([[0.0, 1.0], [1.0, 0.0]]), wires=(0,))])
    config = RouterConfig(
        device="ibm/ibm_fez", shots=100, api_key=VALID_KEY, base_url=BASE_URL
    )

    with pytest.raises(ValueError, match="QubitUnitary"):
        Circuit(spec, all_z(1), provider=config)

    assert fake_router.requests == []
