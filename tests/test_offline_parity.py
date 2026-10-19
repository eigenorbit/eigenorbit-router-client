"""Offline codec parity against the router's own contract v1.

The router runs the same kit adapter server-side, so what can break between a
kit user and a real QPU is the wire codec round trip and version skew. This
suite checks both with no network, against the router's schemas and examples
vendored byte-for-byte under tests/contract/v1 (see scripts/sync_contract.py):

- vendoring: the tree matches its SOURCE.json manifest, and the sync script
  detects tampering and drift from a router checkout;
- (a) requests the client builds from real kit circuits validate against the
  JobRequest schema, and the contract's RY example is rebuilt exactly;
- (b) result payloads decode to the kit numpy path's shape, dtype, values and
  qubit0_msb bit order;
- (c) the contract's error examples map to the right RouterError;
- the in-process FakeRouter emits only schema-valid bodies, so the double the
  rest of the suite trusts cannot drift from the contract.

Never hand-edit tests/contract/: re-run the sync script from a router commit.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest
from eigenkit import Circuit, CostWarning
from eigenkit.core.blocks import AngleEncoding, BasicVariational, assemble
from eigenkit.core.ir import CircuitSpec, Gate, InputRef, WeightRef
from eigenkit.core.observables import X, Y, Z, all_z
from eigenkit.core.wire import circuit_spec_from_json, observables_from_json
from jsonschema import Draft202012Validator

from eigenorbit_router_client import RouterConfig, RouterError
from eigenorbit_router_client._decode import decode_result
from eigenorbit_router_client.backend import RouterBackend
from fake_router import VALID_KEY, FakeRouter


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_DIR = ROOT / "tests" / "contract"
SOURCE_JSON = CONTRACT_DIR / "SOURCE.json"
SCHEMAS_DIR = CONTRACT_DIR / "v1" / "schemas"
EXAMPLES_DIR = CONTRACT_DIR / "v1" / "examples"
SYNC_SCRIPT = ROOT / "scripts" / "sync_contract.py"

SCHEMA_NAMES = (
    "AccountBalanceOut",
    "AccountJobOut",
    "AccountJobPageOut",
    "ApiKeyCreateIn",
    "ApiKeyCreatedOut",
    "ApiKeyListOut",
    "ApiKeyUpdateIn",
    "DeviceListOut",
    "ErrorEnvelope",
    "EstimateOut",
    "HeldListOut",
    "HistoryPageOut",
    "IbmConnectionIn",
    "IbmConnectionOut",
    "IbmConnectionStateOut",
    "IbmDisconnectOut",
    "JobLedgerOut",
    "JobOut",
    "JobRequest",
    "JobResultOut",
    "SessionOut",
    "TermsAcceptIn",
    "TermsAcceptanceOut",
    "TermsStatusOut",
    "TermsTextOut",
    "TopupConfigOut",
    "TopupCreateIn",
    "TopupOut",
)

# Example filename (without .json) -> the schema it must validate against.
# The same mapping as EXAMPLE_MODELS in the router's test_contract_freeze.py.
EXAMPLE_SCHEMAS = {
    "estimate_request": "JobRequest",
    "estimate_response": "EstimateOut",
    "job_submit_request": "JobRequest",
    "job_accepted_response": "JobOut",
    "job_status_completed": "JobOut",
    "job_status_failed_circuit": "JobOut",
    "job_result_run": "JobResultOut",
    "job_result_zero_state_probability": "JobResultOut",
    "error_concurrency_limit_exceeded": "ErrorEnvelope",
    "error_device_unavailable": "ErrorEnvelope",
    "error_insufficient_balance": "ErrorEnvelope",
    "error_job_not_cancellable": "ErrorEnvelope",
    "error_terms_not_accepted": "ErrorEnvelope",
    "error_access_restricted": "ErrorEnvelope",
    "error_user_blocked": "ErrorEnvelope",
    "error_region_blocked": "ErrorEnvelope",
    "error_result_expired": "ErrorEnvelope",
    "error_protocol_unsupported": "ErrorEnvelope",
    "error_validation": "ErrorEnvelope",
    "key_created": "ApiKeyCreatedOut",
    "key_list": "ApiKeyListOut",
    "key_update_request": "ApiKeyUpdateIn",
    "error_key_revoked": "ErrorEnvelope",
    "error_key_budget_exceeded": "ErrorEnvelope",
    "error_key_name_taken": "ErrorEnvelope",
    "devices_list": "DeviceListOut",
    "session_exchanged": "SessionOut",
    "account_balance": "AccountBalanceOut",
    "terms_status": "TermsStatusOut",
    "terms_text": "TermsTextOut",
    "terms_accept_request": "TermsAcceptIn",
    "terms_accepted": "TermsAcceptanceOut",
    "topup_config": "TopupConfigOut",
    "topup_create_request": "TopupCreateIn",
    "topup_created": "TopupOut",
    "topup_credited": "TopupOut",
    "topup_refused": "TopupOut",
    "account_job": "AccountJobOut",
    "account_jobs_page": "AccountJobPageOut",
    "history_page": "HistoryPageOut",
    "held_list": "HeldListOut",
    "job_ledger": "JobLedgerOut",
    # IBM Quantum.
    "job_status_ibm_completed": "JobOut",
    "job_status_ibm_failed_target_mismatch": "JobOut",
    "estimate_response_ibm": "EstimateOut",
    "ibm_connection_state": "IbmConnectionStateOut",
    "ibm_connection_none": "IbmConnectionStateOut",
    "ibm_connection_request": "IbmConnectionIn",
    "ibm_disconnected": "IbmDisconnectOut",
    "error_ibm_account_not_connected": "ErrorEnvelope",
    "error_ibm_connection_invalid": "ErrorEnvelope",
    "error_ibm_token_rejected": "ErrorEnvelope",
    "error_ibm_connection_busy": "ErrorEnvelope",
    "devices_list_ibm": "DeviceListOut",
}

METHODS = ("run", "sample", "probabilities", "zero_state_probability")
BASE_URL = "https://router.test"
NO_INPUTS = np.zeros(0)


# ---- fixtures and helpers -----------------------------------------------------


def _read_json(path: Path):
    assert path.is_file(), f"missing vendored contract file {path.relative_to(ROOT)}"
    return json.loads(path.read_text())


def _schema(name: str) -> dict:
    return _read_json(SCHEMAS_DIR / f"{name}.schema.json")


@functools.cache
def _validator(name: str) -> Draft202012Validator:
    schema = _schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)


def _assert_valid(name: str, instance) -> None:
    errors = sorted(_validator(name).iter_errors(instance), key=lambda e: list(e.path))
    assert not errors, f"{name} rejects the body: " + "; ".join(
        f"{list(e.path)}: {e.message}" for e in errors
    )


def _example(stem: str) -> dict:
    return _read_json(EXAMPLES_DIR / f"{stem}.json")


@functools.cache
def _load_sync():
    spec = importlib.util.spec_from_file_location("sync_contract", SYNC_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sync():
    """scripts/sync_contract.py loaded as a module."""
    assert SYNC_SCRIPT.is_file(), f"missing {SYNC_SCRIPT.relative_to(ROOT)}"
    return _load_sync()


def _run_sync(*args: str) -> subprocess.CompletedProcess:
    assert SYNC_SCRIPT.is_file(), f"missing {SYNC_SCRIPT.relative_to(ROOT)}"
    return subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _client_copy(tmp_path: Path) -> Path:
    """A throwaway client root holding a copy of the vendored tree."""
    root = tmp_path / "client"
    assert CONTRACT_DIR.is_dir(), "missing tests/contract"
    shutil.copytree(CONTRACT_DIR, root / "tests" / "contract")
    return root


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=parity-test",
            "-c",
            "user.email=parity-test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _router_like_repo(tmp_path: Path) -> Path:
    """A throwaway git repo whose contract/v1 holds the vendored files."""
    repo = tmp_path / "router"
    target = repo / "contract" / "v1"
    for kind in ("schemas", "examples"):
        shutil.copytree(CONTRACT_DIR / "v1" / kind, target / kind)
    (target / "CONTRACT.md").write_text("not vendored\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "contract v1")
    return repo


def _commit_all(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _config(device: str = "braket/sv1", **overrides) -> RouterConfig:
    return RouterConfig(device, api_key=VALID_KEY, base_url=BASE_URL, **overrides)


@pytest.fixture(autouse=True)
def _compile_against_the_double(fake_router):
    """Compiling asks the router for one estimate, so every compile in this module
    talks to the in-process double, never a socket."""
    return fake_router


def _compile(spec: CircuitSpec, observables, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CostWarning)
        return RouterBackend(_config(**overrides)).compile(spec, list(observables))


# ---- vendoring: manifest and sync check -----------------------------------------


def test_the_vendored_tree_matches_its_manifest():
    assert _sync().verify(ROOT) == []


def test_verify_exits_0_on_the_fresh_tree():
    completed = _run_sync("--verify")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_source_json_records_the_router_commit_and_every_file():
    source = _read_json(SOURCE_JSON)
    assert source["repo"] == "eigenorbit-router"
    assert source["path"] == "contract/v1"
    assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
    vendored = sorted(
        p.relative_to(CONTRACT_DIR / "v1").as_posix()
        for p in (CONTRACT_DIR / "v1").rglob("*")
        if p.is_file()
    )
    assert sorted(source["files"]) == vendored
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in source["files"].values())
    assert SOURCE_JSON.read_text().endswith("}\n")


def test_exactly_the_router_schemas_and_examples_are_vendored():
    schemas = sorted(p.name for p in SCHEMAS_DIR.glob("*"))
    examples = sorted(p.stem for p in EXAMPLES_DIR.glob("*"))
    assert schemas == sorted(f"{name}.schema.json" for name in SCHEMA_NAMES)
    assert examples == sorted(EXAMPLE_SCHEMAS)
    assert not (CONTRACT_DIR / "v1" / "CONTRACT.md").exists()


def test_a_one_byte_edit_fails_verify_and_names_the_file(tmp_path):
    root = _client_copy(tmp_path)
    target = root / "tests" / "contract" / "v1" / "examples" / "job_result_run.json"
    content = bytearray(target.read_bytes())
    content[-2] = ord(" ") if content[-2] != ord(" ") else ord("\t")
    target.write_bytes(bytes(content))

    completed = _run_sync("--verify", "--root", str(root))

    assert completed.returncode == 1
    assert "examples/job_result_run.json" in completed.stdout + completed.stderr


def test_a_removed_or_unlisted_file_fails_verify(tmp_path):
    root = _client_copy(tmp_path)
    v1 = root / "tests" / "contract" / "v1"
    (v1 / "examples" / "key_created.json").unlink()
    (v1 / "examples" / "extra.json").write_text("{}\n")

    problems = _sync().verify(root)

    assert any("examples/key_created.json" in p for p in problems)
    assert any("examples/extra.json" in p for p in problems)
    assert _run_sync("--verify", "--root", str(root)).returncode == 1


def test_check_is_clean_against_a_router_with_the_same_contract(tmp_path):
    repo = _router_like_repo(tmp_path)

    assert _sync().diff_against_router(ROOT, repo, "HEAD") == []
    completed = _run_sync("--check", "--router", str(repo))
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_check_names_a_changed_added_or_removed_router_file(tmp_path):
    repo = _router_like_repo(tmp_path)
    examples = repo / "contract" / "v1" / "examples"
    changed = json.loads((examples / "error_validation.json").read_text())
    changed["error"]["message"] = "Validation failed"
    (examples / "error_validation.json").write_text(json.dumps(changed, indent=2) + "\n")
    (examples / "job_result_counts.json").write_text("{}\n")
    (repo / "contract" / "v1" / "schemas" / "ApiKeyListOut.schema.json").unlink()
    _commit_all(repo, "drift")

    problems = _sync().diff_against_router(ROOT, repo, "HEAD")
    completed = _run_sync("--check", "--router", str(repo))

    output = completed.stdout + completed.stderr
    assert completed.returncode == 1
    for name in (
        "examples/error_validation.json",
        "examples/job_result_counts.json",
        "schemas/ApiKeyListOut.schema.json",
    ):
        assert any(name in p for p in problems), name
        assert name in output


def test_check_reads_the_given_ref_not_the_working_tree(tmp_path):
    repo = _router_like_repo(tmp_path)
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "contract" / "v1" / "examples" / "key_created.json").write_text("{}\n")
    _commit_all(repo, "drift")

    assert _sync().diff_against_router(ROOT, repo, first) == []
    assert _sync().diff_against_router(ROOT, repo, "HEAD") != []


def test_vendoring_from_a_router_commit_rebuilds_the_tree(tmp_path):
    repo = _router_like_repo(tmp_path)
    root = _client_copy(tmp_path)
    v1 = root / "tests" / "contract" / "v1"
    (v1 / "schemas" / "JobOut.schema.json").unlink()
    (v1 / "examples" / "stale.json").write_text("{}\n")

    completed = _run_sync("--router", str(repo), "--root", str(root))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _sync().verify(root) == []
    source = json.loads((root / "tests" / "contract" / "SOURCE.json").read_text())
    assert source["commit"] == _git(repo, "rev-parse", "HEAD")
    assert source["files"] == _read_json(SOURCE_JSON)["files"]
    assert not (v1 / "examples" / "stale.json").exists()
    assert not (v1 / "CONTRACT.md").exists()


# ---- schemas and examples -------------------------------------------------------


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_each_vendored_schema_is_a_valid_draft_2020_12_schema(name):
    Draft202012Validator.check_schema(_schema(name))


@pytest.mark.parametrize(("stem", "schema"), sorted(EXAMPLE_SCHEMAS.items()))
def test_each_vendored_example_validates_against_its_schema(stem, schema):
    _assert_valid(schema, _example(stem))


# ---- (a) request encoding --------------------------------------------------------


def _ry_example_circuit():
    example = _example("job_submit_request")
    spec = circuit_spec_from_json(example["circuit_spec"])
    observables = observables_from_json(example["observables"])
    return spec, observables, np.array([0.5]), NO_INPUTS


def _angle_variational_circuit():
    spec = assemble([AngleEncoding(3), BasicVariational(1)])
    return spec, all_z(3), np.array([0.3, -0.7, 1.1]), np.array([0.2, 0.4, -0.6])


def _input_cnot_weighted_circuit():
    spec = CircuitSpec(
        n_qubits=2,
        n_weights=2,
        n_inputs=1,
        gates=(
            Gate("RX", (0,), (InputRef(0),)),
            Gate("RY", (1,), (WeightRef(0),)),
            Gate("CNOT", (0, 1)),
            Gate("RZ", (1,), (WeightRef(1),)),
        ),
    )
    observables = [
        (0.5 * Z(0) * Z(1) - 1.5 * X(1) + 0.25 * Y(0)).to_observable(),
        Z(1).to_observable(),
    ]
    return spec, observables, np.array([0.1, 0.2]), np.array([0.3])


CIRCUITS = {
    "ry_example": _ry_example_circuit,
    "angle_encoding_basic_variational": _angle_variational_circuit,
    "input_cnot_weighted_observable": _input_cnot_weighted_circuit,
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("circuit", sorted(CIRCUITS))
def test_built_requests_validate_against_the_job_request_schema(circuit, method):
    spec, observables, weights, inputs = CIRCUITS[circuit]()
    compiled = _compile(spec, observables, shots=100)
    try:
        request = compiled.build_request(method, weights, inputs)
    finally:
        compiled.close()

    _assert_valid("JobRequest", request)
    assert json.loads(json.dumps(request)) == request, "not plain JSON types"
    assert request["method"] == method
    assert request["weights"] == weights.tolist()
    assert request["inputs"] == inputs.tolist()


def test_the_ry_example_request_is_rebuilt_exactly():
    example = _example("job_submit_request")
    spec, observables, weights, inputs = _ry_example_circuit()
    compiled = _compile(spec, observables, device=example["device"])
    try:
        request = compiled.build_request(example["method"], weights, inputs)
    finally:
        compiled.close()

    assert request == example
    assert request == _example("estimate_request")


# ---- (b) result decoding ------------------------------------------------------


def _numpy(spec, observables):
    return Circuit(spec, list(observables), provider="numpy")


def _result_payload(method: str, result: dict) -> dict:
    return {
        "job_id": str(uuid.uuid4()),
        "method": method,
        "bit_order": "qubit0_msb",
        "result": result,
    }


def _array_result(values: np.ndarray) -> dict:
    return {
        "kind": "array",
        "dtype": "float64",
        "shape": list(values.shape),
        "data": values.tolist(),
    }


def test_the_run_example_decodes_to_the_numpy_run():
    payload = _example("job_result_run")
    spec, observables, weights, inputs = _ry_example_circuit()

    got = decode_result(
        payload, method="run", n_qubits=1, n_observables=1, job_id=payload["job_id"]
    )

    want = np.asarray(_numpy(spec, observables)(weights, inputs), dtype=np.float64)
    assert got.shape == (1,) and got.dtype == np.float64
    assert want.shape == got.shape
    np.testing.assert_allclose(got, [math.cos(0.5)], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_the_zero_state_probability_example_decodes_to_the_numpy_value():
    payload = _example("job_result_zero_state_probability")
    spec, observables, weights, inputs = _ry_example_circuit()

    got = decode_result(
        payload,
        method="zero_state_probability",
        n_qubits=1,
        n_observables=1,
        job_id=payload["job_id"],
    )

    want = _numpy(spec, observables).zero_state_probability(weights, inputs)
    assert isinstance(got, np.ndarray)
    assert got.shape == () and got.dtype == np.float64
    assert math.isclose(float(got), math.cos(0.25) ** 2, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-12)


def test_a_probabilities_payload_decodes_to_the_numpy_probabilities():
    spec = CircuitSpec(
        n_qubits=2,
        n_weights=2,
        n_inputs=0,
        gates=(Gate("RY", (0,), (WeightRef(0),)), Gate("RY", (1,), (WeightRef(1),))),
    )
    weights = np.array([0.3, 1.1])
    want = np.asarray(
        _numpy(spec, all_z(2)).probabilities(weights, NO_INPUTS), dtype=np.float64
    )
    payload = _result_payload("probabilities", _array_result(want))
    _assert_valid("JobResultOut", payload)

    got = decode_result(payload, method="probabilities", n_qubits=2, n_observables=2)

    assert got.shape == (4,) and got.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_qubit_0_is_the_most_significant_bit_in_probabilities_and_counts():
    spec = CircuitSpec(n_qubits=2, n_weights=0, n_inputs=0, gates=(Gate("X", (0,)),))
    probabilities = np.asarray(
        _numpy(spec, all_z(2)).probabilities(np.zeros(0), NO_INPUTS), dtype=np.float64
    )
    assert probabilities[2] == pytest.approx(1.0, abs=1e-12)

    array_payload = _result_payload("probabilities", _array_result(probabilities))
    _assert_valid("JobResultOut", array_payload)
    decoded = decode_result(array_payload, method="probabilities", n_qubits=2, n_observables=2)
    assert decoded[2] == pytest.approx(1.0, abs=1e-12)

    counts_payload = _result_payload(
        "sample", {"kind": "counts", "n_qubits": 2, "data": {"10": 1000}}
    )
    _assert_valid("JobResultOut", counts_payload)
    counts = decode_result(counts_payload, method="sample", n_qubits=2, n_observables=2)
    assert counts == {"10": 1000}
    assert int("10", 2) == int(np.argmax(probabilities)) == 2


# ---- (c) error mapping -----------------------------------------------------------


def _error_from_example(stem: str, status: int) -> RouterError:
    return RouterError.from_response(httpx.Response(status, json=_example(stem)))


def test_the_insufficient_balance_example_maps_to_402():
    example = _example("error_insufficient_balance")["error"]

    err = _error_from_example("error_insufficient_balance", 402)

    assert (err.code, err.http_status) == ("insufficient_balance", 402)
    assert err.details["job_id"] == example["details"]["job_id"]


def test_the_protocol_unsupported_example_maps_to_426_verbatim():
    example = _example("error_protocol_unsupported")["error"]

    err = _error_from_example("error_protocol_unsupported", 426)

    assert (err.code, err.http_status) == ("protocol_unsupported", 426)
    assert err.message == example["message"]
    assert "eigenorbit-router-client" in err.message
    assert "min_client_version" in err.details
    assert err.details == example["details"]


def test_the_validation_example_maps_to_422_with_its_errors():
    example = _example("error_validation")["error"]

    err = _error_from_example("error_validation", 422)

    assert (err.code, err.http_status) == ("validation_error", 422)
    assert err.details["errors"] == example["details"]["errors"]


def test_the_device_unavailable_example_maps_to_422_with_the_device():
    example = _example("error_device_unavailable")["error"]

    err = _error_from_example("error_device_unavailable", 422)

    assert (err.code, err.http_status) == ("device_unavailable", 422)
    assert err.details == example["details"]
    assert err.message.startswith(example["message"])
    assert example["details"]["device"] in err.message
    assert "GET /v1/devices" in err.message


def test_the_concurrency_example_maps_to_429_with_both_numbers():
    example = _example("error_concurrency_limit_exceeded")["error"]

    err = _error_from_example("error_concurrency_limit_exceeded", 429)

    assert (err.code, err.http_status) == ("concurrency_limit_exceeded", 429)
    assert err.details == example["details"]
    assert "wait for a job to finish or cancel one" in err.message


def test_the_job_not_cancellable_example_maps_to_409_with_the_status():
    example = _example("error_job_not_cancellable")["error"]

    err = _error_from_example("error_job_not_cancellable", 409)

    assert (err.code, err.http_status) == ("job_not_cancellable", 409)
    assert err.details == example["details"]
    assert f"already {example['details']['status']}" in err.message


def test_the_region_blocked_example_maps_to_451_with_the_sanctions_note():
    example = _example("error_region_blocked")["error"]

    err = _error_from_example("error_region_blocked", 451)

    assert (err.code, err.http_status) == ("region_blocked", 451)
    assert err.details == example["details"] == {}
    assert err.message.startswith(example["message"])
    assert "sanctions and export rules" in err.message


def test_the_access_restricted_example_maps_to_403_with_the_support_address():
    example = _example("error_access_restricted")["error"]

    err = _error_from_example("error_access_restricted", 403)

    assert (err.code, err.http_status) == ("access_restricted", 403)
    assert err.message.startswith(example["message"])
    assert "support@eigenorbit.com" in err.message


def test_the_user_blocked_example_maps_to_403():
    example = _example("error_user_blocked")["error"]

    err = _error_from_example("error_user_blocked", 403)

    assert (err.code, err.http_status) == ("user_blocked", 403)
    assert err.details == example["details"]
    assert err.message.startswith(example["message"])
    assert "will not help" in err.message


def test_the_result_expired_example_maps_to_410_with_expired_at():
    example = _example("error_result_expired")["error"]

    err = _error_from_example("error_result_expired", 410)

    assert (err.code, err.http_status) == ("result_expired", 410)
    assert err.details == example["details"]
    assert err.message.startswith(example["message"])
    assert example["details"]["expired_at"] in err.message


def test_the_failed_circuit_example_maps_to_its_category():
    example = _example("job_status_failed_circuit")

    err = RouterError.from_job(example)

    assert err.code == example["error"]["code"]
    assert err.category == "circuit"
    assert err.message == example["error"]["message"]
    assert err.details == {"status": "failed", "billing": "released"}


# ---- the FakeRouter double against the schemas -----------------------------------


def _headers(**extra: str) -> dict[str, str]:
    return {
        "EO-Protocol-Version": "1",
        "Authorization": f"Bearer {VALID_KEY}",
        **extra,
    }


def _request_body(method: str = "run", *, shots: int = 100) -> dict:
    spec, observables, weights, inputs = _input_cnot_weighted_circuit()
    compiled = _compile(spec, observables, shots=shots)
    try:
        return compiled.build_request(method, weights, inputs)
    finally:
        compiled.close()


@pytest.fixture
def double():
    router = FakeRouter()
    with httpx.Client(base_url=BASE_URL, transport=router.transport) as http:
        yield router, http


def _submit(http: httpx.Client, body: dict, key: str | None = None) -> httpx.Response:
    return http.post(
        "/v1/jobs", json=body, headers=_headers(**{"Idempotency-Key": key or uuid.uuid4().hex})
    )


def _poll_to_terminal(http: httpx.Client, job_id: str) -> list[dict]:
    seen = []
    for _ in range(10):
        response = http.get(f"/v1/jobs/{job_id}", headers=_headers())
        assert response.status_code == 200
        seen.append(response.json())
        if seen[-1]["status"] in ("completed", "failed", "cancelled", "rejected"):
            return seen
    raise AssertionError(f"job {job_id} never reached a terminal status")


def test_the_double_estimate_body_is_a_valid_estimate_out(double):
    _, http = double

    response = http.post("/v1/jobs/estimate", json=_request_body(), headers=_headers())

    assert response.status_code == 200
    _assert_valid("EstimateOut", response.json())


def test_the_double_submit_and_replay_bodies_are_valid_job_outs(double):
    _, http = double
    body = _request_body()

    created = _submit(http, body, key="replay-key")
    replayed = _submit(http, body, key="replay-key")

    assert created.status_code == 202
    assert replayed.status_code == 200
    assert replayed.headers["Idempotent-Replayed"] == "true"
    _assert_valid("JobOut", created.json())
    _assert_valid("JobOut", replayed.json())
    assert replayed.json()["id"] == created.json()["id"]


def test_the_double_status_bodies_are_valid_job_outs(double):
    _, http = double
    job_id = _submit(http, _request_body()).json()["id"]

    seen = _poll_to_terminal(http, job_id)

    assert [job["status"] for job in seen] == ["queued", "running", "completed"]
    for job in seen:
        _assert_valid("JobOut", job)


def test_the_double_failed_job_body_is_a_valid_job_out():
    router = FakeRouter(fail_with="vendor_failed")
    with httpx.Client(base_url=BASE_URL, transport=router.transport) as http:
        job_id = _submit(http, _request_body()).json()["id"]
        seen = _poll_to_terminal(http, job_id)

    assert seen[-1]["status"] == "failed"
    assert seen[-1]["error"]["code"] == "vendor_failed"
    for job in seen:
        _assert_valid("JobOut", job)


@pytest.mark.parametrize("method", METHODS)
def test_the_double_result_bodies_are_valid_and_decode(double, method):
    _, http = double
    body = _request_body(method)
    job_id = _submit(http, body).json()["id"]
    _poll_to_terminal(http, job_id)

    response = http.get(f"/v1/jobs/{job_id}/result", headers=_headers())

    assert response.status_code == 200
    payload = response.json()
    _assert_valid("JobResultOut", payload)
    decode_result(payload, method=method, n_qubits=2, n_observables=2, job_id=job_id)


def _error_cases(router: FakeRouter, http: httpx.Client):
    body = _request_body()
    yield "unauthenticated", 401, http.post(
        "/v1/jobs/estimate",
        json=body,
        headers={"EO-Protocol-Version": "1", "Authorization": "Bearer nope"},
    )
    _submit(http, body, key="conflict-key")
    yield "idempotency_conflict", 409, _submit(
        http, _request_body(shots=200), key="conflict-key"
    )
    router.supported_methods = frozenset({"run", "zero_state_probability"})
    yield "method_not_supported", 422, _submit(http, _request_body("probabilities"))
    yield "protocol_unsupported", 426, http.post(
        "/v1/jobs/estimate", json=body, headers={"Authorization": f"Bearer {VALID_KEY}"}
    )
    job_id = _submit(http, body).json()["id"]
    yield "job_not_completed", 409, http.get(f"/v1/jobs/{job_id}/result", headers=_headers())
    yield "job_not_found", 404, http.get(f"/v1/jobs/{uuid.uuid4()}", headers=_headers())
    yield "job_not_found", 404, http.post(
        f"/v1/jobs/{uuid.uuid4()}/cancel", headers=_headers()
    )
    _poll_to_terminal(http, job_id)
    yield "job_not_cancellable", 409, http.post(
        f"/v1/jobs/{job_id}/cancel", headers=_headers()
    )
    router.device_status = {"braket/sv1": "offline"}
    yield "device_unavailable", 422, _submit(http, body)
    router.device_status = {}
    router.stale_devices = frozenset({"braket/sv1"})
    yield "device_unavailable", 422, _submit(http, body)
    router.stale_devices = frozenset()
    router.supported_methods = frozenset(METHODS)
    # The idempotency-conflict job above is still queued: one job in flight.
    router.concurrency_limit = 1
    yield "concurrency_limit_exceeded", 429, _submit(http, body)


def test_the_double_error_bodies_are_valid_error_envelopes(double):
    router, http = double
    seen = []

    for code, status, response in _error_cases(router, http):
        assert response.status_code == status, code
        _assert_valid("ErrorEnvelope", response.json())
        assert response.json()["error"]["code"] == code
        seen.append(code)

    assert {"unauthenticated", "idempotency_conflict", "method_not_supported"} <= set(seen)
    assert {"protocol_unsupported", "job_not_cancellable", "device_unavailable"} <= set(seen)
    assert "concurrency_limit_exceeded" in seen


def test_the_double_cancel_bodies_are_valid_job_outs(double):
    router, http = double
    immediate = _submit(http, _request_body()).json()["id"]
    first = http.post(f"/v1/jobs/{immediate}/cancel", headers=_headers())
    again = http.post(f"/v1/jobs/{immediate}/cancel", headers=_headers())
    router.cancel_mode = "vendor"
    at_vendor = _submit(http, _request_body()).json()["id"]
    requested = http.post(f"/v1/jobs/{at_vendor}/cancel", headers=_headers())
    seen = _poll_to_terminal(http, at_vendor)

    assert (first.status_code, again.status_code, requested.status_code) == (202, 202, 202)
    assert first.json()["status"] == "cancelled"
    assert first.json()["billing"] == "released"
    assert requested.json()["status"] == "queued"
    assert seen[-1]["status"] == "cancelled"
    for body in (first.json(), again.json(), requested.json(), *seen):
        _assert_valid("JobOut", body)


def test_the_double_estimate_reports_the_new_reasons(double):
    router, http = double

    router.device_status = {"braket/sv1": "retired"}
    retired = http.post("/v1/jobs/estimate", json=_request_body(), headers=_headers())
    router.device_status = {}
    router.concurrency_limit = 0
    capped = http.post("/v1/jobs/estimate", json=_request_body(), headers=_headers())

    assert retired.json()["would_reject_reason"] == "device_unavailable"
    assert capped.json()["would_reject_reason"] == "concurrency_limit_exceeded"
    for response in (retired, capped):
        _assert_valid("EstimateOut", response.json())
