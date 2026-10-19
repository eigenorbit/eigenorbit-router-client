"""Offline tests for scripts/live_parity.py.

The live script compares a device through the user's own vendor credentials
with the same device through the router. Its pure parts (tolerances,
comparison), its router leg (against the in-process FakeRouter) and its command
line are tested here. Nothing here opens a socket, and the script imports no
AWS SDK: the own-credentials leg only runs by hand. Only the mc3 lowering test
imports the Braket SDK, and only when the live group installs it. The IBM leg
is tested the same way, with the own leg replaced by exact values and no IBM
token ever sent anywhere.
"""

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from eigenorbit_router_client import RouterConfig, _client
from fake_router import VALID_KEY


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_parity.py"
BASE_URL = "https://router.test"


def _braket_modules() -> set[str]:
    return {name for name in sys.modules if name == "braket" or name.startswith("braket.")}


def _qiskit_modules() -> set[str]:
    return {name for name in sys.modules if name == "qiskit" or name.startswith("qiskit.")}


class _MissingScript:
    """Stands in for the script when it does not exist, so every test fails on
    an assertion naming it instead of erroring in the fixture."""

    def __getattr__(self, name):
        raise AssertionError(f"{SCRIPT} does not exist (looked up {name!r})")


@pytest.fixture(scope="module")
def parity():
    if not SCRIPT.is_file():
        yield _MissingScript()
        return
    before = _braket_modules()
    qiskit_before = _qiskit_modules()
    spec = importlib.util.spec_from_file_location("live_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["live_parity"] = module
    spec.loader.exec_module(module)
    module._braket_modules_added_on_import = _braket_modules() - before
    module._qiskit_modules_added_on_import = _qiskit_modules() - qiskit_before
    yield module
    sys.modules.pop("live_parity", None)


@pytest.fixture
def quiet():
    """Compiles against the router emit a CostWarning; these tests ignore it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---- import hygiene ---------------------------------------------------------------


def test_importing_the_script_imports_no_aws_sdk(parity):
    assert parity._braket_modules_added_on_import == set()


# ---- tolerance --------------------------------------------------------------------


def test_the_exact_tolerance_is_1e_9(parity):
    assert parity.tolerance(0, coeffs=[1.0], term_expectations=[0.6]) == 1e-9
    assert parity.tolerance(0, p=0.5) == 1e-9


def test_a_single_term_tolerance_is_k_sigma_over_both_legs(parity):
    got = parity.tolerance(1000, coeffs=[1.0], term_expectations=[0.6], k=5)

    assert abs(got - 5 * math.sqrt(2 * (1 - 0.36) / 1000)) <= 1e-12


def test_the_zero_state_probability_tolerance_is_binomial(parity):
    got = parity.tolerance(1000, p=0.9, k=5)

    assert abs(got - 5 * math.sqrt(2 * 0.9 * 0.1 / 1000)) <= 1e-12


def test_sigma_is_floored_at_one_over_n_when_the_term_is_certain(parity):
    got = parity.tolerance(1000, coeffs=[1.0], term_expectations=[1.0], k=5)

    assert abs(got - 5 * math.sqrt(2) / 1000) <= 1e-12


def test_weighted_terms_add_their_sigmas(parity):
    got = parity.tolerance(
        400, coeffs=[0.5, -0.25], term_expectations=[0.0, 0.8], k=5, legs=1
    )
    sigma = 0.5 * math.sqrt(1 / 400) + 0.25 * math.sqrt((1 - 0.64) / 400)

    assert abs(got - 5 * sigma) <= 1e-12


# ---- compare ----------------------------------------------------------------------


def test_compare_fails_on_different_shapes(parity):
    result = parity.compare(np.zeros(2), np.zeros(1), 1e-3, case="ry/run/shots=0")

    assert result["passed"] is False
    assert "shape" in result["reason"]


def test_compare_fails_on_different_dtypes(parity):
    result = parity.compare(
        np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float32), 1e-3, case="c"
    )

    assert result["passed"] is False
    assert "dtype" in result["reason"]


def test_compare_passes_within_the_tolerance(parity):
    result = parity.compare(np.array([0.5]), np.array([0.505]), 0.01, case="c")

    assert result["passed"] is True


def test_compare_fails_just_past_the_tolerance(parity):
    tol = 0.01
    result = parity.compare(np.array([0.5]), np.array([0.5 + tol + 1e-6]), tol, case="c")

    assert result["passed"] is False


def test_a_failure_report_names_the_case_and_both_values(parity):
    result = parity.compare(0.25, 0.75, 1e-3, case="cnot3/zero_state_probability/shots=1000")

    assert result["passed"] is False
    assert "cnot3/zero_state_probability/shots=1000" in result["report"]
    assert "0.25" in result["report"]
    assert "0.75" in result["report"]


def test_compare_takes_a_tolerance_per_observable(parity):
    result = parity.compare(
        np.array([0.1, 0.2]), np.array([0.1, 0.25]), np.array([0.01, 0.1]), case="c"
    )

    assert result["passed"] is True


# ---- cases and references ---------------------------------------------------------


def test_the_cases_skip_exact_zero_state_probability(parity):
    cases = parity.build_cases(parity.select_circuits("all"), shots=1000)

    labels = {case.label for case in cases}
    assert len(cases) == 9
    for name in ("ry", "ansatz2", "cnot3"):
        assert f"{name}/run/shots=0" in labels
        assert f"{name}/run/shots=1000" in labels
        assert f"{name}/zero_state_probability/shots=1000" in labels
        assert f"{name}/zero_state_probability/shots=0" not in labels


def test_the_all_set_is_ry_ansatz2_and_cnot3(parity):
    names = [circuit.name for circuit in parity.select_circuits("all")]

    assert names == ["ry", "ansatz2", "cnot3"]


def test_the_garnet_set_is_ansatz2_and_mc3(parity):
    names = [circuit.name for circuit in parity.select_circuits("garnet")]

    assert names == ["ansatz2", "mc3"]


def test_one_circuit_can_be_selected_by_name(parity):
    assert [circuit.name for circuit in parity.select_circuits("mc3")] == ["mc3"]


def test_the_command_line_offers_the_garnet_set_and_mc3(parity):
    for which in ("all", "garnet", "ry", "ansatz2", "cnot3", "mc3"):
        assert parity._parser().parse_args(["--circuits", which]).circuits == which
    assert parity._parser().parse_args([]).circuits == "all"


def test_mc3_is_a_3_qubit_mcz_mcx_cry_circuit_measured_with_z_only(parity):
    [mc3] = parity.select_circuits("mc3")

    assert mc3.spec.n_qubits == 3
    assert mc3.spec.n_weights == 4
    assert mc3.spec.n_inputs == 0
    assert len(mc3.weights) == 4
    assert len(mc3.inputs) == 0
    assert {"MCZ", "MCX", "CRY"} <= {gate.name for gate in mc3.spec.gates}
    assert len(mc3.observables) == 3
    for observable in mc3.observables:
        for term in observable.terms:
            assert {pauli for pauli, _ in term.paulis} == {"Z"}
    exact = parity.exact_value(parity.Case(mc3, "run", 0))
    assert exact.shape == (3,)
    assert np.all(np.abs(exact) <= 1.0)


def test_garnet_qpu_cases_are_run_and_zero_state_probability_per_circuit(parity):
    cases = parity.build_cases(parity.select_circuits("garnet"), 100, exact=False)

    assert [case.label for case in cases] == [
        "ansatz2/run/shots=100",
        "ansatz2/zero_state_probability/shots=100",
        "mc3/run/shots=100",
        "mc3/zero_state_probability/shots=100",
    ]


def test_mc3_lowers_to_gates_a_braket_qpu_accepts(parity):
    # Only with the live group: the script itself never imports the AWS SDK.
    pytest.importorskip("braket")
    from eigenkit.configs import BraketConfig

    [mc3] = parity.select_circuits("mc3")
    garnet = parity.DEVICES["braket/iqm_garnet"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the CostWarning of a QPU compile
        kit_circuit = parity.Circuit(
            mc3.spec, mc3.observables, provider=BraketConfig(device_arn=garnet.arn, shots=100)
        )
    compiled = kit_circuit.compiled

    # Compiling builds no AwsDevice, so this stays offline.
    assert compiled.dry_run_info()["n_circuit_runs_per_call"] == 1
    names = {
        type(instruction.operator).__name__
        for instruction in compiled._circuit_for("run").instructions
    }
    assert names <= {"CCNot", "CNot", "H", "Ry"}
    assert {"CCNot", "CNot"} <= names


def test_exact_terms_split_the_observable_into_its_pauli_terms(parity):
    [cnot3] = parity.select_circuits("cnot3")

    terms = parity.exact_terms(
        cnot3.spec, cnot3.observables[0], cnot3.weights, cnot3.inputs
    )

    assert [coeff for coeff, _ in terms] == [0.5, 0.25]
    for _, value in terms:
        assert -1.0 <= value <= 1.0
    exact = parity.exact_value(parity.Case(cnot3, "run", 0))
    assert exact[0] == pytest.approx(0.5 * terms[0][1] + 0.25 * terms[1][1], abs=1e-12)


# ---- router leg -------------------------------------------------------------------


def test_the_router_leg_returns_kit_shapes(parity, fake_router, quiet):
    cases = parity.build_cases(parity.select_circuits("all"), shots=1000)
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY, base_url=BASE_URL)

    values = parity.run_router_leg(cases, config)

    assert len(values) == len(cases)
    for case, value in zip(cases, values):
        if case.method == "run":
            assert isinstance(value, np.ndarray)
            assert value.shape == (len(case.circuit.observables),)
            assert value.dtype == np.float64
        else:
            assert isinstance(value, float)
        np.testing.assert_allclose(value, parity.exact_value(case), atol=1e-12)


def test_the_router_leg_sends_each_case_at_its_shots(parity, fake_router, quiet):
    cases = parity.build_cases(parity.select_circuits("ry"), shots=250)
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY, base_url=BASE_URL)

    parity.run_router_leg(cases, config)

    submits = [
        json.loads(r.content)
        for r in fake_router.requests
        if r.method == "POST" and r.url.path == "/v1/jobs"
    ]
    assert [(s["method"], s["shots"]) for s in submits] == [
        ("run", 0),
        ("run", 250),
        ("zero_state_probability", 250),
    ]


def test_the_submit_check_records_the_job_handle(parity, fake_router, quiet):
    [ry] = parity.select_circuits("ry")
    config = RouterConfig(
        device="braket/sv1", shots=100, api_key=VALID_KEY, base_url=BASE_URL
    )

    record = parity.run_submit_check(ry, config)

    assert record["job_id"] in fake_router.jobs
    assert record["status_before_result"] in {"queued", "running", "completed"}
    assert record["status_after_result"] == "completed"
    assert record["shape"] == [1]
    assert record["passed"] is True


@pytest.fixture
def opened_clients(fake_router, monkeypatch):
    """Every httpx client the package builds, so a test can check they are closed."""
    make = _client._make_http_client
    opened = []

    def recording(*, base_url, timeout):
        http = make(base_url=base_url, timeout=timeout)
        opened.append(http)
        return http

    monkeypatch.setattr(_client, "_make_http_client", recording)
    return opened


def test_the_router_leg_closes_every_client_it_opens(parity, opened_clients, quiet):
    cases = parity.build_cases(parity.select_circuits("all"), shots=100)
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY, base_url=BASE_URL)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        parity.run_router_leg(cases, config)

    assert opened_clients
    assert all(http.is_closed for http in opened_clients)


def test_a_failing_router_leg_still_closes_its_clients(
    parity, fake_router, opened_clients, quiet
):
    fake_router.fail_with = "vendor_failed"
    cases = parity.build_cases(parity.select_circuits("ry"), shots=100)
    config = RouterConfig(device="braket/sv1", api_key=VALID_KEY, base_url=BASE_URL)

    values = parity.run_router_leg(cases, config)

    assert all(isinstance(value, parity.LegFailure) for value in values)
    assert opened_clients
    assert all(http.is_closed for http in opened_clients)


def test_the_submit_check_closes_its_client(parity, opened_clients, quiet):
    [ry] = parity.select_circuits("ry")
    config = RouterConfig(
        device="braket/sv1", shots=100, api_key=VALID_KEY, base_url=BASE_URL
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        record = parity.run_submit_check(ry, config)

    assert record["passed"] is True
    assert opened_clients
    assert all(http.is_closed for http in opened_clients)


# ---- command line -----------------------------------------------------------------


def test_main_refuses_to_run_without_an_explicit_router_url(parity, monkeypatch, capsys):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)

    code = parity.main(["--router-only"])

    assert code == 2
    assert "EIGENORBIT_ROUTER_URL" in capsys.readouterr().err


def test_main_refuses_to_run_without_an_api_key(parity, monkeypatch, capsys):
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)

    code = parity.main(["--router-only"])

    assert code == 2
    assert "EIGENORBIT_ROUTER_API_KEY" in capsys.readouterr().err


def test_router_only_mode_passes_against_the_double_and_writes_no_key(
    parity, fake_router, monkeypatch, tmp_path, capsys, quiet
):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)
    report_path = tmp_path / "router-only.json"

    code = parity.main(["--router-only", "--shots", "500", "--report", str(report_path)])

    assert code == 0
    text = report_path.read_text()
    assert "eo_test_" not in text
    assert "eo_test_" not in capsys.readouterr().out
    report = json.loads(text)
    assert report["passed"] is True
    assert report["mode"] == "router-only"
    assert report["router_host"] == "router.test"
    assert len(report["cases"]) == 9
    assert all(case["passed"] for case in report["cases"])
    assert report["submit_check"]["passed"] is True
    assert report["kit_version"]
    assert report["client_version"]


def test_a_failing_case_makes_main_exit_1(parity, fake_router, monkeypatch, tmp_path, quiet):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)
    fake_router.fail_with = "vendor_rejected"
    report_path = tmp_path / "r.json"

    code = parity.main(["--router-only", "--circuits", "ry", "--report", str(report_path)])

    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert "vendor_rejected" in report["cases"][0]["reason"]


# ---- devices and the QPU rule -----------------------------------------------------

GARNET = "braket/iqm_garnet"
CEPHEUS = "braket/rigetti_cepheus_1_108q"


def _submits(fake_router) -> list:
    return [r for r in fake_router.requests if r.method == "POST" and r.url.path == "/v1/jobs"]


@pytest.fixture
def router_env(monkeypatch):
    monkeypatch.setenv("EIGENORBIT_ROUTER_API_KEY", VALID_KEY)
    monkeypatch.setenv("EIGENORBIT_ROUTER_URL", BASE_URL)


@pytest.fixture
def qpu_router(fake_router):
    """The FakeRouter with the router's two Braket QPUs in its catalog."""
    fake_router.devices = fake_router.devices | {GARNET, CEPHEUS}
    return fake_router


@pytest.fixture
def own_leg(parity, monkeypatch):
    """Replaces the own-credentials leg: exact values plus `offset`, recorded."""
    calls = []
    state = {"offset": 0.0}

    def fake(cases, device=None):
        calls.append((list(cases), device))
        return [
            np.asarray(parity.exact_value(case)) + state["offset"]
            if case.method == "run"
            else float(parity.exact_value(case)) + state["offset"]
            for case in cases
        ]

    monkeypatch.setattr(parity, "run_own_leg", fake)
    fake.calls = calls
    fake.state = state
    return fake


def test_the_devices_table_carries_the_router_catalog_arns_and_regions(parity):
    garnet = parity.DEVICES[GARNET]
    assert garnet.arn == "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
    assert garnet.region == "eu-north-1"
    assert garnet.kind == "qpu"
    assert (garnet.min_shots, garnet.max_shots) == (1, 20000)
    cepheus = parity.DEVICES[CEPHEUS]
    assert cepheus.arn == "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
    assert cepheus.region == "us-west-1"
    assert cepheus.kind == "qpu"
    assert (cepheus.min_shots, cepheus.max_shots) == (10, 50000)
    sv1 = parity.DEVICES["braket/sv1"]
    assert (sv1.arn, sv1.region, sv1.kind) == (parity.SV1_ARN, "us-east-1", "simulator")
    dm1 = parity.DEVICES["braket/dm1"]
    assert dm1.arn == "arn:aws:braket:::device/quantum-simulator/amazon/dm1"
    assert (dm1.region, dm1.kind) == ("us-east-1", "simulator")


def test_the_default_device_is_sv1(parity):
    assert parity._parser().parse_args([]).device == "braket/sv1"


def test_an_unknown_device_exits_2_and_lists_the_supported_ids(
    parity, router_env, fake_router, capsys
):
    code = parity.main(["--device", "braket/nope", "--router-only"])

    assert code == 2
    err = capsys.readouterr().err
    assert "braket/nope" in err
    for device_id in parity.DEVICES:
        assert device_id in err
    assert fake_router.requests == []


def test_the_own_leg_runs_on_the_device_arn_in_its_region(parity, monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    seen = {}

    def fake_run_leg(leg, cases, make_provider):
        seen["leg"] = leg
        seen["config"] = make_provider(100)
        seen["regions"] = (
            parity.os.environ.get("AWS_REGION"),
            parity.os.environ.get("AWS_DEFAULT_REGION"),
        )
        return []

    monkeypatch.setattr(parity, "_run_leg", fake_run_leg)

    parity.run_own_leg([], parity.DEVICES[GARNET])

    assert seen["leg"] == "own"
    assert seen["config"].device_arn == "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
    assert seen["config"].shots == 100
    assert seen["regions"] == ("eu-north-1", "eu-north-1")


def test_qpu_cases_have_no_exact_run(parity):
    cases = parity.build_cases(parity.select_circuits("all"), 100, exact=False)

    labels = {case.label for case in cases}
    assert len(cases) == 6
    assert all(case.shots == 100 for case in cases)
    assert "ry/run/shots=100" in labels
    assert "ry/zero_state_probability/shots=100" in labels


def test_the_qpu_tolerance_is_never_below_the_absolute_floor(parity):
    assert parity.QPU_ABS_TOL == 0.15
    assert parity.qpu_tolerance(20000, coeffs=[1.0], term_expectations=[0.6]) == 0.15
    assert parity.qpu_tolerance(20000, p=0.5) == 0.15
    wide = parity.tolerance(10, coeffs=[1.0], term_expectations=[0.0])
    assert wide > 0.15
    assert parity.qpu_tolerance(10, coeffs=[1.0], term_expectations=[0.0]) == wide


def test_the_qpu_rule_passes_0_12_apart_and_fails_0_2_apart(parity):
    [ry] = parity.select_circuits("ry")
    for method in ("run", "zero_state_probability"):
        case = parity.Case(ry, method, 20000)
        tol = parity.case_tolerance(case, 5.0, 2, kind="qpu")
        exact = np.asarray(parity.exact_value(case))

        assert parity.compare(exact + 0.12, exact, tol)["passed"] is True
        assert parity.compare(exact + 0.2, exact, tol)["passed"] is False


def test_the_qpu_rule_keeps_shape_and_dtype_exact(parity):
    [ry] = parity.select_circuits("ry")
    tol = parity.case_tolerance(parity.Case(ry, "run", 20000), 5.0, 2, kind="qpu")

    shape = parity.compare(np.zeros(2), np.zeros(1), tol)
    dtype = parity.compare(np.zeros(1), np.zeros(1, dtype=np.float32), tol)

    assert shape["passed"] is False and "shape" in shape["reason"]
    assert dtype["passed"] is False and "dtype" in dtype["reason"]


def test_the_simulator_rule_is_unchanged(parity):
    [ry] = parity.select_circuits("ry")
    case = parity.Case(ry, "run", 20000)

    default = parity.case_tolerance(case, 5.0, 2)
    simulator = parity.case_tolerance(case, 5.0, 2, kind="simulator")

    np.testing.assert_array_equal(default, simulator)
    assert float(np.max(simulator)) < 0.05


def test_a_qpu_without_i_accept_cost_prints_the_estimate_and_submits_nothing(
    parity, router_env, qpu_router, own_leg, capsys, quiet
):
    code = parity.main(["--device", GARNET, "--shots", "100"])

    assert code == 3
    assert _submits(qpu_router) == []
    assert own_leg.calls == []
    captured = capsys.readouterr()
    assert "--i-accept-cost" in captured.err
    # 6 QPU cases at the FakeRouter's 90000 micro-USD (75000 vendor + 15000 fee).
    assert "$0.540000" in captured.out
    assert "$0.450000" in captured.out
    assert "$0.090000" in captured.out
    assert "eo_test_" not in captured.out + captured.err


def test_the_garnet_set_without_i_accept_cost_prints_the_estimate_and_submits_nothing(
    parity, router_env, qpu_router, own_leg, capsys, quiet
):
    code = parity.main(["--device", GARNET, "--circuits", "garnet", "--shots", "100"])

    assert code == 3
    assert _submits(qpu_router) == []
    assert own_leg.calls == []
    estimates = [
        json.loads(r.content)
        for r in qpu_router.requests
        if r.method == "POST" and r.url.path == "/v1/jobs/estimate"
    ]
    assert [(body["method"], body["shots"]) for body in estimates] == [
        ("run", 100),
        ("zero_state_probability", 100),
    ] * 2
    captured = capsys.readouterr()
    assert "--i-accept-cost" in captured.err
    # 4 QPU cases at the FakeRouter's 90000 micro-USD (75000 vendor + 15000 fee).
    assert "router estimate for 4 jobs" in captured.out
    assert "$0.360000" in captured.out
    assert "mc3/run/shots=100" in captured.out
    assert "eo_test_" not in captured.out + captured.err


def test_the_garnet_set_passes_router_only_on_a_simulator(
    parity, router_env, fake_router, tmp_path, quiet
):
    report_path = tmp_path / "rehearse.json"

    code = parity.main(
        ["--router-only", "--circuits", "garnet", "--shots", "100", "--report", str(report_path)]
    )

    assert code == 0
    report = json.loads(report_path.read_text())
    assert [case["case"] for case in report["cases"]] == [
        "ansatz2/run/shots=0",
        "ansatz2/run/shots=100",
        "ansatz2/zero_state_probability/shots=100",
        "mc3/run/shots=0",
        "mc3/run/shots=100",
        "mc3/zero_state_probability/shots=100",
    ]
    assert all(case["passed"] for case in report["cases"])


def test_router_only_on_a_qpu_also_needs_i_accept_cost(
    parity, router_env, qpu_router, capsys, quiet
):
    code = parity.main(["--device", GARNET, "--router-only", "--shots", "100"])

    assert code == 3
    assert _submits(qpu_router) == []


def test_shots_below_the_device_minimum_are_refused_before_anything_is_sent(
    parity, router_env, qpu_router, own_leg, capsys
):
    code = parity.main(["--device", CEPHEUS, "--shots", "5", "--i-accept-cost"])

    assert code == 2
    assert qpu_router.requests == []
    assert own_leg.calls == []
    assert "10" in capsys.readouterr().err


def test_shots_above_the_device_maximum_are_refused(parity, router_env, qpu_router, capsys):
    code = parity.main(["--device", GARNET, "--shots", "20001", "--i-accept-cost"])

    assert code == 2
    assert qpu_router.requests == []


def test_a_qpu_run_estimates_first_then_passes_within_the_qpu_rule(
    parity, router_env, qpu_router, own_leg, tmp_path, capsys, quiet
):
    own_leg.state["offset"] = 0.12
    report_path = tmp_path / "parity-garnet.json"

    code = parity.main(
        [
            "--device",
            GARNET,
            "--shots",
            "20000",
            "--i-accept-cost",
            "--report",
            str(report_path),
        ]
    )

    assert code == 0
    paths = [r.url.path for r in qpu_router.requests]
    first_submit = paths.index("/v1/jobs")
    assert paths[:6] == ["/v1/jobs/estimate"] * 6
    assert first_submit >= 6
    [(own_cases, own_device)] = own_leg.calls
    assert own_device is parity.DEVICES[GARNET]
    assert len(own_cases) == 6
    text = report_path.read_text()
    assert "eo_test_" not in text
    report = json.loads(text)
    assert report["passed"] is True
    assert report["router_device"] == GARNET
    assert report["device_kind"] == "qpu"
    assert report["qpu_abs_tol"] == 0.15
    assert "0.15" in report["acceptance_rule"]
    assert "shape" in report["acceptance_rule"]
    assert "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet" in report["reference"]
    assert report["own_region"] == "eu-north-1"
    assert report["estimate"]["estimate_usd_micro"] == 6 * 90_000
    assert len(report["cases"]) == 6
    assert all("shots=0" not in case["case"] for case in report["cases"])
    for case in report["cases"]:
        assert case["max_abs_diff"] == pytest.approx(0.12, abs=1e-3)
        assert min(np.ravel(case["tolerance"])) >= 0.15
    assert "0.15" in capsys.readouterr().out


def test_a_qpu_run_fails_when_the_legs_are_0_2_apart(
    parity, router_env, qpu_router, own_leg, tmp_path, quiet
):
    own_leg.state["offset"] = 0.2
    report_path = tmp_path / "r.json"

    code = parity.main(
        ["--device", GARNET, "--shots", "20000", "--i-accept-cost", "--report", str(report_path)]
    )

    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert all(case["passed"] is False for case in report["cases"])


def test_a_qpu_the_router_would_reject_is_not_run(
    parity, router_env, qpu_router, own_leg, capsys, quiet
):
    qpu_router.would_reject_reason = "insufficient_balance"

    code = parity.main(["--device", GARNET, "--shots", "100", "--i-accept-cost"])

    assert code == 2
    assert _submits(qpu_router) == []
    assert own_leg.calls == []
    assert "insufficient_balance" in capsys.readouterr().err


def test_the_sv1_report_keeps_the_simulator_rule(
    parity, router_env, fake_router, tmp_path, quiet
):
    report_path = tmp_path / "sv1.json"

    code = parity.main(["--router-only", "--circuits", "ry", "--report", str(report_path)])

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["router_device"] == "braket/sv1"
    assert report["device_kind"] == "simulator"
    assert report.get("estimate") is None
    assert "shot-noise" in report["acceptance_rule"]
    assert [case["case"] for case in report["cases"]] == [
        "ry/run/shots=0",
        "ry/run/shots=1000",
        "ry/zero_state_probability/shots=1000",
    ]


# ---- IBM devices ------------------------------------------------------------------

IBM_TEST = "ibm/ibm_test"
IBM_TOKEN = "ibm-token-never-printed-0123456789"
IBM_CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/acct:inst::"


def _estimates(router) -> list[dict]:
    return [
        json.loads(r.content)
        for r in router.requests
        if r.method == "POST" and r.url.path == "/v1/jobs/estimate"
    ]


@pytest.fixture
def ibm_router(fake_router):
    """The FakeRouter with ibm/ibm_test in its catalog, running IBM jobs on the
    IBM account the user connected."""
    fake_router.devices = fake_router.devices | {IBM_TEST}
    fake_router.credential_source = "own"
    return fake_router


@pytest.fixture
def ibm_env(monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", IBM_TOKEN)
    monkeypatch.setenv("IBM_INSTANCE_CRN", IBM_CRN)


@pytest.fixture
def no_ibm_env(monkeypatch):
    for name in ("QISKIT_IBM_TOKEN", "IBM_INSTANCE_CRN"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def ibm_own_leg(parity, monkeypatch):
    """Replaces the own leg with exact values (deterministic counts for `sample`,
    the FakeRouter's own rule), shifted by `offset`; `reverse_keys` writes the
    sample keys qubit-0-LSB."""
    from fake_router import largest_remainder_counts

    calls = []
    state = {"offset": 0.0, "reverse_keys": False}

    def value(case):
        exact = parity.exact_value(case)
        n_qubits = case.circuit.spec.n_qubits
        if case.method == "sample":
            counts = largest_remainder_counts(np.asarray(exact), case.shots, n_qubits)
            if state["reverse_keys"]:
                counts = {key[::-1]: count for key, count in counts.items()}
            return counts
        return np.asarray(exact, dtype=np.float64) + state["offset"]

    def fake(cases, device=None):
        calls.append((list(cases), device))
        return [value(case) for case in cases]

    monkeypatch.setattr(parity, "run_own_leg", fake)
    fake.calls = calls
    fake.state = state
    return fake


def _ibm_main(parity, *extra):
    return parity.main(["--device", IBM_TEST, *extra])


def test_importing_the_script_imports_no_qiskit(parity):
    assert parity._qiskit_modules_added_on_import == set()
    assert "QiskitRuntimeConfig" not in vars(parity)


def test_ibm_cases_are_run_sample_and_probabilities_only(parity):
    cases = parity.build_ibm_cases(parity.select_circuits("all"), 100)

    labels = [case.label for case in cases]
    assert len(cases) == 9
    for name in ("ry", "ansatz2", "cnot3"):
        for method in ("run", "sample", "probabilities"):
            assert f"{name}/{method}/shots=100" in labels
    assert not any("zero_state_probability" in label for label in labels)
    assert not any("shots=0" in label for label in labels)


def test_any_ibm_backend_is_an_ibm_qpu(parity):
    device = parity.ibm_device(IBM_TEST)

    assert (device.kind, device.vendor, device.backend_name) == ("qpu", "ibm", "ibm_test")
    assert (device.min_shots, device.max_shots) == (1, 100_000)
    assert device.is_ibm and not parity.DEVICES["braket/sv1"].is_ibm


def test_an_ibm_run_without_i_accept_cost_estimates_100_shot_cases_and_submits_nothing(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, capsys, quiet
):
    code = _ibm_main(parity)

    assert code == 3
    assert _submits(ibm_router) == []
    assert ibm_own_leg.calls == []
    sent = [(body["method"], body["shots"], body["device"]) for body in _estimates(ibm_router)]
    assert len(sent) == 9
    assert {method for method, _, _ in sent} == {"run", "sample", "probabilities"}
    assert {shots for _, shots, _ in sent} == {100}
    assert {device for _, _, device in sent} == {IBM_TEST}
    captured = capsys.readouterr()
    assert "--i-accept-cost" in captured.err
    assert "no router charge" in captured.out
    assert "quantum seconds of the IBM instance each leg runs on" in captured.out
    assert captured.out.index("no router charge") < captured.out.index("router estimate")


def test_the_ibm_own_leg_names_its_missing_variables_before_anything_is_sent(
    parity, router_env, ibm_router, no_ibm_env, ibm_own_leg, capsys
):
    code = _ibm_main(parity, "--i-accept-cost")

    assert code == 2
    err = capsys.readouterr().err
    assert "QISKIT_IBM_TOKEN" in err
    assert "IBM_INSTANCE_CRN" in err
    assert ibm_router.requests == []
    assert ibm_own_leg.calls == []


def test_the_ibm_own_leg_builds_the_kit_config_without_the_token(
    parity, monkeypatch, ibm_env, capsys
):
    from eigenkit.core.backends.qiskit_runtime_backend import (
        QiskitRuntimeConfig,
    )

    seen = {}

    def fake_run_leg(leg, cases, make_provider, *rest):
        seen["leg"] = leg
        seen["config"] = make_provider(100)
        return []

    monkeypatch.setattr(parity, "_run_leg", fake_run_leg)

    parity.run_own_leg([], parity.ibm_device(IBM_TEST))

    config = seen["config"]
    assert seen["leg"] == "own"
    assert isinstance(config, QiskitRuntimeConfig)
    assert (config.backend_name, config.shots, config.instance) == ("ibm_test", 100, IBM_CRN)
    assert config.token is None  # the kit reads QISKIT_IBM_TOKEN itself
    assert config.primitive == QiskitRuntimeConfig().primitive
    captured = capsys.readouterr()
    assert IBM_TOKEN not in captured.out + captured.err


def test_an_ibm_run_passes_when_every_router_job_ran_on_the_users_own_account(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, tmp_path, capsys, quiet
):
    report_path = tmp_path / "parity-ibm.json"

    code = _ibm_main(parity, "--i-accept-cost", "--report", str(report_path))

    assert code == 0
    [(own_cases, own_device)] = ibm_own_leg.calls
    assert own_device.id == IBM_TEST
    assert len(own_cases) == 9
    text = report_path.read_text()
    assert "eo_test_" not in text
    assert IBM_TOKEN not in text
    assert IBM_CRN not in text
    report = json.loads(text)
    assert report["passed"] is True
    assert report["expected_credential_source"] == "own"
    assert report["device_kind"] == "qpu"
    assert "QiskitRuntimeConfig" in report["reference"]
    assert report["own_device_arn"] is None
    for case in report["cases"]:
        assert case["credential_source"] == "own"
        assert case["router_job_id"] in ibm_router.jobs
        assert case["ibm_usage_seconds"] == 2.315
    captured = capsys.readouterr()
    assert IBM_TOKEN not in captured.out + captured.err


def test_an_ibm_run_fails_when_a_router_job_ran_on_an_ibm_account_eigenorbit_provides(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, tmp_path, quiet
):
    ibm_router.credential_source = "organization"
    report_path = tmp_path / "r.json"

    code = _ibm_main(
        parity,
        "--circuits",
        "ry",
        "--i-accept-cost",
        "--report",
        str(report_path),
    )

    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    for case in report["cases"]:
        assert case["passed"] is False
        assert case["credential_source"] == "organization"
        assert "credential_source 'organization', expected 'own'" in case["reason"]


def test_an_ibm_run_of_ansatz2_compares_run_sample_and_probabilities(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, tmp_path, quiet
):
    report_path = tmp_path / "ansatz2.json"

    code = _ibm_main(
        parity, "--circuits", "ansatz2", "--i-accept-cost", "--report", str(report_path)
    )

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["expected_credential_source"] == "own"
    assert [case["case"] for case in report["cases"]] == [
        "ansatz2/run/shots=100",
        "ansatz2/sample/shots=100",
        "ansatz2/probabilities/shots=100",
    ]
    sample = report["cases"][1]
    assert sum(sample["own_counts"].values()) == 100
    assert all(len(key) == 2 for key in sample["router_counts"])


def test_an_ibm_run_the_router_would_refuse_is_not_run(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, capsys, quiet
):
    ibm_router.ibm_refusal = "ibm_account_not_connected"

    code = _ibm_main(parity, "--i-accept-cost")

    assert code == 2
    assert _submits(ibm_router) == []
    assert ibm_own_leg.calls == []
    assert "ibm_account_not_connected" in capsys.readouterr().err


def test_router_only_on_ibm_needs_no_ibm_credentials(
    parity, router_env, ibm_router, no_ibm_env, monkeypatch, tmp_path, quiet
):
    def no_own_leg(*args, **kwargs):
        raise AssertionError("--router-only ran the own leg")

    monkeypatch.setattr(parity, "run_own_leg", no_own_leg)
    report_path = tmp_path / "ibm-router-only.json"

    code = _ibm_main(
        parity, "--router-only", "--i-accept-cost", "--report", str(report_path)
    )

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["mode"] == "router-only"
    assert len(report["cases"]) == 9
    assert report["submit_check"]["passed"] is True
    assert report["submit_check"]["credential_source"] == "own"


def test_the_ibm_qpu_rule_applies_to_probabilities(parity):
    [ansatz2] = parity.select_circuits("ansatz2")
    case = parity.Case(ansatz2, "probabilities", 20000)
    tol = parity.case_tolerance(case, 5.0, 2, kind="qpu")
    exact = np.asarray(parity.exact_value(case))

    assert tol.shape == exact.shape
    assert min(tol) >= 0.15
    near = parity.compare_distribution(exact + 0.12, exact, tol, case=case, exact=exact)
    far = parity.compare_distribution(exact + 0.2, exact, tol, case=case, exact=exact)
    assert near["passed"] is True
    assert far["passed"] is False


def _sample_case(parity, name="cnot3", shots=1000):
    [circuit] = parity.select_circuits(name)
    case = parity.Case(circuit, "sample", shots)
    exact = np.asarray(parity.exact_value(case))
    return case, exact, parity.case_tolerance(case, 5.0, 2, kind="qpu")


def _counts(exact, shots, n_qubits):
    from fake_router import largest_remainder_counts

    return largest_remainder_counts(exact, shots, n_qubits)


def test_matching_samples_pass_and_record_both_counts(parity):
    case, exact, tol = _sample_case(parity)
    counts = _counts(exact, 1000, 3)

    result = parity.compare_distribution(counts, dict(counts), tol, case=case, exact=exact)

    assert result["passed"] is True
    assert result["own_counts"] == counts
    assert result["shape"] == [8]
    assert result["dtype"] == "float64"


def test_a_sample_key_of_the_wrong_width_fails_exactly(parity):
    case, exact, tol = _sample_case(parity)
    counts = _counts(exact, 1000, 3)
    wide = {"0" + key: count for key, count in counts.items()}

    result = parity.compare_distribution(counts, wide, tol, case=case, exact=exact)

    assert result["passed"] is False
    assert "not 3 characters of 0/1" in result["reason"]


def test_a_sample_whose_shots_do_not_add_up_fails(parity):
    case, exact, tol = _sample_case(parity)
    counts = _counts(exact, 1000, 3)
    short = dict(counts)
    short[next(iter(short))] -= 1

    result = parity.compare_distribution(counts, short, tol, case=case, exact=exact)

    assert result["passed"] is False
    assert "not 1000 shots" in result["reason"]


def test_a_bit_reversed_sample_fails_on_the_bit_order(parity):
    case, exact, tol = _sample_case(parity)
    counts = _counts(exact, 1000, 3)
    reversed_ = {key[::-1]: count for key, count in counts.items()}

    result = parity.compare_distribution(counts, reversed_, tol, case=case, exact=exact)

    assert result["passed"] is False
    assert "router bit order" in result["reason"]


def test_a_bit_reversed_own_leg_fails_the_ibm_run(
    parity, router_env, ibm_router, ibm_env, ibm_own_leg, tmp_path, quiet
):
    ibm_own_leg.state["reverse_keys"] = True
    report_path = tmp_path / "r.json"

    code = _ibm_main(
        parity,
        "--circuits",
        "cnot3",
        "--i-accept-cost",
        "--report",
        str(report_path),
    )

    assert code == 1
    report = json.loads(report_path.read_text())
    [sample] = [case for case in report["cases"] if "/sample/" in case["case"]]
    assert "own bit order" in sample["reason"]


def test_sample_distributions_are_compared_loosely(parity):
    case, exact, tol = _sample_case(parity, shots=100)
    own = _counts(exact, 100, 3)
    keys = sorted(own, key=own.get)
    router = dict(own)
    router[keys[-1]] -= 5  # 0.05 of the shots moved: inside the 0.15 floor
    router[keys[0]] += 5

    result = parity.compare_distribution(own, router, tol, case=case, exact=exact)

    assert result["passed"] is True
    assert result["max_abs_diff"] == pytest.approx(0.05)


def test_the_ibm_token_is_redacted(parity, monkeypatch):
    monkeypatch.setenv("QISKIT_IBM_TOKEN", IBM_TOKEN)

    assert IBM_TOKEN not in parity._redact(f"IBM said: bad token {IBM_TOKEN}")
