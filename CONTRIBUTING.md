# Contributing to eigenorbit-router-client

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

The suite is offline: sockets are refused and the router is replaced by
`tests/fake_router.py`. The dev dependency group installs eigenkit from PyPI;
the `live` group adds its `braket` and `qiskit` extras for the live scripts
below.

## Contract sync

`tests/contract/` vendors the router's wire contract v1, pinned by
`tests/contract/SOURCE.json`; never edit it by hand. After a router contract
change:

```bash
uv run python scripts/sync_contract.py --router <router checkout> [--ref <commit>]
```

`--check` compares without writing. CI runs `scripts/sync_contract.py --verify`.

## Releasing

Releases run from GitHub Actions: open Actions, pick the `release` workflow
and run it on `main`. It re-runs the CI workflow first. Then
python-semantic-release picks the next version from the conventional commits
since the last release (`feat:` minor, `fix:` patch), tags it, updates
`CHANGELOG.md` and creates the GitHub release, and a separate job publishes
the wheel and the sdist to PyPI through PyPI's trusted publisher. Pull
requests are squash-merged, so a pull request title is a conventional commit.

## Live parity

`scripts/live_parity.py` runs the same eigenkit circuits on a vendor device
with your own credentials and on the same device through the router, and
exits 0 only when every case passes. A router worker moves to a new eigenkit
version only after a live parity run of this script with that version on both
legs.

It needs network access to the router, `EIGENORBIT_ROUTER_API_KEY` and
`EIGENORBIT_ROUTER_URL` (required here, never defaulted), plus the vendor
credentials of the own leg:

```bash
uv run --group live python scripts/live_parity.py --report parity.json
```

Options:

- `--device`: the router device (default `braket/sv1`; also `braket/dm1`,
  `braket/iqm_garnet`, `braket/rigetti_cepheus_1_108q` and `ibm/<backend>`).
  On Braket the own leg runs on the device's ARN in its region (us-east-1 for
  the simulators, eu-north-1 for Garnet, us-west-1 for Cepheus) with an
  `AWS_PROFILE` that has Braket access there.
- `--circuits {all,garnet,ry,ansatz2,cnot3,mc3}`: `all` (the default) is
  `ry`, `ansatz2` and `cnot3`; `garnet` is `ansatz2` and `mc3`, a 3-qubit
  circuit with MCZ, MCX and CRY, for IQM Garnet.
- `--shots N` (inside the device's range) and `--k K` (the shot-noise sigmas
  allowed, default 5).
- `--router-only`: no vendor credentials; compares the router with the exact
  local reference and drives one `submit()` handle.
- `--i-accept-cost`: required for a QPU or IBM run.
- `--report PATH`: writes a JSON report. The output and the report never
  contain the API key, the IBM token or the instance CRN.

Exit status: 0 every case passes, 1 a case fails, 2 a usage or configuration
error, 3 a QPU or IBM run without `--i-accept-cost`.

On a simulator the cases are `run` at shots 0 and N and
`zero_state_probability` at N. Shapes and dtypes must match exactly, values
within 1e-9 at shots 0 and within k shot-noise sigmas at N. A full SV1 run
needs about $0.81 of balance.

A QPU has no exact mode, so a QPU run has only the sampled cases (`run` and
`zero_state_probability` at N). Shapes, dtypes and bit ordering must match
exactly, and each value must lie within max(k x shot-noise sigma,
`QPU_ABS_TOL`), where `QPU_ABS_TOL` is 0.15 for expectation values and
probabilities (hardware drift and readout error between two QPU runs). The
report records each case's tolerance and difference.

A QPU run costs real money on both legs. The script first asks the router for
the estimate of every job and prints the total (vendor cost plus EigenOrbit
fee) and your balance. It submits nothing without `--i-accept-cost` (exit 3),
and nothing if the router would refuse a job or your balance does not cover
the total (exit 2). Run it without `--i-accept-cost` first to see the
estimate:

```bash
export AWS_PROFILE=<your profile with Braket access in eu-north-1>
uv run --group live python scripts/live_parity.py --device braket/iqm_garnet \
  --circuits garnet --shots 100 --i-accept-cost --report parity-garnet.json
```

The `garnet` set at 100 shots is 4 tasks on each leg, about $4 for both legs
at the `braket/iqm_garnet@1` price. The printed estimate is the router leg's
exact figure.

`--device ibm/<backend>` compares your own IBM credentials with the same
device through the router. The own leg is
`QiskitRuntimeConfig(backend_name=<backend>, shots=N, instance=$IBM_INSTANCE_CRN)`,
with the token read by eigenkit from `QISKIT_IBM_TOKEN`. The router leg uses
the router account behind `EIGENORBIT_ROUTER_API_KEY`, which has the same
token and instance connected on its IBM account page; every router job must
report `credential_source` `"own"`. It runs `run`, `sample` and
`probabilities` at N shots (default 100) under the QPU rule above. IBM jobs
carry no router charge, but both legs spend the instance's quantum time, so
an IBM run also needs `--i-accept-cost`:

```bash
export QISKIT_IBM_TOKEN=<your IBM token>
export IBM_INSTANCE_CRN=<your instance CRN>
uv run --group live python scripts/live_parity.py --device ibm/ibm_fez \
  --circuits ansatz2 --i-accept-cost --report parity-ibm.json
```

## Live cancel

`scripts/live_cancel.py` submits one long circuit on `braket/sv1` at 100 shots
through the router, waits until the job is `running`, calls `handle.cancel()`
and polls it to a terminal status. It prints the job ID, the final status and
the router's billing, and exits 0 when the job ended `cancelled`, or
`completed` because the vendor finished first. It creates one paid SV1 task
(about $0.09 held, a few cents charged at most) and needs
`EIGENORBIT_ROUTER_API_KEY` and `EIGENORBIT_ROUTER_URL`:

```bash
uv run python scripts/live_cancel.py --report cancel.json
```

`--device braket/iqm_garnet --shots 10` cancels a small QPU job while it
waits in the vendor queue (up to $0.38 held). Other options: `--qubits`,
`--layers`, `--cancel-at {queued,running}`, `--wait-seconds`, `--timeout`.
