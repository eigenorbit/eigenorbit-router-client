# eigenorbit-router-client

**Run eigenkit circuits on real quantum hardware.**

eigenorbit-router-client lets [eigenkit](https://pypi.org/project/eigenkit/)
users run circuits on real quantum hardware through EigenOrbit Router, with one
EigenOrbit API key instead of separate vendor credentials.

You write the same eigenkit code as with your own vendor credentials and pass
`provider=RouterConfig(...)`. The router runs the circuit on an Amazon Braket
simulator or QPU, or on an IBM Quantum device. Results have the same shape,
dtype and bit ordering as the same code on the vendor directly. Braket jobs
are billed to your prepaid EigenOrbit balance. IBM jobs run on your own IBM
Quantum account and carry no router charge.

## Install

```bash
pip install "eigenkit[router]"
```

The `router` extra of eigenkit installs this client.

## API key

1. Sign in to EigenOrbit Router at https://router.eigenorbit.com and accept
   the router terms.
2. Add funds on the Balance page if you run Braket jobs.
3. Create an API key on the Keys page. The key is shown once; copy it then.

```bash
export EIGENORBIT_ROUTER_API_KEY=eo_live_...
```

| Variable | Used when |
|----------|-----------|
| `EIGENORBIT_ROUTER_API_KEY` | `RouterConfig(api_key=...)` is None |
| `EIGENORBIT_ROUTER_URL` | `RouterConfig(base_url=...)` is None; unset means https://router.eigenorbit.com |

The router URL must be `https://` (plain `http://` only for localhost). The
key is never written to a pickle or a saved model, and never appears in `repr`.

## Quickstart

```python
import numpy as np
from eigenkit import Circuit
from eigenkit.core.blocks import (
    AngleEncoding, BasicVariational, assemble,
)
from eigenkit.core.observables import all_z
from eigenorbit_router_client import RouterConfig

spec = assemble([
    AngleEncoding(in_features=2),
    BasicVariational(depth=1),
])

# The API key is read from EIGENORBIT_ROUTER_API_KEY.
# On a real QPU: device="braket/iqm_garnet" (see Devices)
provider = RouterConfig(device="braket/sv1", shots=1000)
circuit = Circuit(spec, all_z(2), provider=provider)

weights = np.linspace(-0.9, 1.1, spec.n_weights)
inputs = np.array([0.2, 0.6])
values = circuit(weights, inputs)  # one router job
print(values)
```

`values` holds one expectation value per observable, as on any other eigenkit
provider. `sample(...)` and `probabilities(...)` work as well, and a 2-D batch
runs as one router job per row.

## Devices

A device ID is `<vendor>/<device>`, for example `braket/sv1`, `braket/dm1`,
`braket/iqm_garnet` or `ibm/ibm_fez`. The catalog at
https://router.eigenorbit.com/devices (no sign-in needed) lists every device
with its status, availability windows, shot ranges and price.

## Cost

Building a circuit asks the router for an estimate and shows one
`CostWarning` with the price of a job: the vendor's cost plus the EigenOrbit
fee. `circuit.dry_run_info()` returns the same estimate without sending
anything. Every job holds its estimate on your balance first, so the router
never runs a job your balance cannot cover. Each row of a batch and each
parameter-shift gradient evaluation is a separate paid job.

## Jobs

Every call blocks until its job finishes. To come back to a job later:

```python
handle = circuit.submit(weights, inputs)
print(handle.job_id(), handle.status())
values = handle.result()
```

`handle.cancel()` asks the router to cancel the job. A job still queued at
the router is cancelled at once; a job at the vendor ends the way the vendor
decides. The job ID is also on the router web app's Jobs page.

## Errors

Every router failure raises `eigenorbit_router_client.RouterError` with
`code`, `message`, `details` and `http_status`. Branch on `code`, never on the
message. The most common codes:

| Code | What to do |
|------|------------|
| `unauthenticated` | Check `EIGENORBIT_ROUTER_API_KEY`. |
| `terms_not_accepted` | Accept the current terms at `details["accept_url"]`. |
| `insufficient_balance` | Add funds on the Balance page. |
| `device_unavailable` | The device is offline; pick another or retry later. |
| `shots_invalid` | Use shots within `details["min_shots"]..details["max_shots"]`. |
| `router_unreachable` | No answer after the retries; see below. |

The client retries 502, 503, 504 and dropped connections with the same
`Idempotency-Key`, so a retry never creates a second job. If every attempt
loses its answer, `router_unreachable` carries `details["idempotency_key"]`:
the job may already exist. Pass that key as
`circuit.compiled.submit(..., idempotency_key=...)` to get the same job back
instead of a second, separately billed one.

## IBM Quantum

IBM devices are `ibm/<backend>` and run on your own IBM Quantum account.
Connect it once on the router's IBM account page
(https://router.eigenorbit.com/ibm) with your IBM API key and instance CRN.
IBM jobs are free through the router; IBM bills your instance for the quantum
time they use.

## Saved models

A model whose provider is a `RouterConfig` saves the device and the settings,
never the API key. Pass the router config again when you load it:

```python
from eigenkit import QSVC
from eigenorbit_router_client import RouterConfig

model = QSVC.load("model.eigenkit", provider=RouterConfig(device="braket/sv1"))
```

## Contributing

Development, the contract sync, releases and the live checks are described in
[CONTRIBUTING.md](https://github.com/eigenorbit/eigenorbit-router-client/blob/main/CONTRIBUTING.md).

## Support

Write to support@eigenorbit.com. Include the router job ID for a question
about a job.

## License

eigenorbit-router-client is licensed under the Apache License 2.0.
