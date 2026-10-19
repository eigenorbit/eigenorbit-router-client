"""The router API key never reaches a saved artifact.

A `.eigenkit` bundle is a deflated zip, so a key written to `params.json` would not
show up in the raw file bytes. `_bundle_texts` returns the raw bytes and every
decompressed member, and a positive control (the device ID, which must be
found inside a member) keeps the negative scan from passing vacuously.

Pickle and joblib (checkpoints, process pools) drop the key through the kit's
SecretFieldsMixin; copies inside one process (`copy.deepcopy`,
`sklearn.base.clone`) keep it so a grid search still authenticates.
"""

import copy
import io
import pickle
import warnings
import zipfile

import numpy as np
import pytest


pytest.importorskip("sklearn")
import eigenkit.configs  # noqa: E402
import joblib  # noqa: E402
from eigenkit import QSVC  # noqa: E402
from sklearn.base import clone  # noqa: E402

from eigenorbit_router_client import RouterConfig  # noqa: E402
from fake_router import VALID_KEY  # noqa: E402


KEY = VALID_KEY
DEVICE = "braket/sv1"
BASE_URL = "https://router.test"


def _config() -> RouterConfig:
    return RouterConfig(device=DEVICE, api_key=KEY, base_url=BASE_URL)


def _config_without_key() -> RouterConfig:
    return RouterConfig(device=DEVICE, base_url=BASE_URL)


def _eigenkit_rebuilds_router_configs() -> bool:
    """Whether this eigenkit can rebuild a RouterConfig saved in a bundle."""
    return "RouterConfig" in getattr(eigenkit.configs, "_EXTERNAL_CONFIGS", {})


def _bundle_texts(path) -> list[bytes]:
    """The raw file bytes, then the decompressed bytes of every zip member."""
    with open(path, "rb") as fh:
        texts = [fh.read()]
    with zipfile.ZipFile(path) as zf:
        texts.extend(zf.read(name) for name in zf.namelist())
    return texts


def _saved_qsvc(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 2))
    y = np.array([0, 1] * 4)
    model = QSVC(feature_map="Angle", reps=1, provider="numpy").fit(X, y)
    model.set_params(provider=_config())
    path = tmp_path / "m.eigenkit"
    model.save(path)
    return path


class _Holder:
    """Stands in for a model or checkpoint holding a provider config."""

    def __init__(self, provider):
        self.provider = provider


def test_the_eigenkit_bundle_carries_no_key(tmp_path):
    texts = _bundle_texts(_saved_qsvc(tmp_path))

    assert all(KEY.encode() not in t for t in texts)
    assert all(b"eo_test_" not in t for t in texts)
    assert all(b"eo_live_" not in t for t in texts)
    # Positive control: the scan does read inside the deflated members.
    assert any(DEVICE.encode() in t for t in texts[1:])


def test_the_eigenkit_bundle_loads_with_an_explicit_router_config(tmp_path, fake_router):
    path = _saved_qsvc(tmp_path)

    # Loading rebuilds the kernel, which compiles against the router: the
    # estimate goes to the in-process double, and without a key it is
    # reported as unavailable in a CostWarning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = QSVC.load(path, provider=_config_without_key())

    assert isinstance(loaded.provider, RouterConfig)
    assert loaded.provider.api_key is None
    assert loaded.provider.device == DEVICE


@pytest.mark.skipif(
    not _eigenkit_rebuilds_router_configs(),
    reason="this eigenkit cannot rebuild a RouterConfig from a bundle; pass provider= to load()",
)
def test_the_eigenkit_bundle_rebuilds_its_router_config_without_key_or_url(
    tmp_path, fake_router
):
    path = _saved_qsvc(tmp_path)

    # A bundle never chooses where the loader's key goes, so the URL comes back
    # unset along with the key.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = QSVC.load(path)

    assert isinstance(loaded.provider, RouterConfig)
    assert loaded.provider.api_key is None
    assert loaded.provider.base_url is None
    assert loaded.provider.device == DEVICE


def test_pickle_drops_the_key_and_loads_it_back_as_none():
    cfg = _config()

    for obj in (cfg, _Holder(cfg)):
        data = pickle.dumps(obj)
        assert KEY.encode() not in data
        assert b"eo_test_" not in data
        assert b"eo_live_" not in data
        assert DEVICE.encode() in data

    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.api_key is None
    assert restored.device == DEVICE
    assert restored.base_url == BASE_URL
    assert pickle.loads(pickle.dumps(_Holder(cfg))).provider.api_key is None


def test_joblib_drops_the_key():
    buf = io.BytesIO()
    joblib.dump({"cfg": _config()}, buf)

    assert KEY.encode() not in buf.getvalue()
    assert DEVICE.encode() in buf.getvalue()


def test_deepcopy_and_sklearn_clone_keep_the_key():
    cfg = _config()

    assert copy.deepcopy(cfg).api_key == KEY
    assert copy.copy(cfg).api_key == KEY
    assert clone(QSVC(provider=cfg)).provider.api_key == KEY


def test_repr_and_str_never_show_the_key():
    cfg = _config()

    for text in (repr(cfg), str(cfg)):
        assert KEY not in text
        assert "eo_test_" not in text
