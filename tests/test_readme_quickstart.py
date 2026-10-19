"""README.md: its quickstart imports, its install line and its wording.

The README is the package's PyPI page and first stop for users. Its first
Python block is the router web app's quickstart, so every name it imports must
exist in the installed eigenkit and this client: a renamed module or class
breaks this test instead of a user's first run. Nothing here runs the example
or touches the network.
"""

import ast
import importlib
import re
from pathlib import Path

import pytest


README = Path(__file__).resolve().parent.parent / "README.md"

INSTALL_LINE = 'pip install "eigenkit[router]"'

_PYTHON_BLOCK = re.compile(r"^```python\n(?P<body>.*?)^```", re.DOTALL | re.MULTILINE)


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _python_blocks() -> list[str]:
    blocks = [m.group("body") for m in _PYTHON_BLOCK.finditer(_text())]
    assert blocks, "README.md has no ```python block"
    return blocks


def _imports(source: str) -> list[tuple[str, list[str]]]:
    """Every `from <module> import a, b` in `source`."""
    return [
        (node.module, [alias.name for alias in node.names])
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]


def _assert_imports_resolve(source: str) -> None:
    for module_name, names in _imports(source):
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name} has no {name}"


def test_the_quickstart_is_valid_python():
    ast.parse(_python_blocks()[0])


def test_the_quickstart_imports_exist_in_eigenkit_and_the_client():
    quickstart = _python_blocks()[0]
    modules = {module for module, _ in _imports(quickstart)}

    assert "eigenorbit_router_client" in modules
    assert any(module == "eigenkit" or module.startswith("eigenkit.") for module in modules)
    _assert_imports_resolve(quickstart)


def test_the_quickstart_runs_a_router_config_on_a_device():
    quickstart = _python_blocks()[0]

    assert "from eigenorbit_router_client import RouterConfig" in quickstart
    assert 'RouterConfig(device="braket/sv1", shots=1000)' in quickstart
    assert "provider=provider" in quickstart


def test_every_python_block_parses_and_its_imports_exist():
    for source in _python_blocks():
        _assert_imports_resolve(source)


def test_every_install_line_is_the_eigenkit_router_extra():
    lines = [line.strip() for line in _text().splitlines() if "pip install" in line]

    assert lines, "README.md has no install line"
    assert all(line == INSTALL_LINE for line in lines), lines


def test_the_readme_names_no_other_package_index():
    assert "index-url" not in _text()


@pytest.mark.parametrize("word", ["staff", "pooled"])
def test_the_readme_says_nothing_about_shared_ibm_access(word):
    assert re.search(rf"\b{word}\b", _text(), re.IGNORECASE) is None


@pytest.mark.parametrize(
    "name",
    [
        "EIGENORBIT_ROUTER_API_KEY",
        "EIGENORBIT_ROUTER_URL",
        "https://router.eigenorbit.com",
        "support@eigenorbit.com",
    ],
)
def test_the_readme_names_the_settings_and_the_support_address(name):
    assert name in _text()
