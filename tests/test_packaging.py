"""Packaging guards: the eigenkit dependency, package metadata and release config."""

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
WORKFLOWS = ROOT / ".github" / "workflows"

# Keys the release config never carries: semantic-release builds and tags,
# and the release workflow's pypi job is the only publisher.
DROPPED_RELEASE_KEYS = (
    "version_source",
    "commit_version_number",
    "branch",
    "changelog_file",
    "dist_path",
    "upload_to_release",
    "upload_to_pypi",
    "remove_dist",
    "build_command_env",
)


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _all_requirements() -> list[str]:
    """Runtime dependencies plus every dependency group."""
    data = _pyproject()
    requirements = list(data["project"]["dependencies"])
    for group in data.get("dependency-groups", {}).values():
        requirements.extend(r for r in group if isinstance(r, str))
    return requirements


def test_eigenkit_is_not_a_uv_source():
    # A uv source leaks into dependents and breaks `uv add "eigenkit[router]"`.
    sources = _pyproject().get("tool", {}).get("uv", {}).get("sources", {})
    assert "eigenkit" not in sources


def test_runtime_eigenkit_requirement_is_one_pypi_range():
    # The client uses eigenkit internals, so a new eigenkit major must not
    # install under it.
    deps = _pyproject()["project"]["dependencies"]
    kit = [d for d in deps if d.startswith("eigenkit")]
    assert kit == ["eigenkit>=1.0.1,<2"]


def test_no_requirement_comes_from_git_or_ssh():
    # Every requirement resolves from PyPI by name, so a clean install needs no
    # repository access; a direct reference (`name @ url`) would.
    offending = [r for r in _all_requirements() if "@" in r or "://" in r]
    assert offending == []


def test_the_dev_group_pins_the_eigenkit_the_router_worker_runs():
    assert "eigenkit[datasets]==1.0.1" in _pyproject()["dependency-groups"]["dev"]


def test_the_live_group_pins_the_eigenkit_the_router_worker_runs():
    assert _pyproject()["dependency-groups"]["live"] == ["eigenkit[braket,qiskit]==1.0.1"]


def test_client_error_codes_is_a_package_export():
    # Callers tell client-side failures from router codes with it.
    import eigenorbit_router_client as package
    from eigenorbit_router_client import CLIENT_ERROR_CODES

    assert "CLIENT_ERROR_CODES" in package.__all__
    assert CLIENT_ERROR_CODES == {"invalid_response", "poll_timeout", "router_unreachable"}


def test_the_license_is_the_apache_expression_with_its_file():
    project = _pyproject()["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]


def test_the_license_file_is_the_apache_2_text():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.lstrip().startswith("Apache License")
    assert "Version 2.0" in text


def test_semantic_release_cuts_0_1_0_from_the_first_feat_commit():
    config = _pyproject()["tool"]["semantic_release"]
    assert config["version_toml"] == ["pyproject.toml:project.version"]
    assert config["allow_zero_version"] is True
    assert config["major_on_zero"] is False
    assert config["assets"] == ["uv.lock"]


def test_semantic_release_carries_no_publish_or_environment_keys():
    config = _pyproject()["tool"]["semantic_release"]
    assert [k for k in DROPPED_RELEASE_KEYS if k in config] == []


def test_the_changelog_keeps_the_insertion_flag():
    # semantic-release only updates a changelog that carries the flag.
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "<!-- version list -->" in text


def test_the_package_ships_its_type_marker():
    assert (ROOT / "eigenorbit_router_client" / "py.typed").is_file()


# The workflows are read as text: the dev group carries no YAML parser.
def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ci_runs_on_pull_requests_and_as_the_release_gate():
    ci = _workflow("ci.yml")
    assert "pull_request:" in ci
    assert "workflow_call:" in ci


def test_ci_runs_the_locked_offline_suite_on_every_supported_python():
    ci = _workflow("ci.yml")
    assert "uv sync --locked" in ci
    assert "sync_contract.py --verify" in ci
    assert 'python-version: ["3.12", "3.13", "3.14"]' in ci


def test_ci_installs_the_built_wheel_into_a_clean_environment():
    ci = _workflow("ci.yml")
    assert "uv build" in ci
    assert "uv venv --python 3.12" in ci
    assert "-I -c" in ci


def test_ci_needs_no_secret_and_no_ssh_agent():
    # Pull requests from forks run it, so it reaches nothing private.
    ci = _workflow("ci.yml")
    assert "secrets." not in ci
    assert "ssh" not in ci.lower()


def test_release_is_manual_and_gated_by_ci():
    release = _workflow("release.yml")
    assert "workflow_dispatch:" in release
    assert "uses: ./.github/workflows/ci.yml" in release
    assert "push:" not in release
    assert "secrets: inherit" not in release


def test_release_publishes_through_the_pypi_trusted_publisher():
    release = _workflow("release.yml")
    assert "environment: pypi" in release
    assert "pypa/gh-action-pypi-publish@" in release
    assert "vars.RELEASE_BOT_CLIENT_ID" in release


def test_release_grants_each_write_permission_once():
    release = _workflow("release.yml")
    assert release.count("id-token: write") == 1
    # Only the release-bot token writes, and only to contents.
    assert release.count("contents: write") == 1
    assert "permission-contents: write" in release


def test_every_action_is_pinned_to_a_commit():
    # A tag or a branch can move; these jobs hold the release token and
    # id-token: write.
    for name in ("ci.yml", "release.yml"):
        for ref in re.findall(r"uses:\s*([^\s#]+)", _workflow(name)):
            if not ref.startswith("./"):
                assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), (name, ref)


def test_no_checkout_keeps_its_credentials():
    for name in ("ci.yml", "release.yml"):
        text = _workflow(name)
        assert text.count("actions/checkout@") == text.count("persist-credentials: false"), name
