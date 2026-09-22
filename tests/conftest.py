"""Pytest configuration for the WAMpy test suite."""

import os
from pathlib import Path

import pytest

from .plunit.plunit_collector import PlUnitTestModule
from .prolog.logtalk_collector import LogtalkTestFile

pytest_plugins = ("pytester",)

_TEST_QUERY_PATH = Path(__file__).resolve().parent / "api" / "test_query_legacy.py"
_PROLOG_TEST_ROOT = Path(__file__).resolve().parent / "prolog"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip dispatcher-specific tests when Numba execution is disabled."""
    if os.environ.get("NUMBA_DISABLE_JIT") != "1":
        return

    skip_jit = pytest.mark.skip(reason="requires enabled Numba JIT execution")
    for item in items:
        if "requires_numba_jit" in item.keywords:
            item.add_marker(skip_jit)


def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    """Collect each vendored Logtalk ``tests.lgt`` as a native pytest file."""

    try:
        file_path.resolve().relative_to(_PROLOG_TEST_ROOT)
    except ValueError:
        return None

    if file_path.name != "tests.lgt":
        return None

    return LogtalkTestFile.from_parent(
        parent,
        path=file_path,
        prolog_root=_PROLOG_TEST_ROOT,
    )


def pytest_pycollect_makemodule(
    module_path: Path,
    parent: pytest.Collector,
) -> pytest.Module | None:
    """Use the custom module collector only for the legacy query behavior module."""

    if module_path.resolve() != _TEST_QUERY_PATH:
        return None

    return PlUnitTestModule.from_parent(
        parent,
        path=module_path,
    )
