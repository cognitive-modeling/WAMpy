"""Tests for the public WAMpy package facades."""

import wampy
import wampy.api as api_package


def test_root_package_declares_high_level_public_exports():
    """The root package should expose the high-level object-oriented API."""

    exported_names = set(wampy.__all__)

    assert {
        "Prolog",
        "Query",
        "Solution",
    } <= exported_names

    assert "display" not in exported_names
    assert not hasattr(wampy, "display")


def test_api_package_declares_core_public_exports():
    """The api package should expose the functional research API."""

    exported_names = set(api_package.__all__)

    assert {
        "ast",
        "clear_compiled_program",
        "compile_program",
        "decode_bindings",
        "decode_term",
        "decode_variables",
        "undo_hypothesis",
    } <= exported_names

    # does not contain Python
    assert {
        "Prolog",
        "Query",
        "Solution",
    }.isdisjoint(exported_names)
