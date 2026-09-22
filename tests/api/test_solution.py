"""Tests for the public Solution API and import layout."""

from wampy import Prolog, Query, Solution
from wampy import Prolog as PackageProlog
from wampy import Query as PackageQuery
from wampy import Solution as PackageSolution
from wampy.api.prolog import Prolog as DirectProlog
from wampy.api.query import Query as DirectQuery
from wampy.frontend.solution import Solution as DirectSolution
from wampy.frontend.solution import decode_bindings


def test_api_package_and_direct_submodule_imports():
    """All supported import paths must expose the same public objects."""

    assert Prolog is PackageProlog is DirectProlog
    assert Query is PackageQuery is DirectQuery
    assert Solution is PackageSolution is DirectSolution


def test_public_classes_are_defined_in_their_expected_modules():
    """The package split must not leave classes defined in the old api.py module."""

    assert Prolog.__module__ == "wampy.api.prolog"
    assert Query.__module__ == "wampy.api.query"


def test_solution_remains_the_builtin_dict_type():
    """Preserve the current public result representation."""

    assert Solution is dict

    solution = Solution(X="anakin", Y="luke")

    assert solution == {"X": "anakin", "Y": "luke"}
    assert isinstance(solution, dict)


def test_query_returns_solution_dictionary():
    """A successful non-ground query must yield a Solution-compatible mapping."""

    prolog = Prolog("parent(anakin, luke).")

    solution = prolog.query_once("parent(X, luke).")

    assert solution == {"X": "anakin"}
    assert isinstance(solution, Solution)


def test_ground_query_returns_empty_solution_dictionary():
    """A successful ground query has one solution with no variable bindings."""

    prolog = Prolog("parent(anakin, luke).")

    solution = prolog.query_once("parent(anakin, luke).")

    assert solution == {}
    assert isinstance(solution, Solution)


def test_decode_bindings_reads_current_machine_bindings():
    prolog = Prolog("parent(anakin, luke).")
    prepared = prolog.prepare("parent(X, luke).")

    with prolog.query(prepared) as query:
        next(query)
        assert query._machine is not None
        assert decode_bindings(query._machine, prepared, prolog.symbol_table) == {
            "X": "anakin",
        }


def test_decode_bindings_returns_empty_for_successful_ground_query():
    prolog = Prolog("parent(anakin, luke).")
    prepared = prolog.prepare("parent(anakin, luke).")

    with prolog.query(prepared) as query:
        next(query)
        assert query._machine is not None
        assert decode_bindings(query._machine, prepared, prolog.symbol_table) == {}


def test_each_backtracking_result_is_an_independent_snapshot():
    """Later backtracking must not mutate a previously returned solution."""

    prolog = Prolog("""
        value(a).
        value(b).
        """)

    with prolog.query("value(X).") as query:
        first = next(query)
        second = next(query)

    assert first == {"X": "a"}
    assert second == {"X": "b"}
    assert first is not second

    first["X"] = "changed"

    assert first == {"X": "changed"}
    assert second == {"X": "b"}


def test_mutating_a_solution_does_not_affect_future_queries():
    """Returned dictionaries must not retain references to mutable WAM state."""

    prolog = Prolog("value(a).")
    first = prolog.query_once("value(X).")

    assert first is not None
    first["X"] = "changed"

    second = prolog.query_once("value(X).")

    assert second == {"X": "a"}
    assert second is not first
