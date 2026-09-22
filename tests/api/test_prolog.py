import numpy as np
import pytest
from wampy import Prolog, Query, Solution
from wampy import Prolog as ApiProlog
from wampy import Query as ApiQuery
from wampy import Solution as ApiSolution
from wampy.api.prolog import Prolog as DirectProlog
from wampy.api.query import Query as DirectQuery
from wampy.compiler.compiled_query import CompiledQuery
from wampy.frontend.solution import Solution as DirectSolution
from wampy.frontend.symbol_table import SymbolTable


def _sample_prolog() -> Prolog:
    return Prolog("""
        parent(anakin, luke).
        parent(padme, luke).
        parent(han, ben).
        """)


def test_api_class_import_paths():
    assert Prolog is ApiProlog is DirectProlog
    assert Query is ApiQuery is DirectQuery
    assert Solution is ApiSolution is DirectSolution

    assert Prolog.__module__ == "wampy.api.prolog"
    assert Query.__module__ == "wampy.api.query"


def test_query_once_returns_first_solution():
    prolog = _sample_prolog()

    assert prolog.query_once("parent(X, luke).") == {"X": "anakin"}


def test_prolog_uses_the_shared_symbol_table():
    prolog = Prolog("parent(alice, bob).")

    symbol_table = prolog.symbol_table

    assert isinstance(symbol_table, SymbolTable)

    assert isinstance(symbol_table.symbol_by_id, dict)
    assert isinstance(symbol_table.id_by_symbol, dict)

    assert symbol_table.symbol_by_id[48] == "alice"
    assert symbol_table.symbol_by_id[49] == "bob"
    assert symbol_table.symbol_by_id[50] == "parent"

    assert symbol_table.id_by_symbol["alice"] == 48
    assert symbol_table.id_by_symbol["bob"] == 49
    assert symbol_table.id_by_symbol["parent"] == 50


def test_query_once_returns_none_on_failure():
    prolog = _sample_prolog()

    assert prolog.query_once("parent(han, luke).") is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("parent(padme, luke).", True),
        ("parent(han, luke).", False),
    ],
)
def test_exists(query, expected):
    prolog = _sample_prolog()

    assert prolog.exists(query) is expected


def test_query_batch_exists():
    prolog = _sample_prolog()

    result = prolog.query_batch(
        [
            "parent(anakin, luke).",
            "parent(han, luke).",
            "parent(padme, luke).",
        ],
        mode="exists",
    )

    np.testing.assert_array_equal(
        result,
        np.array([True, False, True]),
    )


def test_query_batch_count():
    prolog = _sample_prolog()

    result = prolog.query_batch(
        [
            "parent(X, luke).",
            "parent(han, luke).",
        ],
        mode="count",
    )

    np.testing.assert_array_equal(result, np.array([2, 0]))


def test_prepared_queries_can_be_reused():
    prolog = _sample_prolog()
    prepared = prolog.prepare("parent(_, Y).")

    assert isinstance(prepared, CompiledQuery)
    assert prepared.variable_name_ids[1] == prolog.symbol_table.id_by_symbol["Y"]
    assert prepared.variable_registers[0] >= 0

    expected = [
        {"Y": "luke"},
        {"Y": "luke"},
        {"Y": "ben"},
    ]

    assert prolog.query_all(prepared) == expected
    assert prolog.query_all(prepared) == expected


@pytest.mark.xfail(
    reason="CompiledQuery ownership validation is not implemented yet",
    strict=True,
)
def test_compiled_query_cannot_be_used_with_another_prolog():
    first = Prolog("parent(anakin, luke).")
    second = Prolog("other(anakin, luke).")

    prepared = first.prepare("parent(X, luke).")

    with pytest.raises(ValueError, match="different Prolog"):
        second.query_all(prepared)


def test_parallel_batch_execution_is_explicitly_unimplemented():
    prolog = _sample_prolog()

    with pytest.raises(NotImplementedError, match="parallel"):
        prolog.query_batch(
            ["parent(anakin, luke)."],
            parallel=True,
        )


def test_grouped_negation_executes_all_goals():
    prolog = Prolog(
        r"""
        a.
        b.
        c.
        yes :- \+ (a, missing).
        no :- \+ (a, b).
        no_three :- \+ (a, b, c).
        """
    )

    assert prolog.query_all("yes.") == [{}]
    assert prolog.query_all("no.") == []
    assert prolog.query_all("no_three.") == []


def test_grouped_negation_preserves_variables_across_inner_calls():
    prolog = Prolog(
        r"""
        value(a).
        accept(a).
        good :- \+ (value(X), missing(X)).
        bad :- \+ (value(X), accept(X)).
        """
    )

    assert prolog.query_all("good.") == [{}]
    assert prolog.query_all("bad.") == []


def test_grouped_negation_supports_nested_naf_and_multiple_goals():
    prolog = Prolog(
        r"""
        a.
        ok :- \+ (missing, \+ a).
        """
    )

    assert prolog.query_all("ok.") == [{}]


def test_grouped_negation_rejects_inner_cut():
    with pytest.raises(ValueError, match="Cut inside grouped negation"):
        Prolog(r"a. bad :- \+ (a, !).")
