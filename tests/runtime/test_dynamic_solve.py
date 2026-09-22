import pytest
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table

from ..plunit.helpers import solve_from_str_ontop


def _static_context(static_src: str, config=DEFAULT_CONFIG):
    static_program, symbol_table = parse(
        static_src, empty_symbol_table(config.frontend.ast), config
    )
    compiler_state = init_compiler_state(config)
    compiled_program = init_compiled_program(config)
    compile_program(
        static_program,
        compiled_program,
        compiler_state,
        config,
    )
    return compiled_program, compiler_state, symbol_table


def _solve_ontop(
    static_src: str,
    dynamic_src: str,
    query_src: str,
    config=DEFAULT_CONFIG,
):
    static_compiled_program, static_state, symbol_table = _static_context(static_src, config)
    return solve_from_str_ontop(
        dynamic_src,
        query_src,
        static_compiled_program,
        static_state,
        symbol_table,
        config,
    )


def _assert_answers(result, expected) -> None:
    assert result == expected


def test_dynamic_answers_are_returned_before_static_fallback() -> None:
    result = _solve_ontop(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        """
        likes(john, pasta).
        likes(john, sushi).
        """,
        "likes(john, X).",
    )

    _assert_answers(
        result,
        [
            {"X": "pasta"},
            {"X": "sushi"},
            {"X": "pizza"},
        ],
    )


def test_failed_dynamic_same_predicate_clauses_still_fall_through_to_static():
    result = _solve_ontop(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        """
        likes(mary, pasta).
        likes(mary, pizza).
        """,
        "likes(john, X).",
    )

    _assert_answers(result, [{"X": "pizza"}])


def test_dynamic_program_can_add_new_symbol_and_then_fall_back_to_static() -> None:
    result = _solve_ontop(
        """
        likes(john, pizza).
        """,
        """
        likes(peter, pizza).
        """,
        "likes(X, pizza).",
    )

    _assert_answers(
        result,
        [
            {"X": "peter"},
            {"X": "john"},
        ],
    )


def test_dynamic_program_can_add_new_predicate():
    result = _solve_ontop(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        """
        dislikes(peter, pizza).
        dislikes(peter, pasta).
        """,
        "dislikes(peter, X).",
    )

    _assert_answers(
        result,
        [
            {"X": "pizza"},
            {"X": "pasta"},
        ],
    )


def test_single_static_clause_fallback_returns_dynamic_then_static_answer() -> None:
    result = _solve_ontop(
        """
        likes(john, pizza).
        """,
        """
        likes(john, sushi).
        """,
        "likes(john, X).",
    )

    _assert_answers(
        result,
        [
            {"X": "sushi"},
            {"X": "pizza"},
        ],
    )


def test_dynamic_rule_can_call_static_predicate_and_fall_back_to_static_rule() -> None:
    answers = _solve_ontop(
        """
        father(ted, bob).
        mother(jane, bob).
        parent(X, Y) :- father(X, Y).
        """,
        """
        parent(X, Y) :- mother(X, Y).
        """,
        "parent(X, bob).",
    )

    assert answers == [
        {"X": "jane"},
        {"X": "ted"},
    ]


def test_second_dynamic_compilation_does_not_leak_first_dynamic_program() -> None:
    static_compiled_program, static_state, symbol_table = _static_context(
        """
        likes(john, pizza).
        """
    )

    first = solve_from_str_ontop(
        """
        likes(peter, pizza).
        """,
        "likes(X, pizza).",
        static_compiled_program,
        static_state,
        symbol_table,
        DEFAULT_CONFIG,
    )

    _assert_answers(
        first,
        [
            {"X": "peter"},
            {"X": "john"},
        ],
    )

    second = solve_from_str_ontop(
        """
        dislikes(peter, pasta).
        """,
        "likes(X, pizza).",
        static_compiled_program,
        static_state,
        symbol_table,
    )

    _assert_answers(second, [{"X": "john"}])


def test_replacing_extension_invalidates_old_extension_only_predicates() -> None:
    static_compiled_program, static_state, symbol_table = _static_context("likes(john, pizza).")

    first = solve_from_str_ontop(
        "dislikes(peter, pasta).",
        "dislikes(peter, pasta).",
        static_compiled_program,
        static_state,
        symbol_table,
    )

    _assert_answers(first, [{}])

    with pytest.raises(RuntimeError, match="INVALID_PC"):
        solve_from_str_ontop(
            "likes(peter, pizza).",
            "dislikes(peter, pasta).",
            static_compiled_program,
            static_state,
            symbol_table,
        )
