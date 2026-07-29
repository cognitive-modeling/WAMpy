import pytest

from wampy.compiler.compiler import compile, init_compiler
from wampy.config import load_config_toml
from wampy.frontend.parser import build_new_program
from wampy.status import WAMStatus
from ..test_engine import solve_from_str_ontop

CONFIG_TOML = """
[ast]
num_terms = 80
max_terms_nodes = 240
max_terms_nodes_childs = 5

[entry]
max_functors = 256
max_arity = 8

[solver]
max_steps = 10000
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _static_context(static_src: str, config):
    static_program, symbol_table = build_new_program(static_src, config)
    static_state = compile(static_program, init_compiler(config), config)
    return static_state, static_state.entry.copy(), int(static_state.pc.value), symbol_table


def _solve_ontop(static_src: str, dynamic_src: str, query_src: str, config):
    static_state, static_entry, static_size, symbol_table = _static_context(static_src, config)
    return solve_from_str_ontop(
        dynamic_src,
        query_src,
        static_state,
        static_entry,
        static_size,
        symbol_table,
        config,
    )


def _assert_answers(result, expected):
    assert result["answers"] == expected
    assert result["n_answers"] == len(expected)
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED


def test_dynamic_answers_are_returned_before_static_fallback(wam_config):
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
        wam_config,
    )

    _assert_answers(
        result,
        [
            "likes(john, pasta).",
            "likes(john, sushi).",
            "likes(john, pizza).",
        ],
    )


def test_failed_dynamic_same_predicate_clauses_still_fall_through_to_static(wam_config):
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
        wam_config,
    )

    _assert_answers(result, ["likes(john, pizza)."])


def test_dynamic_program_can_add_new_symbol_and_then_fall_back_to_static(wam_config):
    result = _solve_ontop(
        """
        likes(john, pizza).
        """,
        """
        likes(peter, pizza).
        """,
        "likes(X, pizza).",
        wam_config,
    )

    _assert_answers(result, ["likes(peter, pizza).", "likes(john, pizza)."])


def test_dynamic_program_can_add_new_predicate(wam_config):
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
        wam_config,
    )

    _assert_answers(result, ["dislikes(peter, pizza).", "dislikes(peter, pasta)."])


def test_single_static_clause_fallback_returns_dynamic_then_static_answer(wam_config):
    result = _solve_ontop(
        """
        likes(john, pizza).
        """,
        """
        likes(john, sushi).
        """,
        "likes(john, X).",
        wam_config,
    )

    _assert_answers(result, ["likes(john, sushi).", "likes(john, pizza)."])


def test_dynamic_rule_can_call_static_predicate_and_fall_back_to_static_rule(wam_config):
    result = _solve_ontop(
        """
        father(ted, bob).
        mother(jane, bob).
        parent(X, Y) :- father(X, Y).
        """,
        """
        parent(X, Y) :- mother(X, Y).
        """,
        "parent(X, bob).",
        wam_config,
    )

    _assert_answers(result, ["parent(jane, bob).", "parent(ted, bob)."])


def test_second_dynamic_compilation_does_not_leak_first_dynamic_program(wam_config):
    static_state, static_entry, static_size, symbol_table = _static_context(
        """
        likes(john, pizza).
        """,
        wam_config,
    )

    first = solve_from_str_ontop(
        """
        likes(peter, pizza).
        """,
        "likes(X, pizza).",
        static_state,
        static_entry,
        static_size,
        symbol_table,
        wam_config,
    )
    _assert_answers(first, ["likes(peter, pizza).", "likes(john, pizza)."])

    second = solve_from_str_ontop(
        """
        dislikes(peter, pasta).
        """,
        "likes(X, pizza).",
        static_state,
        static_entry,
        static_size,
        symbol_table,
        wam_config,
    )
    _assert_answers(second, ["likes(john, pizza)."])
