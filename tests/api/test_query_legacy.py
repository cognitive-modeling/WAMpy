"""Behavioral query tests migrated from the removed execution layer."""

import pytest
from wampy import Prolog, QueryError, QueryPhase, WAMStatus, load_config_toml


def test_undefined_predicate_inside_rule_body_returns_no_solutions():
    assert Prolog("p :- missing.").query_all("p.") == []


def test_anonymous_and_repeated_query_variables_are_handled_by_query():
    prolog = Prolog("p(a, b).")

    assert prolog.query_all("p(_, _).") == [{}]
    assert prolog.query_all("p(X, X).") == []


def test_recursive_loop_reports_the_runtime_status():
    config = load_config_toml("""
        [runtime]
        max_steps = 1
        """)

    with pytest.raises(QueryError, match="WAM error: STEP_LIMIT"):
        Prolog("loop :- loop.", config).query_all("loop.")


def test_tuple_shorthand_matches_comma_functor():
    tuple_query = Prolog("next_num((zero, one)).").query_all("next_num(','(zero, one)).")
    explicit_query = Prolog("next_num(','(zero, one)).").query_all("next_num((zero, one)).")

    assert tuple_query == [{}]
    assert explicit_query == [{}]


def test_repeated_body_variable_tuple_returns_without_freezing():
    program = """
    p1(((V1, V2), V1)) :- true.
    p2(((V1, V2), V2)) :- true.
    bad(V1) :- p2((V2, V2)), true.
    """

    assert Prolog(program).query_all("bad(a).") == [{}]


def test_nested_compound_arguments_match_and_fail():
    prolog = Prolog("q(((X, Y), X)).")

    assert prolog.query_all("q(((a, b), a)).") == [{}]
    assert prolog.query_all("q(((a, b), b)).") == []


def test_nested_compound_arguments_work_across_calls_and_negation():
    program = """
    q(((X, Y), X)).
    p((X, Y)) :- q((X, Z)), q((Z, Y)).
    """

    assert Prolog(program).query_all(r"\+ p((((a, b), c), b)).") == [{}]
    assert Prolog(program).query_all("p((((a, b), c), b)).") == []


def test_top_level_negation_reserves_target_argument_registers():
    prolog = Prolog("q(((X, Y), X)).")

    assert prolog.query_all(r"\+ q((((a, b), c), b)).") == [{}]


def test_top_level_negation_structure_overflow_is_compile_time_error():
    config = load_config_toml(
        """
        [compiler]
        max_arity = 2

        [runtime]
        max_x_registers = 2
        """
    )

    with pytest.raises(ValueError, match="X register capacity exceeded"):
        Prolog("q(X).", config).query_all(r"\+ q((((a, b), c), d)).")


def test_unify_step_limit_is_reported_as_a_query_error():
    config = load_config_toml("""
        [runtime]
        max_unify_steps = 1
        """)

    with pytest.raises(QueryError, match="WAM error: UNIFY_STEP_LIMIT"):
        Prolog("same(X, X).", config).query_all("same((a, b), (a, b)).")


def test_redo_heap_overflow_is_reported_as_query_error() -> None:
    program = """
    p(a).
    p(X) :- add(s(s(s(zero))), s(s(s(zero))), X).
    add(zero, Y, Y).
    add(s(X), Y, s(Z)) :- add(X, Y, Z).
    """

    config = load_config_toml("""
        [runtime]
        heap_size = 30
        trail_size = 1024
        choice_point_size = 512
        unify_stack_size = 1024
    """)

    query = Prolog(program, config).query("p(X).")

    assert next(query) == {"X": "a"}

    with pytest.raises(QueryError) as exc:
        next(query)

    assert exc.value.status == WAMStatus.HEAP_OVERFLOW


def test_query_lifecycle_reports_exhaustion_and_closes_state():
    query = Prolog("p(a).").query("p(X).")

    assert query.phase is QueryPhase.NEW
    assert next(query) == {"X": "a"}
    with pytest.raises(StopIteration):
        next(query)
    assert query.phase is QueryPhase.EXHAUSTED
    assert query.closed
