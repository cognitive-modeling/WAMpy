import numpy as np
import pytest
from wampy import Prolog, QueryError, QueryPhase, WAMStatus, load_config_toml
from wampy.compiler.opcodes import OP


def _sample_prolog() -> Prolog:
    return Prolog("""
        parent(anakin, luke).
        parent(padme, luke).
        parent(han, ben).
        """)


def test_query_supports_python_and_explicit_next() -> None:
    query = _sample_prolog().query("parent(X, luke).")

    assert iter(query) is query
    assert query.phase is QueryPhase.NEW
    assert query.status is None
    assert not query.closed

    assert next(query) == {"X": "anakin"}
    assert query.phase is QueryPhase.ACTIVE
    assert query.status is WAMStatus.SUCCESS
    assert not query.closed

    assert query.next() == {"X": "padme"}
    assert query.phase is QueryPhase.ACTIVE
    assert query.status is WAMStatus.SUCCESS
    assert not query.closed

    with pytest.raises(StopIteration):
        next(query)

    assert query.phase is QueryPhase.EXHAUSTED
    assert query.status is WAMStatus.EXHAUSTED
    assert query.closed

    assert query.next() is None


def test_queries_have_independent_backtracking_state() -> None:
    prolog = _sample_prolog()

    with prolog.query("parent(X, luke).") as parents:
        with prolog.query("parent(anakin, Y).") as children:
            assert next(parents) == {"X": "anakin"}
            assert next(children) == {"Y": "luke"}
            assert next(parents) == {"X": "padme"}

            with pytest.raises(StopIteration):
                next(children)

            with pytest.raises(StopIteration):
                next(parents)

    assert parents.closed
    assert children.closed


def test_ground_query_returns_empty_bindings():
    prolog = _sample_prolog()

    assert prolog.query_all("parent(anakin, luke).") == [{}]


def test_failed_query_returns_no_solutions():
    prolog = _sample_prolog()

    assert prolog.query_all("parent(han, luke).") == []


def test_anonymous_variables_are_omitted_from_bindings():
    prolog = _sample_prolog()

    assert prolog.query_all("parent(_, Y).") == [
        {"Y": "luke"},
        {"Y": "luke"},
        {"Y": "ben"},
    ]


def test_repeated_variable_uses_one_binding():
    prolog = Prolog("""
        pair(a, a).
        pair(a, b).
        """)

    assert prolog.query_all("pair(X, X).") == [{"X": "a"}]


def test_nested_terms_are_decoded():
    prolog = Prolog("pair(f(a, b), g(c)).")

    assert prolog.query_all("pair(f(X, b), g(Y)).") == [{"X": "a", "Y": "c"}]


def test_left_nested_clause_head_can_construct_an_output_term():
    prolog = Prolog("shape(((a, b), c)).")

    assert prolog.query_all("shape(Result).") == [{"Result": "((a, b), c)"}]


def test_close_is_idempotent_and_stops_iteration():
    query = _sample_prolog().query("parent(X, luke).")

    assert next(query) == {"X": "anakin"}

    query.close()
    query.close()

    assert query.closed
    assert query.phase is QueryPhase.CLOSED

    with pytest.raises(StopIteration):
        next(query)


def test_context_manager_closes_query():
    prolog = _sample_prolog()

    with prolog.query("parent(X, luke).") as query:
        assert next(query) == {"X": "anakin"}
        assert not query.closed

    assert query.closed
    assert query.phase is QueryPhase.CLOSED


def test_all_queries_use_local_code_and_preserve_shared_compilation():
    prolog = Prolog("p(a).")
    code_before = prolog.compiled_program.code.copy()
    pc_before = int(prolog.compiler_state.pc[0])

    ordinary = prolog.prepare("p(X).")
    succeeds = prolog.prepare(r"\+(p(b)).")
    fails = prolog.prepare(r"\+(p(a)).")

    assert not np.shares_memory(ordinary.code, prolog.compiled_program.code)
    assert OP.CALL in ordinary.code[: ordinary.code_size[0], 0]
    assert OP.PROCEED == ordinary.code[ordinary.code_size[0] - 1, 0]

    assert prolog.query_all(ordinary) == [{"X": "a"}]
    succeeds_query = prolog.query(succeeds)
    fails_query = prolog.query(fails)

    assert next(succeeds_query) == {}

    with pytest.raises(StopIteration):
        next(fails_query)

    with pytest.raises(StopIteration):
        next(succeeds_query)

    assert succeeds_query.closed
    assert fails_query.closed
    np.testing.assert_array_equal(prolog.compiled_program.code, code_before)
    assert int(prolog.compiler_state.pc[0]) == pc_before


@pytest.mark.xfail(
    reason="Parser should not detect the undefined predicate.",
    strict=True,
)
def test_undefined_predicate_closes_query_and_records_error_status():
    query = Prolog("p(a).").query("missing(X).")

    with pytest.raises(
        QueryError,
        match="WAM error: UNDEFINED_PREDICATE",
    ) as error:
        next(query)

    assert error.value.status is WAMStatus.UNDEFINED_PREDICATE
    assert query.phase is QueryPhase.FAILED
    assert query.status is WAMStatus.UNDEFINED_PREDICATE
    assert query.closed

    with pytest.raises(StopIteration):
        next(query)


def test_runtime_error_closes_query_and_records_error_status():
    config = load_config_toml("""
        [runtime]
        max_steps = 1
        max_answers = 1
        max_answer_nodes = 32
        """)
    query = Prolog("loop :- loop.", config).query("loop.")

    with pytest.raises(QueryError, match="WAM error: STEP_LIMIT") as error:
        next(query)

    assert error.value.status is WAMStatus.STEP_LIMIT
    assert query.phase is QueryPhase.FAILED
    assert query.status is WAMStatus.STEP_LIMIT
    assert query.closed

    with pytest.raises(StopIteration):
        next(query)
