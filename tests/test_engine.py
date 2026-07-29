"""
WAM Interpreter Test Suite
=========================

These tests validate runtime semantics of the WAM interpreter:
- unification
- variable binding
- backtracking
- choice points
- call / execute / proceed behavior

They assume the compiler is correct and test only interpreter behavior.
"""

import pytest

from wampy.compiler.compiler import compile, compile_program_onto, init_compiler
from wampy.config import DEFAULT_CONFIG, load_config_toml
from wampy.engine import _build_result, query
from wampy.frontend.answers import init_answers
from wampy.frontend.ast.program import init_program
from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.parser import (
    _extract_symbols,
    _split_clauses,
    add_prolog_term,
    build_new_program,
    build_query,
    ensure_query_symbols,
)
from wampy.frontend.symbol_table import SymbolTable
from wampy.runtime.interpreter import init_machine
from wampy.runtime.stack import init_stack
from wampy.status import WAMStatus
from wampy.utils.debug import debug_compiled_program

# Assumes cases.py sits next to this test file (same package / directory).
# If your tests directory is not a package, adjust to: `from cases import ...`
from .cases import (
    CASES,
    INVALID_QUERIES,
    RUNTIME_ERRORS,
    Case,
    ErrorCase,
    InvalidQueryCase,
)


def solve_from_str(
    program_src: str,
    query_src: str,
    config=DEFAULT_CONFIG,
    *,
    is_debug_compiler: bool = False,
):
    program, symbol_table = build_new_program(program_src, config)
    ensure_query_symbols(program, symbol_table, query_src)

    compiled = compile(program, init_compiler(config), config)

    if is_debug_compiler:
        debug_compiled_program(compiled, symbol_table)

    return _solve_result(compiled, query_src, symbol_table, config)


def solve_from_str_ontop(
    dynamic_src: str,
    query_src: str,
    static_state,
    static_entry,
    static_size,
    symbol_table,
    config=DEFAULT_CONFIG,
    *,
    is_debug_compiler: bool = False,
):
    extend_symbol_table_from_source(symbol_table, dynamic_src)

    dynamic_program = init_program(config)
    for clause in _split_clauses(dynamic_src):
        add_prolog_term(dynamic_program, clause, symbol_table)

    ensure_query_symbols(dynamic_program, symbol_table, query_src)

    compiled = compile_program_onto(
        static_state,
        static_entry,
        static_size,
        dynamic_program,
        config,
    )

    if is_debug_compiler:
        debug_compiled_program(compiled, symbol_table)

    return _solve_result(compiled, query_src, symbol_table, config)


def _solve_result(code_area, query_src, symbol_table, config):
    goals = build_query(query_src, symbol_table, config)
    stack = init_stack(config)
    answers = init_answers(
        config.solver.answer_max_answers,
        config.solver.answer_max_nodes,
        config.ast.num_terms,
    )
    machine = init_machine(code_area.code, code_area.entry, stack)

    nsol, status = query(code_area, machine, goals, stack, answers, config)
    return _build_result(query_src, status, nsol, answers, symbol_table)


def _next_nonvar_symbol_id(symbol_table: SymbolTable) -> int:
    max_id = -1
    for key in symbol_table.symbol_by_id.keys():
        key = int(key)
        if 0 <= key < SymID.VAR_FIRST:
            if key > max_id:
                max_id = key
    return max_id + 1


def extend_symbol_table_from_source(symbol_table: SymbolTable, source: str) -> None:
    seen = set()
    ordered_symbols = []

    for symbol in _extract_symbols(_split_clauses(source)):
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered_symbols.append(symbol)

    next_id = _next_nonvar_symbol_id(symbol_table)

    for symbol in ordered_symbols:
        if symbol in symbol_table.id_by_symbol:
            continue
        symbol_table.symbol_by_id[next_id] = symbol
        symbol_table.id_by_symbol[symbol] = next_id
        next_id += 1


CONFIG_TOML = """
[ast]
num_terms = 30
max_terms_nodes = 100
max_terms_nodes_childs = 5
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _assert_case(program, query, expected, expected_nsol, wam_config):
    result = solve_from_str(program, query, wam_config)

    assert result["answers"] == expected
    assert result["n_answers"] == expected_nsol
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in CASES])
def test_wam_cases(case: Case, wam_config):
    _assert_case(case.program, case.query, case.expected, case.n, wam_config)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in RUNTIME_ERRORS])
def test_expected_runtime_errors(case: ErrorCase, wam_config):
    with pytest.raises(RuntimeError) as exc:
        solve_from_str(case.program, case.query, wam_config)
    assert case.msg_substr in str(exc.value)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in INVALID_QUERIES])
def test_invalid_query_raises(case: InvalidQueryCase, wam_config):
    with pytest.raises(Exception):
        solve_from_str(case.program, case.query, wam_config)


@pytest.mark.requires_numba_jit
def test_tuple_shorthand_matches_comma_functor(wam_config):
    tuple_program = "next_num((zero, one))."
    explicit_program = "next_num(','(zero, one))."

    tuple_query = solve_from_str(tuple_program, "next_num(','(zero, one)).", wam_config)
    explicit_query = solve_from_str(explicit_program, "next_num((zero, one)).", wam_config)

    assert tuple_query["n_answers"] == 1
    assert explicit_query["n_answers"] == 1
    assert query.nopython_signatures


def test_repeated_body_variable_tuple_returns_without_freezing(wam_config):
    program = """
    p1(((V1, V2), V1)) :- true.
    p2(((V1, V2), V2)) :- true.
    bad(V1) :- p2((V2, V2)), true.
    """

    c_TOML = """

    [solver]
    answer_max_answers = 1
    """
    c = load_config_toml(CONFIG_TOML + c_TOML)

    result = solve_from_str(program, "bad(a).", c)

    assert result["n_answers"] == 1
    assert result["answers"] == ["bad(a)."]
    assert WAMStatus(int(result["status"])) == WAMStatus.SUCCESS


def test_recursive_generated_candidate_after_add_returns_bounded_error():
    program = """
    circle :- true.
    box :- true.
    zero :- true.
    one :- true.
    two :- true.
    three :- true.
    four :- true.
    five :- true.
    six :- true.
    seven :- true.
    eight :- true.
    nine :- true.
    ten :- true.
    next_num((zero, one)) :- true.
    next_num((one, two)) :- true.
    next_num((two, three)) :- true.
    next_num((three, four)) :- true.
    next_num((four, five)) :- true.
    next_num((five, six)) :- true.
    next_num((six, seven)) :- true.
    next_num((seven, eight)) :- true.
    next_num((eight, nine)) :- true.
    next_num((nine, ten)) :- true.
    first_next(((V1, V2), (V3, V4))) :- next_num((V1, V3)), true.
    second_next(((V1, V2), (V3, V4))) :- next_num((V2, V4)), true.
    transfer_one(((V3, V2), (V1, V2))) :- first_next((V3, V1)), second_next((V1, V3)), true.
    add(((V2, zero), V2)) :- true.
    add(V2) :- transfer_one((V2, V1)), add(V1), true.
    remove_both(V1) :- add((V2, V1)), remove_both((V1, V2)), next_num(V1), true.
    """
    query = "remove_both((((five, three), two), ((four, two), two)))."
    config = load_config_toml("""
        [ast]
        num_terms = 120
        max_terms_nodes = 100
        max_terms_nodes_childs = 2

        [entry]
        max_arity = 2
        max_functors = 200

        [stack]
        heap_size = 1024
        trail_size = 256
        cp_size = 128
        unify_stack_size = 128

        [solver]
        max_steps = 2048
        answer_max_answers=1
        answer_max_nodes=32
        """)

    with pytest.raises(RuntimeError, match="UNIFY_STEP_LIMIT|STEP_LIMIT|STACK_OVERFLOW"):
        solve_from_str(program, query, config)


def test_unify_step_limit_is_reported_as_wam_error():
    config = load_config_toml("""
        [ast]
        num_terms = 20
        max_terms_nodes = 40
        max_terms_nodes_childs = 2

        [entry]
        max_arity = 2

        [solver]
        max_unify_steps = 1
        answer_max_answers = 1
        """)

    program = """
    same(X, X).
    """

    with pytest.raises(RuntimeError, match="WAM error: UNIFY_STEP_LIMIT"):
        solve_from_str(program, "same((a, b), (a, b)).", config)


def test_redo_exception_is_mapped_to_wam_status():
    # First answer is cheap (p(a)); second answer forces deep add/3 construction on redo.
    program = """
    p(a).
    p(X) :- add(s(s(s(zero))), s(s(s(zero))), X).
    add(zero, Y, Y).
    add(s(X), Y, s(Z)) :- add(X, Y, Z).
    """
    query = "p(X)."

    tiny_stack_cfg = """
        [ast]
        num_terms = 60
        max_terms_nodes = 200
        max_terms_nodes_childs = 5

        [stack]
        heap_size = 30
        trail_size = 1024
        cp_size = 512
        unify_stack_size = 1024
        """

    c_TOML = """

    [solver]
    answer_max_answers = 1
    answer_max_nodes=512
    """
    c = load_config_toml(tiny_stack_cfg + c_TOML)

    one_answer = solve_from_str(
        program,
        query,
        c,
    )
    assert one_answer["n_answers"] == 1
    assert one_answer["answers"] == ["p(a)."]

    c_TOML = """

    [solver]
    answer_max_answers = 2
    answer_max_nodes=512
    """
    c = load_config_toml(tiny_stack_cfg + c_TOML)

    with pytest.raises(RuntimeError, match="STACK_OVERFLOW"):
        solve_from_str(program, query, c)


def test_top_level_not_exception_is_mapped_to_wam_status():
    program = """
    q(a).
    q(b).
    p(X) :- q(X), p(X).
    """

    tiny_stack_cfg = load_config_toml("""
        [ast]
        num_terms = 80
        max_terms_nodes = 300
        max_terms_nodes_childs = 4

        [stack]
        heap_size = 300
        trail_size = 8
        cp_size = 200
        unify_stack_size = 512

        [solver]
        max_steps = 50000
        answer_max_answers = 1
        answer_max_nodes = 128
        """)

    with pytest.raises(RuntimeError, match="STACK_OVERFLOW"):
        solve_from_str(program, "\\+ p(a).", tiny_stack_cfg)


def test_top_level_not_true_fails(wam_config):
    result = solve_from_str("", "\\+ true.", wam_config)

    assert result["n_answers"] == 0
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED


def test_top_level_not_proven_goal_fails(wam_config):
    result = solve_from_str("p(a).", "\\+ p(a).", wam_config)

    assert result["n_answers"] == 0
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED


def test_top_level_not_unproven_goal_succeeds(wam_config):
    result = solve_from_str("p(a).", "\\+ p(b).", wam_config)

    assert result["n_answers"] == 1
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED


def test_top_level_nested_not_preserves_parity(wam_config):
    result = solve_from_str("p(a).", "\\+ (\\+ p(a)).", wam_config)

    assert result["n_answers"] == 1
    assert WAMStatus(int(result["status"])) == WAMStatus.EXHAUSTED
