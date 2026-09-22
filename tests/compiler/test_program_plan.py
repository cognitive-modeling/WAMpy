import numpy as np
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.program_plan import (
    PLAN_CLAUSE_BODY,
    PLAN_CLAUSE_NEXT,
    PLAN_CLAUSE_TID,
    PLAN_SIGNATURE_ARITY,
    PLAN_SIGNATURE_FIRST_CLAUSE,
    PLAN_SIGNATURE_SLOT,
    PLAN_SIGNATURE_WIDTH,
    build_program_compile_plan,
)
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.ast.ops.analysis import get_term_count, get_user_symbol_id
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.program import init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


def _build_plan(program, config=DEFAULT_CONFIG):
    compiler_state = init_compiler_state(config)
    plan = build_program_compile_plan(
        program,
        compiler_state,
        config.compiler,
        config.runtime.max_x_registers,
    )
    return plan, compiler_state


def _signature_index(plan, predicate_slot, arity):
    for signature_index in range(plan.signature_count):
        if (
            plan.signatures[signature_index, PLAN_SIGNATURE_SLOT] == predicate_slot
            and plan.signatures[signature_index, PLAN_SIGNATURE_ARITY] == arity
        ):
            return signature_index
    raise AssertionError("Signature is missing from compile plan")


def _signature_clause_tids(plan, signature_index):
    clause_index = plan.signatures[signature_index, PLAN_SIGNATURE_FIRST_CLAUSE]
    tids = []
    while clause_index >= 0:
        tids.append(int(plan.clauses[clause_index, PLAN_CLAUSE_TID]))
        clause_index = plan.clauses[clause_index, PLAN_CLAUSE_NEXT]
    return tids


def test_plan_groups_interleaved_clauses_by_slot_and_arity() -> None:
    program, symbol_table = parse(
        """
        p(a).
        q(x).
        p(a, b).
        p(c).
        q(y).
        """,
        empty_symbol_table(),
    )
    plan, compiler_state = _build_plan(program)

    p_slot = next(
        index
        for index, symbol_id in enumerate(compiler_state.symbol_by_predicate_slot)
        if symbol_id == symbol_table.id_by_symbol["p"]
    )
    q_slot = next(
        index
        for index, symbol_id in enumerate(compiler_state.symbol_by_predicate_slot)
        if symbol_id == symbol_table.id_by_symbol["q"]
    )

    assert plan.clause_count == 5
    assert plan.signature_count == 3
    assert plan.clauses.shape == (5, 4)
    assert plan.signatures.shape == (5, PLAN_SIGNATURE_WIDTH)
    assert _signature_clause_tids(plan, _signature_index(plan, p_slot, 1)) == [0, 3]
    assert _signature_clause_tids(plan, _signature_index(plan, q_slot, 1)) == [1, 4]
    assert _signature_clause_tids(plan, _signature_index(plan, p_slot, 2)) == [2]


def test_plan_expands_multiple_heads_from_one_ast_term() -> None:
    program = init_ast_program(DEFAULT_CONFIG)
    assert allocate_term(program, CoreSymID.CLAUSE) == 0

    p_symbol = get_user_symbol_id(program, 0)
    q_symbol = get_user_symbol_id(program, 1)
    head, ok = add_child_node(program, 0, 0, CoreSymID.CONJUNCTION)
    assert ok
    _, ok = add_child_node(program, 0, head, p_symbol)
    assert ok
    _, ok = add_child_node(program, 0, head, q_symbol)
    assert ok
    body, ok = add_child_node(program, 0, 0, CoreSymID.TRUE)
    assert ok

    plan, _ = _build_plan(program)

    assert plan.clause_count == 2
    assert get_term_count(program) == 1
    assert plan.signature_count == 2
    assert plan.clauses[:, PLAN_CLAUSE_TID].tolist() == [0, 0]
    assert np.all(plan.clauses[:, PLAN_CLAUSE_BODY] == body)


def test_empty_program_builds_an_empty_sparse_plan() -> None:
    program, _ = parse("", empty_symbol_table())
    plan, _ = _build_plan(program)

    assert plan.clause_count == 0
    assert plan.signature_count == 0
    assert plan.clauses.shape == (0, 4)
    assert plan.signatures.shape == (0, PLAN_SIGNATURE_WIDTH)


def test_plan_storage_depends_on_concrete_clauses_not_compiler_limits() -> None:
    config = DEFAULT_CONFIG._replace(
        compiler=DEFAULT_CONFIG.compiler._replace(
            max_predicates=512,
            max_arity=64,
        ),
        runtime=DEFAULT_CONFIG.runtime._replace(max_x_registers=64),
    )
    program, _ = parse(
        "father(a, b). father(c, d). father(e, f).",
        empty_symbol_table(config.frontend.ast),
        config,
    )

    plan, _ = _build_plan(program, config)

    assert plan.clause_count == 3
    assert get_term_count(program) == 3
    assert plan.signature_count == 1
    assert plan.clauses.shape == (3, 4)
    assert plan.signatures.shape == (3, PLAN_SIGNATURE_WIDTH)
