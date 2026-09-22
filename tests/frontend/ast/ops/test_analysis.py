import numpy as np
import pytest
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import (
    collect_head_nodes,
    count_clauses,
    get_child_count,
    get_clause_head_body,
    get_term_count,
    is_compound_node,
    is_node_allocated,
    is_variable_symbol,
    terms_have_equal_structure,
    variable_index,
)
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.program import init_ast_program
from wampy.frontend.ast.symbol_id import (
    CoreSymID,
)
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


@pytest.fixture(scope="session")
def wam_config():
    return WAMConfig()


def _symbol_id(binding, symbol):
    return next(symbol_id for symbol_id, symbol_text in binding.items() if symbol_text == symbol)


def test_analysis_operations_are_imported_from_analysis_module():
    assert collect_head_nodes is not None
    assert count_clauses is not None
    assert get_clause_head_body is not None
    assert get_term_count is not None
    assert get_child_count is not None
    assert is_compound_node is not None
    assert is_node_allocated is not None
    assert is_variable_symbol is not None
    assert variable_index is not None
    assert terms_have_equal_structure is not None


def test_count_clauses_counts_predicates_by_functor_and_arity(wam_config):
    program, binding = parse(
        """
        p(a).
        p(b).
        q.
        r(X, Y).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    counts = count_clauses(program)

    assert counts[_symbol_id(binding, "p"), 1] == 2
    assert counts[_symbol_id(binding, "q"), 0] == 1
    assert counts[_symbol_id(binding, "r"), 2] == 1
    assert np.sum(counts) == 4


def test_count_clauses_ignores_true_head_directives(wam_config):
    program, binding = parse(
        """
        true :- p(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    counts = count_clauses(program)

    assert counts.shape == (0, 0)
    assert np.sum(counts) == 0
    assert _symbol_id(binding, "p") >= 0


def test_count_clauses_handles_empty_program(wam_config):
    program, _binding = parse("", empty_symbol_table(wam_config.frontend.ast), wam_config)

    counts = count_clauses(program)

    assert counts.shape == (0, 0)


def test_count_clauses_splits_conjoined_heads(wam_config):
    program = init_ast_program(wam_config)
    p_fun = np.uint16(program.first_user_symbol_id)
    q_fun = np.uint16(program.first_user_symbol_id + 1)

    assert allocate_term(program, CoreSymID.CLAUSE) == 0
    head_and, ok = add_child_node(program, 0, 0, np.uint16(CoreSymID.CONJUNCTION))
    assert ok
    p_head, ok = add_child_node(program, 0, head_and, p_fun)
    assert ok
    q_head, ok = add_child_node(program, 0, head_and, q_fun)
    assert ok
    _, ok = add_child_node(
        program,
        0,
        p_head,
        np.uint16(program.first_user_symbol_id + 2),
    )
    assert ok
    _, ok = add_child_node(
        program,
        0,
        q_head,
        np.uint16(program.first_user_symbol_id + 3),
    )
    assert ok
    _, ok = add_child_node(program, 0, 0, np.uint16(CoreSymID.TRUE))
    assert ok

    counts = count_clauses(program)

    assert counts[p_fun, 1] == 1
    assert counts[q_fun, 1] == 1
    assert np.sum(counts) == 2
