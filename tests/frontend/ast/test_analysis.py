import numpy as np
import pytest

import wampy.ast as ast
from wampy.config import load_config_toml
from wampy.frontend.ast import (
    copy_and_override_term,
    get_count_clauses,
    is_structure_of_terms_equal,
)
from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.ast.program import add_child_node, init_program
from wampy.frontend.parser import build_new_program

CONFIG_TOML = """
[ast]
num_terms = 30
max_terms_nodes = 100
max_terms_nodes_childs = 5
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _symbol_id(binding, symbol):
    return next(symbol_id for symbol_id, symbol_text in binding.items() if symbol_text == symbol)


def test_wampy_ast_import_exposes_public_analysis_helpers():
    assert ast.clause_head_body is not None
    assert ast.collect_head_nodes is not None
    assert ast.get_arity_of_node is not None
    assert ast.get_count_clauses is get_count_clauses
    assert ast.is_structure_of_terms_equal is is_structure_of_terms_equal
    assert ast.copy_and_override_term is copy_and_override_term


def test_get_count_clauses_counts_predicates_by_functor_and_arity(wam_config):
    program, binding = build_new_program(
        """
        p(a).
        p(b).
        q.
        r(X, Y).
        """,
        wam_config,
    )

    counts = get_count_clauses(program)

    assert counts[_symbol_id(binding, "p"), 1] == 2
    assert counts[_symbol_id(binding, "q"), 0] == 1
    assert counts[_symbol_id(binding, "r"), 2] == 1
    assert np.sum(counts) == 4


def test_get_count_clauses_ignores_true_head_directives(wam_config):
    program, binding = build_new_program(
        """
        true :- p(a).
        """,
        wam_config,
    )

    counts = get_count_clauses(program)

    assert counts.shape == (0, 0)
    assert np.sum(counts) == 0
    assert _symbol_id(binding, "p") >= 0


def test_get_count_clauses_handles_empty_program(wam_config):
    program, _binding = build_new_program("", wam_config)

    counts = get_count_clauses(program)

    assert counts.shape == (0, 0)


def test_get_count_clauses_splits_conjoined_heads(wam_config):
    program = init_program(wam_config)
    p_fun = np.uint16(1)
    q_fun = np.uint16(2)

    program.symbol[0, 0] = np.uint16(SymID.CLAUSE)
    head_and, ok = add_child_node(program, 0, 0, np.uint16(SymID.AND))
    assert ok
    p_head, ok = add_child_node(program, 0, head_and, p_fun)
    assert ok
    q_head, ok = add_child_node(program, 0, head_and, q_fun)
    assert ok
    _, ok = add_child_node(program, 0, p_head, np.uint16(3))
    assert ok
    _, ok = add_child_node(program, 0, q_head, np.uint16(4))
    assert ok
    _, ok = add_child_node(program, 0, 0, np.uint16(SymID.TRUE))
    assert ok

    counts = get_count_clauses(program)

    assert counts[p_fun, 1] == 1
    assert counts[q_fun, 1] == 1
    assert np.sum(counts) == 2
