import numpy as np
from wampy.config import ASTConfig, FrontendConfig, WAMConfig
from wampy.frontend.ast.ops.analysis import get_user_symbol_id
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.ops.validation import (
    is_valid_program,
    is_valid_symbol_id,
    is_valid_term,
)
from wampy.frontend.ast.program import NO_NODE, init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID


def _init_test_program(num_terms: int):
    config = WAMConfig(
        frontend=FrontendConfig(
            ast=ASTConfig(max_nodes_per_term=20),
        ),
    )
    return init_ast_program(config, num_terms=num_terms)


def test_initialized_program_is_valid():
    program = _init_test_program(2)
    assert is_valid_program(program)
    assert is_valid_term(program, 0)


def test_validation_accepts_a_constructed_clause():
    program = _init_test_program(1)
    assert allocate_term(program, CoreSymID.CLAUSE) == 0
    head, ok = add_child_node(program, 0, 0, get_user_symbol_id(program, 0))
    assert ok
    _, ok = add_child_node(program, 0, head, get_user_symbol_id(program, 1))
    assert ok
    _, ok = add_child_node(program, 0, 0, np.uint16(CoreSymID.TRUE))
    assert ok
    assert is_valid_program(program)


def test_symbol_validation_accepts_internal_args_and_rejects_reserved_gaps():
    program = _init_test_program(1)
    assert is_valid_symbol_id(program, CoreSymID.NO_SYMBOL)
    assert is_valid_symbol_id(program, CoreSymID.ARGS)
    assert not is_valid_symbol_id(program, 6)
    assert not is_valid_symbol_id(program, 15)


def test_validation_rejects_a_broken_args_chain():
    program = _init_test_program(1)
    assert allocate_term(program, get_user_symbol_id(program, 0)) == 0
    for index in range(3):
        child_id, ok = add_child_node(
            program,
            0,
            0,
            get_user_symbol_id(program, index + 1),
        )
        assert ok
        assert child_id != NO_NODE

    first_cell = int(program.node_links[0, 0, 0])
    program.node_links[0, first_cell, 1] = NO_NODE

    assert not is_valid_term(program, 0)
