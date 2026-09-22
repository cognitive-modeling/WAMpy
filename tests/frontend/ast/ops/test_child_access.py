import numpy as np
from wampy.config import ASTConfig, FrontendConfig, WAMConfig
from wampy.frontend.ast.ops.analysis import (
    get_child_at,
    get_child_count,
    get_first_child,
    get_user_symbol_id,
)
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.program import NO_NODE, init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID


def _init_program(max_nodes=40):
    config = WAMConfig(
        frontend=FrontendConfig(
            ast=ASTConfig(max_terms=1, max_nodes_per_term=max_nodes),
        ),
    )
    return init_ast_program(config)


def test_child_access_returns_no_node_for_node_without_children():
    program = _init_program()
    assert get_child_count(program, 0, 0) == 0
    assert get_child_at(program, 0, 0, 0) == NO_NODE
    assert get_first_child(program, 0, 0) == NO_NODE


def test_direct_control_nodes_use_two_physical_links():
    program = _init_program()
    assert allocate_term(program, CoreSymID.CONJUNCTION) == 0
    left, ok = add_child_node(program, 0, 0, np.uint16(CoreSymID.TRUE))
    assert ok
    right, ok = add_child_node(program, 0, 0, np.uint16(CoreSymID.TRUE))
    assert ok

    assert get_child_count(program, 0, 0) == 2
    assert get_child_at(program, 0, 0, 0) == left
    assert get_child_at(program, 0, 0, 1) == right
    assert program.node_links[0, 0, 0] == left
    assert program.node_links[0, 0, 1] == right


def test_user_functor_arguments_are_hidden_behind_args_chain():
    program = _init_program()
    assert allocate_term(program, get_user_symbol_id(program, 0)) == 0
    child_ids = []

    for index in range(9):
        child_id, ok = add_child_node(
            program,
            0,
            0,
            get_user_symbol_id(program, index + 1),
        )
        assert ok
        child_ids.append(child_id)

    assert get_child_count(program, 0, 0) == 9
    assert [get_child_at(program, 0, 0, i) for i in range(9)] == child_ids

    first_cell = int(program.node_links[0, 0, 0])
    tail_cell = int(program.node_links[0, 0, 1])
    assert program.node_symbols[0, first_cell] == CoreSymID.ARGS
    assert program.node_symbols[0, tail_cell] == CoreSymID.ARGS
    assert int(program.node_arities[0, tail_cell]) == 1

    cell = first_cell
    cells = 0
    while cell != int(NO_NODE):
        cells += 1
        assert program.node_symbols[0, cell] == CoreSymID.ARGS
        cell = int(program.node_links[0, cell, 1])
    assert cells == 9


def test_child_at_returns_no_node_for_negative_and_out_of_range_indexes():
    program = _init_program()
    assert allocate_term(program, get_user_symbol_id(program, 0)) == 0
    child_id, ok = add_child_node(program, 0, 0, get_user_symbol_id(program, 1))
    assert ok

    assert get_child_at(program, 0, 0, -1) == NO_NODE
    assert get_child_at(program, 0, 0, 1) == NO_NODE
    assert get_child_at(program, 0, 0, 99) == NO_NODE
    assert get_first_child(program, 0, 0) == child_id
