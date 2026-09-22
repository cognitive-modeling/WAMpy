import numpy as np
from wampy.config import ASTConfig, FrontendConfig, WAMConfig
from wampy.frontend.ast.ops.analysis import get_child_at, get_user_symbol_id
from wampy.frontend.ast.ops.mutation import (
    add_child_node,
    allocate_term,
    remove_descendants,
)
from wampy.frontend.ast.program import NO_NODE, init_ast_program
from wampy.frontend.ast.program_metadata import init_program_metadata, update_usage
from wampy.frontend.ast.symbol_id import CoreSymID


def _init_test_program(num_terms: int):
    config = WAMConfig(
        frontend=FrontendConfig(
            ast=ASTConfig(max_nodes_per_term=40),
        ),
    )
    return init_ast_program(config, num_terms=num_terms)


def test_add_child_node_direct_control_links():
    program = _init_test_program(2)
    term_id = 0
    assert allocate_term(program, CoreSymID.CONJUNCTION) == term_id

    child_10, ok = add_child_node(program, term_id, 0, CoreSymID.TRUE)
    assert ok
    child_20, ok = add_child_node(program, term_id, 0, CoreSymID.CUT)
    assert ok

    assert program.node_links[term_id, 0, 0] == child_10
    assert program.node_links[term_id, 0, 1] == child_20
    assert program.node_arities[term_id, 0] == 2


def test_remove_descendants_reclaims_argument_values_and_args_cells():
    program = _init_test_program(1)
    term_id = 0
    assert allocate_term(program, get_user_symbol_id(program, 0)) == term_id

    children = []
    for index in range(5):
        child_id, ok = add_child_node(
            program,
            term_id,
            0,
            get_user_symbol_id(program, index + 1),
        )
        assert ok
        children.append(child_id)

    free_top_before = int(program.free_node_count[term_id])
    assert free_top_before == 39 - 10
    first_cell = int(program.node_links[term_id, 0, 0])
    assert program.node_symbols[term_id, first_cell] == CoreSymID.ARGS

    remove_descendants(term_id, 0, program)

    assert program.node_arities[term_id, 0] == 0
    assert np.all(program.node_links[term_id, 0] == NO_NODE)
    assert int(program.free_node_count[term_id]) == 39
    for child_id in children:
        assert program.node_symbols[term_id, child_id] == CoreSymID.NO_SYMBOL


def test_reclaimed_argument_nodes_are_reused():
    program = _init_test_program(1)
    term_id = 0
    assert allocate_term(program, get_user_symbol_id(program, 0)) == term_id

    first, ok = add_child_node(program, term_id, 0, get_user_symbol_id(program, 1))
    assert ok
    remove_descendants(term_id, 0, program)

    replacement, ok = add_child_node(program, term_id, 0, get_user_symbol_id(program, 2))
    assert ok
    assert replacement != NO_NODE
    assert get_child_at(program, term_id, 0, 0) == replacement
    assert int(program.free_node_count[term_id]) == 37
    assert first == replacement or program.node_symbols[term_id, first] == CoreSymID.ARGS


def test_update_usage_overwrites_usage_rows():
    program = _init_test_program(4)
    metadata = init_program_metadata(program)

    update_usage(metadata, np.array([1, 0, 3, 2], dtype=np.int64))
    update_usage(metadata, np.array([2, 2, 0, 1], dtype=np.int64))

    assert np.array_equal(metadata.usage, np.array([2, 2, 0, 1], dtype=np.int64))


def test_update_usage_handles_short_answer_row():
    program = _init_test_program(4)
    metadata = init_program_metadata(program)
    update_usage(metadata, np.array([9, 9, 9, 9], dtype=np.int64))

    update_usage(metadata, np.array([5, 7], dtype=np.int64))

    assert np.array_equal(metadata.usage, np.array([5, 7, 0, 0], dtype=np.int64))
