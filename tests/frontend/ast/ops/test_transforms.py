import numpy as np
import wampy.api.ast as ast
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import terms_have_equal_structure
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.ops.transforms import copy_program, copy_term
from wampy.frontend.ast.program import ASTProgram, init_ast_program
from wampy.frontend.ast.program_metadata import copy_term_with_metadata, init_program_metadata
from wampy.frontend.ast.symbol_id import CoreSymID


def _init_test_program(num_terms: int):
    return init_ast_program(WAMConfig(), num_terms=num_terms)


def _program_with_values():
    program = _init_test_program(2)
    metadata = init_program_metadata(program)
    assert allocate_term(program, CoreSymID.CLAUSE) == 0
    child_id, ok = add_child_node(program, 0, 0, program.first_user_symbol_id)
    assert ok
    add_child_node(program, 0, child_id, program.first_user_symbol_id + 1)
    metadata.usage[0] = 4
    return program, metadata


def test_copy_program_preserves_values():
    program, _ = _program_with_values()
    copied = copy_program(program)

    assert isinstance(copied, ASTProgram)
    assert copied is not program
    assert np.array_equal(copied.node_links, program.node_links)
    assert np.array_equal(copied.node_symbols, program.node_symbols)
    assert np.array_equal(copied.free_node_stack, program.free_node_stack)
    assert np.array_equal(copied.free_node_count, program.free_node_count)
    assert np.array_equal(copied.node_arities, program.node_arities)
    assert np.array_equal(copied.term_count, program.term_count)
    assert np.array_equal(copied.variable_name_ids, program.variable_name_ids)


def test_copy_program_owns_independent_arrays():
    program, _ = _program_with_values()
    copied = copy_program(program)

    assert copied.node_links is not program.node_links
    assert copied.node_symbols is not program.node_symbols
    assert copied.free_node_stack is not program.free_node_stack
    assert copied.free_node_count is not program.free_node_count
    assert copied.node_arities is not program.node_arities
    assert copied.term_count is not program.term_count
    assert copied.variable_name_ids is not program.variable_name_ids


def test_mutating_program_copy_does_not_change_source():
    program, _ = _program_with_values()
    copied = copy_program(program)
    original_symbol = program.node_symbols[0, 0]
    original_child_slot = program.node_links[0, 0, 0]

    copied.node_symbols[0, 0] = 255
    copied.node_links[0, 0, 0] = ast.NO_NODE

    assert program.node_symbols[0, 0] == original_symbol
    assert copied.node_symbols[0, 0] == 255
    assert program.node_links[0, 0, 0] == original_child_slot
    assert copied.node_links[0, 0, 0] == ast.NO_NODE


def test_mutating_source_does_not_change_program_copy():
    program, _ = _program_with_values()
    copied = copy_program(program)
    previous = copied.node_symbols[0, 0]
    program.node_symbols[0, 0] = 255
    assert copied.node_symbols[0, 0] == previous


def test_copy_program_is_exposed_by_public_ast_api():
    assert callable(ast.copy_program)


def test_copy_term_replaces_destination_contents():
    program = _init_test_program(2)
    assert allocate_term(program, program.first_user_symbol_id) == 0
    source_child, ok = add_child_node(program, 0, 0, program.first_user_symbol_id + 1)
    assert ok
    add_child_node(program, 0, source_child, program.first_user_symbol_id + 2)

    assert allocate_term(program, program.first_user_symbol_id + 3) == 1
    destination_child, ok = add_child_node(program, 1, 0, program.first_user_symbol_id + 4)
    assert ok
    add_child_node(program, 1, destination_child, program.first_user_symbol_id + 5)

    copy_term(program, destination_term=1, source_term=0)

    assert np.array_equal(program.node_symbols[1], program.node_symbols[0])
    assert np.array_equal(program.node_links[1], program.node_links[0])
    assert np.array_equal(program.node_arities[1], program.node_arities[0])
    assert np.array_equal(program.variable_name_ids[1], program.variable_name_ids[0])
    assert np.array_equal(program.free_node_stack[1], program.free_node_stack[0])
    assert program.free_node_count[1] == program.free_node_count[0]
    assert terms_have_equal_structure(program, 0, 1)


def test_copy_term_with_metadata_replaces_ast_and_sidecar():
    program = _init_test_program(2)
    metadata = init_program_metadata(program)
    assert allocate_term(program, program.first_user_symbol_id) == 0
    source_child, ok = add_child_node(program, 0, 0, program.first_user_symbol_id + 1)
    assert ok
    metadata.symbol_kind[0, 0] = 1
    metadata.symbol_kind[0, source_child] = 4
    metadata.usage[0] = 3
    metadata.weights[0] = 0.75

    assert allocate_term(program, program.first_user_symbol_id + 2) == 1
    destination_child, ok = add_child_node(program, 1, 0, program.first_user_symbol_id + 3)
    assert ok
    metadata.symbol_kind[1, 0] = 5
    metadata.symbol_kind[1, destination_child] = 3

    copy_term_with_metadata(program, metadata, destination_term=1, source_term=0)

    assert np.array_equal(program.node_symbols[1], program.node_symbols[0])
    assert np.array_equal(metadata.symbol_kind[1], metadata.symbol_kind[0])
    assert metadata.usage[1] == metadata.usage[0]
    assert metadata.weights[1] == metadata.weights[0]
