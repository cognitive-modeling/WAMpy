import numpy as np
import pytest
from numba import jit
from wampy.api.ast import allocate_term as jitable_allocate_term
from wampy.api.ast import get_term_count as jitable_get_term_count
from wampy.config import ASTConfig, FrontendConfig, WAMConfig
from wampy.frontend.ast.ops.analysis import get_term_capacity, get_term_count
from wampy.frontend.ast.ops.mutation import allocate_term
from wampy.frontend.ast.program import init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID


def test_init_ast_program_uses_fixed_binary_links():
    config = WAMConfig(
        frontend=FrontendConfig(
            ast=ASTConfig(max_terms=2, max_nodes_per_term=10, child_block_size=4),
        ),
    )
    program = init_ast_program(config)

    assert program.node_symbols.shape == (2, 10)
    assert program.node_links.shape == (2, 10, 2)
    assert program.node_arities.shape == (2, 10)
    assert program.variable_name_ids.shape == (2, 32)
    assert program.variable_name_ids.dtype == np.uint16
    assert get_term_capacity(program) == 2
    assert get_term_count(program) == 0
    assert program.term_count.shape == (1,)
    assert program.term_count.dtype == np.int32
    assert not hasattr(program, "child_block_next_ids")
    assert not hasattr(program, "free_child_block_stack")

    for term_id in range(2):
        assert program.free_node_count[term_id] == 9


def test_init_ast_program_num_terms_override():
    config = WAMConfig(
        frontend=FrontendConfig(
            ast=ASTConfig(max_terms=2, max_nodes_per_term=10, child_block_size=4),
        ),
    )

    program = init_ast_program(config, num_terms=3)

    assert program.node_symbols.shape[0] == 3
    assert program.node_links.shape == (3, 10, 2)
    assert get_term_capacity(program) == 3
    assert get_term_count(program) == 0


def test_allocate_term_appends_contiguously_and_preserves_capacity():
    program = init_ast_program(WAMConfig(), num_terms=4)

    term_ids = [allocate_term(program, CoreSymID.CLAUSE) for _ in range(3)]

    assert term_ids == [0, 1, 2]
    assert get_term_capacity(program) == 4
    assert get_term_count(program) == 3


def test_allocate_term_capacity_failure_preserves_count():
    program = init_ast_program(WAMConfig(), num_terms=2)
    allocate_term(program, CoreSymID.CLAUSE)
    allocate_term(program, CoreSymID.QUERY)

    with pytest.raises(RuntimeError, match="No empty term available"):
        allocate_term(program, CoreSymID.CLAUSE)

    assert get_term_count(program) == 2


@pytest.mark.requires_numba_jit
def test_term_count_and_allocation_are_callable_from_numba():
    program = init_ast_program(WAMConfig(), num_terms=2)

    @jit
    def allocate_and_count(ast_program):
        first = jitable_allocate_term(ast_program, CoreSymID.CLAUSE)
        second = jitable_allocate_term(ast_program, CoreSymID.QUERY)
        return first, second, jitable_get_term_count(ast_program)

    assert allocate_and_count(program) == (0, 1, 2)
    assert allocate_and_count.nopython_signatures
