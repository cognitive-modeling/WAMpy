import numpy as np
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import get_term_count
from wampy.frontend.ast.ops.mutation import allocate_term
from wampy.frontend.ast.program import NO_NODE, init_ast_program
from wampy.frontend.ast.program_metadata import (
    SymbolKind,
    add_child_node_with_kind,
    classify_program,
    copy_metadata_term,
    init_program_metadata,
    invalidate_term_metadata,
    remove_descendants_with_metadata,
    reset_program_metadata,
    update_usage,
)
from wampy.frontend.ast.symbol_id import (
    CoreSymID,
)
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


def _init_test_program(num_terms: int):
    return init_ast_program(WAMConfig(), num_terms=num_terms)


def test_metadata_is_optional_and_has_explicit_shapes_and_dtypes():
    program = _init_test_program(3)

    assert not hasattr(program, "usage")
    assert np.all(program.node_symbols == np.uint16(CoreSymID.NO_SYMBOL.value))

    metadata = init_program_metadata(program)

    assert metadata.symbol_kind.shape == program.node_symbols.shape
    assert metadata.symbol_kind.dtype == np.uint8
    assert metadata.usage.shape == (3,)
    assert metadata.usage.dtype == np.int64
    assert metadata.weights.shape == (3,)
    assert metadata.weights.dtype == np.float64
    assert np.all(metadata.symbol_kind == SymbolKind.UNSET)
    assert np.all(metadata.usage == 0)
    assert np.all(metadata.weights == 0.0)


def test_usage_and_metadata_reset_are_term_aligned():
    program = _init_test_program(3)
    metadata = init_program_metadata(program)
    metadata.symbol_kind[0, 1] = SymbolKind.SYMBOL
    metadata.symbol_kind[1, 2] = SymbolKind.VARIABLE
    metadata.usage[:] = (3, 4, 5)
    metadata.weights[:] = (0.25, 0.5, 0.75)

    update_usage(metadata, np.array([8, 6], dtype=np.int64))
    assert np.array_equal(metadata.usage, np.array([8, 6, 0], dtype=np.int64))

    reset_program_metadata(metadata, term_id=1)
    assert np.all(metadata.symbol_kind[1] == SymbolKind.UNSET)
    assert metadata.usage[1] == 0
    assert metadata.weights[1] == 0.0
    assert metadata.usage[0] == 8

    reset_program_metadata(metadata)
    assert np.all(metadata.symbol_kind == SymbolKind.UNSET)
    assert np.all(metadata.usage == 0)
    assert np.all(metadata.weights == 0.0)


def test_structural_invalidation_resets_only_one_term_usage_and_weight():
    program = _init_test_program(3)
    metadata = init_program_metadata(program)
    metadata.usage[:] = (2, 4, 6)
    metadata.weights[:] = (0.2, 0.4, 0.6)

    invalidate_term_metadata(metadata, 1)

    assert metadata.usage[1] == 0
    assert metadata.weights[1] == 0.0
    assert np.array_equal(metadata.usage, np.array([2, 0, 6], dtype=np.int64))
    assert np.array_equal(metadata.weights, np.array([0.2, 0.0, 0.6]))


def test_metadata_aware_allocation_deletion_and_copy():
    program = _init_test_program(2)
    metadata = init_program_metadata(program)
    assert allocate_term(program, CoreSymID.CLAUSE) == 0
    metadata.symbol_kind[0, 0] = SymbolKind.START

    child_id, ok = add_child_node_with_kind(
        program,
        metadata,
        0,
        0,
        program.first_user_symbol_id,
        SymbolKind.TERM,
    )
    assert ok
    grandchild_id, ok = add_child_node_with_kind(
        program,
        metadata,
        0,
        child_id,
        17,
        SymbolKind.SYMBOL,
    )
    assert ok

    copy_metadata_term(metadata, 1, 0)
    assert np.array_equal(metadata.symbol_kind[1], metadata.symbol_kind[0])

    remove_descendants_with_metadata(0, 0, program, metadata)
    assert program.node_links[0, 0, 0] == NO_NODE
    assert program.node_symbols[0, child_id] == CoreSymID.NO_SYMBOL
    assert program.node_symbols[0, grandchild_id] == CoreSymID.NO_SYMBOL
    assert metadata.symbol_kind[0, child_id] == SymbolKind.UNSET
    assert metadata.symbol_kind[0, grandchild_id] == SymbolKind.UNSET


def test_classification_distinguishes_body_and_nested_tuple_and():
    program, _ = parse("p((X, b)) :- q(X), r(b).", empty_symbol_table())
    metadata = init_program_metadata(program)
    classify_program(program, metadata)

    term_id = next(
        term_id
        for term_id in range(get_term_count(program))
        if program.node_symbols[term_id, 0] == CoreSymID.CLAUSE
    )
    assert metadata.symbol_kind[term_id, 0] == SymbolKind.START

    and_kinds = []
    variable_count = 0
    symbol_count = 0
    for node_id in range(program.node_symbols.shape[1]):
        symbol_id = program.node_symbols[term_id, node_id]
        kind = metadata.symbol_kind[term_id, node_id]
        if symbol_id == CoreSymID.CONJUNCTION:
            and_kinds.append(kind)
        if program.first_variable_symbol_id <= symbol_id < program.first_user_symbol_id:
            variable_count += 1
        if kind == SymbolKind.SYMBOL:
            symbol_count += 1

    assert SymbolKind.BODY in and_kinds
    assert SymbolKind.SYMBOL in and_kinds
    assert variable_count > 0
    assert symbol_count > 0
