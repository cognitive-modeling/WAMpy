"""Low-level in-place operations on the array-backed AST."""

import numpy as np

from wampy.frontend.ast.ops.analysis import (
    get_node_symbol,
    get_term_capacity,
    is_user_symbol,
)
from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID


def set_node_symbol(
    program: ASTProgram,
    term_id: int,
    node_id: int,
    symbol_id: int,
) -> None:
    """Set the symbol of one allocated AST node."""
    program.node_symbols[term_id, node_id] = symbol_id


def clear_term(program: ASTProgram, term_id: int) -> None:
    """Reset one complete term slot and return all of its nodes to the allocator."""
    n_nodes = program.node_symbols.shape[1]
    n_child_slots = program.node_links.shape[2]

    for node_id in range(n_nodes):
        program.node_symbols[term_id, node_id] = CoreSymID.NO_SYMBOL
        for slot in range(n_child_slots):
            program.node_links[term_id, node_id, slot] = NO_NODE
        program.node_arities[term_id, node_id] = 0
        if node_id < program.free_node_stack.shape[1]:
            program.free_node_stack[term_id, node_id] = n_nodes - 1 - node_id

    invalid_name_id = np.iinfo(program.variable_name_ids.dtype).max
    for variable_slot in range(program.variable_name_ids.shape[1]):
        program.variable_name_ids[term_id, variable_slot] = invalid_name_id

    program.free_node_count[term_id] = n_nodes - 1


def allocate_term(
    program: ASTProgram,
    root_symbol: int,
) -> int:
    """Append one allocated term and initialize its root."""
    if root_symbol == CoreSymID.NO_SYMBOL:
        raise ValueError("Term root cannot use NO_SYMBOL")

    term_id = int(program.term_count[0])
    if term_id >= get_term_capacity(program):
        raise RuntimeError("No empty term available")

    set_node_symbol(program, term_id, 0, root_symbol)
    program.term_count[0] = term_id + 1
    return term_id


def allocate_node(
    term_id,
    free_nodes,
    free_node_top,
):
    """Allocate one AST node from a term's free-node stack."""
    if free_node_top[term_id] == 0:
        return NO_NODE, False

    free_node_top[term_id] -= 1
    node_id = free_nodes[term_id, free_node_top[term_id]]
    return node_id, True


def clear_child_blocks(
    program: ASTProgram,
    term_id: int,
    node_id: int,
) -> None:
    """Clear a node's two physical links and logical child count."""
    program.node_links[term_id, node_id, 0] = NO_NODE
    program.node_links[term_id, node_id, 1] = NO_NODE
    program.node_arities[term_id, node_id] = 0


def free_node(
    program: ASTProgram,
    term_id: int,
    node_id: int,
) -> None:
    """Clear and return an AST node to the free-node stack."""
    clear_child_blocks(program, term_id, node_id)
    set_node_symbol(program, term_id, node_id, CoreSymID.NO_SYMBOL)
    program.free_node_stack[term_id, program.free_node_count[term_id]] = node_id
    program.free_node_count[term_id] += 1


def _add_direct_child(
    program: ASTProgram,
    term_id: int,
    parent_id: int,
    child_symbol: int,
):
    child_count = int(program.node_arities[term_id, parent_id])
    if child_count >= 2 or program.free_node_count[term_id] == 0:
        return NO_NODE, False

    child_id, ok = allocate_node(
        term_id,
        program.free_node_stack,
        program.free_node_count,
    )
    if not ok:
        return NO_NODE, False

    set_node_symbol(program, term_id, child_id, child_symbol)
    program.node_links[term_id, parent_id, child_count] = child_id
    program.node_arities[term_id, parent_id] += 1
    return child_id, True


def _add_argument_child(
    program: ASTProgram,
    term_id: int,
    parent_id: int,
    child_symbol: int,
):
    """Append one logical argument using a binary ARGS cell."""
    if program.free_node_count[term_id] < 2:
        return NO_NODE, False

    arg_cell, ok = allocate_node(
        term_id,
        program.free_node_stack,
        program.free_node_count,
    )
    if not ok:
        return NO_NODE, False

    child_id, ok = allocate_node(
        term_id,
        program.free_node_stack,
        program.free_node_count,
    )
    if not ok:
        # Defensive rollback. The pre-check above makes this unreachable unless
        # allocator bookkeeping is already inconsistent.
        program.free_node_stack[
            term_id,
            program.free_node_count[term_id],
        ] = arg_cell
        program.free_node_count[term_id] += 1
        return NO_NODE, False

    set_node_symbol(program, term_id, arg_cell, CoreSymID.ARGS)
    set_node_symbol(program, term_id, child_id, child_symbol)
    program.node_links[term_id, arg_cell, 0] = child_id
    program.node_arities[term_id, arg_cell] = 1

    logical_count = int(program.node_arities[term_id, parent_id])
    if logical_count == 0:
        program.node_links[term_id, parent_id, 0] = arg_cell
        program.node_links[term_id, parent_id, 1] = arg_cell
    else:
        tail = program.node_links[term_id, parent_id, 1]
        if tail == NO_NODE:
            return NO_NODE, False
        program.node_links[term_id, tail, 1] = arg_cell
        program.node_arities[term_id, tail] = 2
        program.node_links[term_id, parent_id, 1] = arg_cell

    program.node_arities[term_id, parent_id] += 1
    return child_id, True


def add_child_node(
    program: ASTProgram,
    term_id,
    parent_id,
    child_symbol,
):
    """Append one logical child while keeping physical fan-out at two.

    User functors store arguments in internal ``ARGS`` cells. Compiler control
    nodes (CLAUSE, QUERY, CONJUNCTION and NAF) use their direct binary links.
    """
    if child_symbol == CoreSymID.NO_SYMBOL:
        return NO_NODE, False

    parent_symbol = get_node_symbol(program, term_id, parent_id)
    if is_user_symbol(program, parent_symbol):
        return _add_argument_child(
            program,
            term_id,
            parent_id,
            child_symbol,
        )

    return _add_direct_child(
        program,
        term_id,
        parent_id,
        child_symbol,
    )


def _delete_physical_subtree(
    term_id: int,
    root_id: int,
    program: ASTProgram,
    include_root: bool,
) -> None:
    """Delete reachable physical nodes without recursive Numba calls."""
    max_nodes = program.node_symbols.shape[1]
    stack = np.empty(max_nodes, dtype=np.uint16)
    marked = np.zeros(max_nodes, dtype=np.uint8)
    top = 0

    if include_root:
        if root_id != 0:
            stack[top] = root_id
            top += 1
    else:
        for slot in range(program.node_links.shape[2]):
            child_id = program.node_links[term_id, root_id, slot]
            if child_id != NO_NODE and child_id != 0 and marked[child_id] == 0:
                stack[top] = child_id
                top += 1
                marked[child_id] = 1

    while top > 0:
        top -= 1
        node_id = stack[top]
        marked[node_id] = 1

        for slot in range(program.node_links.shape[2]):
            child_id = program.node_links[term_id, node_id, slot]
            if child_id == NO_NODE or child_id == 0 or marked[child_id] != 0:
                continue
            stack[top] = child_id
            top += 1
            marked[child_id] = 1

    for node_id in range(1, max_nodes):
        if marked[node_id] != 0:
            free_node(program, term_id, node_id)


def remove_descendants(
    term_id,
    parent_id,
    program: ASTProgram,
) -> None:
    """Remove all logical descendants of ``parent_id`` while keeping it."""
    _delete_physical_subtree(term_id, parent_id, program, False)
    clear_child_blocks(program, term_id, parent_id)


def dfs_delete(
    term_id,
    node_id,
    program: ASTProgram,
) -> None:
    """Recursively delete a logical subtree and its internal ARGS cells."""
    _delete_physical_subtree(term_id, node_id, program, True)
