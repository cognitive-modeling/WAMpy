"""Structural and semantic checks for the array-backed AST."""

import numpy as np

from wampy.frontend.ast.ops.analysis import (
    get_child_count,
    get_node_capacity,
    get_node_symbol,
    get_term_capacity,
    get_term_count,
    is_user_symbol,
    is_variable_symbol,
)
from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID


def is_valid_symbol_id(program: ASTProgram, symbol_id) -> bool:
    """Return whether ``symbol_id`` is a defined AST symbol identifier."""
    if symbol_id == CoreSymID.NO_SYMBOL:
        return True
    if (
        symbol_id == CoreSymID.TRUE
        or symbol_id == CoreSymID.ARGS
        or symbol_id == CoreSymID.NEGATION_AS_FAILURE
        or symbol_id == CoreSymID.CONJUNCTION
        or symbol_id == CoreSymID.CLAUSE
        or symbol_id == CoreSymID.QUERY
        or symbol_id == CoreSymID.CUT
    ):
        return True
    if is_variable_symbol(program, symbol_id):
        return True
    return is_user_symbol(program, symbol_id)


def _valid_direct_links(program: ASTProgram, term_id: int, node_id: int, count: int) -> bool:
    if count < 0 or count > 2:
        return False
    for slot in range(2):
        child = program.node_links[term_id, node_id, slot]
        if slot < count:
            if child == NO_NODE:
                return False
        elif child != NO_NODE:
            return False
    return True


def _valid_argument_chain(
    program: ASTProgram,
    term_id: int,
    node_id: int,
    logical_count: int,
    free,
) -> bool:
    first = program.node_links[term_id, node_id, 0]
    tail = program.node_links[term_id, node_id, 1]
    if logical_count == 0:
        return first == NO_NODE and tail == NO_NODE
    if first == NO_NODE or tail == NO_NODE:
        return False

    node_capacity = get_node_capacity(program)
    cell = first
    last = NO_NODE
    for index in range(logical_count):
        cell_id = int(cell)
        if cell_id <= 0 or cell_id >= node_capacity or free[cell_id] != 0:
            return False
        if get_node_symbol(program, term_id, cell_id) != CoreSymID.ARGS:
            return False

        value = program.node_links[term_id, cell_id, 0]
        if value == NO_NODE:
            return False
        value_id = int(value)
        if value_id <= 0 or value_id >= node_capacity or free[value_id] != 0:
            return False

        next_cell = program.node_links[term_id, cell_id, 1]
        expected_physical_count = 1 if index == logical_count - 1 else 2
        if int(program.node_arities[term_id, cell_id]) != expected_physical_count:
            return False
        if index == logical_count - 1:
            if next_cell != NO_NODE:
                return False
        elif next_cell == NO_NODE:
            return False

        last = cell
        cell = next_cell

    return last == tail


def is_structurally_valid_term(program: ASTProgram, term_id: int) -> bool:
    """Check binary-link bookkeeping and tree reachability for one term."""
    if term_id < 0 or term_id >= get_term_capacity(program):
        return False

    term_capacity = get_term_capacity(program)
    node_capacity = get_node_capacity(program)
    if node_capacity < 1:
        return False
    if (
        program.node_links.ndim != 3
        or program.node_arities.ndim != 2
        or program.free_node_stack.ndim != 2
        or program.free_node_count.ndim != 1
    ):
        return False
    if program.node_links.shape != (term_capacity, node_capacity, 2):
        return False
    if program.node_arities.shape != (term_capacity, node_capacity):
        return False
    if program.free_node_stack.shape != (term_capacity, node_capacity - 1):
        return False
    if program.free_node_count.shape[0] != term_capacity:
        return False

    free_top = int(program.free_node_count[term_id])
    if free_top < 0 or free_top > node_capacity - 1:
        return False

    free = np.zeros(node_capacity, dtype=np.uint8)
    for stack_index in range(free_top):
        node_id = int(program.free_node_stack[term_id, stack_index])
        if node_id <= 0 or node_id >= node_capacity or free[node_id] != 0:
            return False
        free[node_id] = 1

    for node_id in range(node_capacity):
        symbol_id = get_node_symbol(program, term_id, node_id)
        count = int(program.node_arities[term_id, node_id])

        if free[node_id] != 0:
            if symbol_id != CoreSymID.NO_SYMBOL or count != 0:
                return False
            if (
                program.node_links[term_id, node_id, 0] != NO_NODE
                or program.node_links[term_id, node_id, 1] != NO_NODE
            ):
                return False
            continue

        if not is_valid_symbol_id(program, symbol_id):
            return False

        if is_user_symbol(program, symbol_id):
            if not _valid_argument_chain(program, term_id, node_id, count, free):
                return False
        elif not _valid_direct_links(program, term_id, node_id, count):
            return False

    visited = np.zeros(node_capacity, dtype=np.uint8)
    stack = np.empty(node_capacity, dtype=np.uint16)
    stack_size = 1
    stack[0] = 0
    visited[0] = 1
    reachable = 1

    while stack_size > 0:
        stack_size -= 1
        node_id = int(stack[stack_size])
        symbol_id = get_node_symbol(program, term_id, node_id)

        physical_count = int(program.node_arities[term_id, node_id])
        if is_user_symbol(program, symbol_id):
            # A user node reaches its arguments through the first ARGS cell;
            # the second parent link is only the cached tail pointer.
            physical_count = 1 if physical_count > 0 else 0

        for slot in range(physical_count):
            child_id = int(program.node_links[term_id, node_id, slot])
            if child_id == NO_NODE:
                return False
            if child_id <= 0 or child_id >= node_capacity:
                return False
            if free[child_id] != 0 or visited[child_id] != 0:
                return False
            if reachable >= node_capacity:
                return False
            visited[child_id] = 1
            reachable += 1
            stack[stack_size] = np.uint16(child_id)
            stack_size += 1

    return reachable == node_capacity - free_top


def is_semantically_valid_term(program: ASTProgram, term_id: int) -> bool:
    """Check AST grammar constraints for one structurally valid term."""
    if not is_structurally_valid_term(program, term_id):
        return False

    node_capacity = get_node_capacity(program)
    for node_id in range(node_capacity):
        if not _is_allocated(program, term_id, node_id):
            continue
        symbol_id = get_node_symbol(program, term_id, node_id)
        arity = get_child_count(program, term_id, node_id)

        if symbol_id == CoreSymID.NO_SYMBOL:
            if node_id != 0 or arity != 0:
                return False
        elif symbol_id == CoreSymID.TRUE:
            if arity != 0:
                return False
        elif symbol_id == CoreSymID.ARGS:
            if node_id == 0 or arity < 1 or arity > 2:
                return False
        elif symbol_id == CoreSymID.NEGATION_AS_FAILURE:
            if arity != 1:
                return False
        elif symbol_id == CoreSymID.CUT:
            if arity != 0:
                return False
        elif symbol_id == CoreSymID.CONJUNCTION:
            if arity != 2:
                return False
        elif symbol_id == CoreSymID.CLAUSE:
            if node_id != 0 or arity < 1 or arity > 2:
                return False
        elif symbol_id == CoreSymID.QUERY:
            if node_id != 0 or arity != 1:
                return False
        elif is_variable_symbol(program, symbol_id) and arity != 0:
            return False

    return get_node_symbol(program, term_id, 0) != CoreSymID.ARGS


def _is_allocated(program: ASTProgram, term_id: int, node_id: int) -> bool:
    if node_id == 0:
        return True
    free_top = int(program.free_node_count[term_id])
    for index in range(free_top):
        if int(program.free_node_stack[term_id, index]) == node_id:
            return False
    return True


def is_valid_term(program: ASTProgram, term_id: int) -> bool:
    """Return whether one term passes structural and semantic checks."""
    return is_semantically_valid_term(program, term_id)


def is_valid_program(program: ASTProgram) -> bool:
    """Return whether every allocated term in ``program`` is valid."""
    for term_id in range(get_term_count(program)):
        if not is_valid_term(program, term_id):
            return False
    return True


def validate_term(program: ASTProgram, term_id: int) -> bool:
    """Compatibility spelling for :func:`is_valid_term`."""
    return is_valid_term(program, term_id)


def validate_program(program: ASTProgram) -> bool:
    """Compatibility spelling for :func:`is_valid_program`."""
    return is_valid_program(program)
