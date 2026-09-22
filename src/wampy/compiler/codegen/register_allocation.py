"""Temporary and permanent WAM register allocation."""

from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE

VAR_UNBOUND = -1
Y_UNBOUND_BASE = -2


def allocate_x(next_x, max_x_registers):
    """Reserve the next X register or fail during compilation."""
    if next_x >= max_x_registers:
        raise ValueError("X register capacity exceeded")

    return next_x, next_x + 1


def encode_unbound_y(y):
    """Encode an assigned-but-not-yet-initialized Y slot in a variable table."""

    return Y_UNBOUND_BASE - y


def resolve_variable_register(var_table, var_slot, next_x, max_x_registers):
    """Resolve one logical variable to ``(index, is_y, first, next_x)``."""

    register = var_table[var_slot]
    if register == VAR_UNBOUND:
        register, next_x = allocate_x(next_x, max_x_registers)
        var_table[var_slot] = register
        return register, False, True, next_x

    if register <= Y_UNBOUND_BASE:
        y = Y_UNBOUND_BASE - register
        var_table[var_slot] = max_x_registers + y
        return y, True, True, next_x

    if register >= max_x_registers:
        return register - max_x_registers, True, False, next_x

    return register, False, False, next_x


def record_variable_occurrences(
    program,
    tid,
    root_node,
    phase,
    first,
    last,
    stack,
):
    """Record variables below one term using a caller-provided traversal stack."""

    top = 0
    stack[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        symbol = ast.get_node_symbol(program, tid, node)
        if ast.is_variable_symbol(program, symbol):
            slot = ast.variable_index(program, symbol)
            if first[slot] < 0:
                first[slot] = phase
            last[slot] = phase
            continue

        child_count = ast.get_child_count(program, tid, node)
        for child_index in range(child_count):
            child = ast.get_child_at(program, tid, node, child_index)
            if child == NO_NODE:
                continue
            if top >= stack.shape[0]:
                raise ValueError("Term nesting is too deep")
            stack[top] = child
            top += 1
