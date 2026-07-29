"""Read-only analysis helpers for the array-backed AST."""

__all__ = [
    "is_sym_a_var",
    "is_compound",
    "get_arity_of_node",
    "clause_head_body",
    "collect_head_nodes",
    "get_count_clauses",
    "is_structure_of_terms_equal",
]

import numpy as np
from numba import jit

from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.ast.program import NONE, Program

NO_NODE = -1


@jit(cache=True)
def is_sym_a_var(sym):
    return SymID.VAR_FIRST <= sym <= SymID.VAR_LAST


@jit(cache=True)
def is_compound(program, tid, node):
    return get_arity_of_node(program, tid, node) > 0


@jit(cache=True)
def clause_head_body(program: Program, tid):
    if program.symbol[tid, 0] != SymID.CLAUSE:
        return NO_NODE, NO_NODE

    children = program.children[tid, 0]

    head = NO_NODE
    body = NO_NODE

    for i in range(children.shape[0]):
        child = children[i]
        if child == NONE:
            continue
        if head == NO_NODE:
            head = int(child)
        elif body == NO_NODE:
            body = int(child)
            break

    if head == NO_NODE:
        return NO_NODE, NO_NODE

    if program.symbol[tid, head] == SymID.TRUE:
        return NO_NODE, NO_NODE

    return head, body


@jit(cache=True)
def get_arity_of_node(program: Program, tid, node):
    arity = 0
    children = program.children[tid, node]
    for i in range(children.shape[0]):
        if children[i] != NONE:
            arity += 1
    return arity


@jit(cache=True)
def collect_head_nodes(program: Program, tid, head_root, out):
    stack = np.empty(program.symbol.shape[1], dtype=np.int32)
    top = 0
    stack[top] = int(head_root)
    top += 1

    n = 0
    while top > 0:
        top -= 1
        node = stack[top]

        sym = program.symbol[tid, node]

        if sym == SymID.AND:
            children = program.children[tid, node]
            left = NONE
            right = NONE
            for child in children:
                if child == NONE:
                    continue
                if left == NONE:
                    left = child
                else:
                    right = child
                    break

            if right != NONE and top < stack.shape[0]:
                stack[top] = int(right)
                top += 1
            if left != NONE and top < stack.shape[0]:
                stack[top] = int(left)
                top += 1

        elif sym == SymID.TRUE:
            continue

        else:
            if n < out.shape[0]:
                out[n] = node
                n += 1
            else:
                break

    return n


@jit(cache=True)
def _count_clause_shape(program: Program):
    heads_buf = np.empty(program.symbol.shape[1], dtype=np.int32)
    max_fun = -1
    max_arity = -1

    for tid in range(program.symbol.shape[0]):
        head_root, _ = clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = collect_head_nodes(program, tid, head_root, heads_buf)
        for j in range(n_heads):
            head = heads_buf[j]
            fun = int(program.symbol[tid, head])
            arity = get_arity_of_node(program, tid, head)

            if fun > max_fun:
                max_fun = fun
            if arity > max_arity:
                max_arity = arity

    return max_fun + 1, max_arity + 1


@jit(cache=True)
def get_count_clauses(program: Program):
    """Return a dense ``counts[functor_id, arity]`` table for program clauses."""
    n_functors, n_arities = _count_clause_shape(program)
    counts = np.zeros((n_functors, n_arities), dtype=np.int32)
    heads_buf = np.empty(program.symbol.shape[1], dtype=np.int32)

    for tid in range(program.symbol.shape[0]):
        head_root, _ = clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = collect_head_nodes(program, tid, head_root, heads_buf)
        for j in range(n_heads):
            head = heads_buf[j]
            fun = int(program.symbol[tid, head])
            arity = get_arity_of_node(program, tid, head)
            counts[fun, arity] += 1

    return counts


@jit(cache=True)
def is_structure_of_terms_equal(program: Program, term_a: int, term_b: int) -> bool:
    max_nodes = program.symbol.shape[1]
    max_slots = program.children.shape[2]

    stack_a = np.empty(max_nodes, dtype=np.uint16)
    stack_b = np.empty(max_nodes, dtype=np.uint16)
    top = 0

    stack_a[top] = 0
    stack_b[top] = 0
    top += 1

    while top > 0:
        top -= 1
        node_a = stack_a[top]
        node_b = stack_b[top]

        if program.symbol[term_a, node_a] != program.symbol[term_b, node_b]:
            return False

        for slot in range(max_slots):
            child_a = program.children[term_a, node_a, slot]
            child_b = program.children[term_b, node_b, slot]

            if child_a == NONE and child_b == NONE:
                continue
            if child_a == NONE or child_b == NONE:
                return False

            if top >= max_nodes:
                return False

            stack_a[top] = child_a
            stack_b[top] = child_b
            top += 1

    return True
