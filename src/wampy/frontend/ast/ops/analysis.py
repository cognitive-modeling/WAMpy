"""Read-only queries and traversal helpers for the array-backed AST."""

import numpy as np

from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID


def get_term_capacity(program: ASTProgram) -> int:
    """Return the number of term slots in the AST."""
    return program.node_symbols.shape[0]


def get_term_count(program: ASTProgram) -> int:
    """Return the number of allocated terms in the AST's contiguous prefix."""
    return int(program.term_count[0])


def get_node_capacity(program: ASTProgram) -> int:
    """Return the maximum number of AST nodes per term."""
    return program.node_symbols.shape[1]


def get_node_symbol(program: ASTProgram, term_id: int, node_id: int):
    """Return the symbol stored at one AST node."""
    return program.node_symbols[term_id, node_id]


def get_symbol_dtype(program: ASTProgram):
    """Return the dtype used for AST symbols."""
    return program.node_symbols.dtype


def get_max_symbol_id(program: ASTProgram) -> int:
    """Return the largest symbol ID representable by this AST."""
    return int(np.iinfo(program.node_symbols.dtype).max)


def get_variable_capacity(program: ASTProgram) -> int:
    """Return the number of reserved variable symbol IDs."""
    return program.first_user_symbol_id - program.first_variable_symbol_id


def get_variable_symbol_id(program: ASTProgram, variable_index: int) -> int:
    """Return the symbol ID reserved for a variable index."""
    return program.first_variable_symbol_id + variable_index


def is_variable_symbol(program: ASTProgram, symbol_id: int) -> bool:
    return program.first_variable_symbol_id <= symbol_id < program.first_user_symbol_id


def is_user_symbol(program: ASTProgram, symbol_id: int) -> bool:
    """Return whether a symbol belongs to the user-symbol range."""
    return symbol_id >= program.first_user_symbol_id


def variable_index(program: ASTProgram, symbol_id: int) -> int:
    return symbol_id - program.first_variable_symbol_id


def is_compound_node(program, tid, node):
    return get_child_count(program, tid, node) > 0


def is_term_empty(program: ASTProgram, term_id: int) -> bool:
    """Return whether a term slot contains no AST term."""
    return get_node_symbol(program, term_id, 0) == CoreSymID.NO_SYMBOL


def get_clause_head_body(program: ASTProgram, tid):
    if get_node_symbol(program, tid, 0) != CoreSymID.CLAUSE:
        return NO_NODE, NO_NODE

    head = get_child_at(program, tid, 0, 0)
    body = get_child_at(program, tid, 0, 1)

    if head == NO_NODE:
        return NO_NODE, NO_NODE

    if get_node_symbol(program, tid, head) == CoreSymID.TRUE:
        return NO_NODE, NO_NODE

    return head, body


def get_child_count(
    program: ASTProgram,
    tid: int,
    node: int,
) -> int:
    """Return the logical number of children of ``node``."""
    return int(program.node_arities[tid, node])


def get_child_at(
    program: ASTProgram,
    term_id,
    node_id,
    index,
):
    """Return a logical child at ``index``, hiding internal ARGS cells."""
    if index < 0:
        return NO_NODE

    count = int(program.node_arities[term_id, node_id])
    if index >= count:
        return NO_NODE

    symbol = program.node_symbols[term_id, node_id]
    if is_user_symbol(program, symbol):
        # Parent link 1 caches the last ARGS cell, making the common final
        # argument lookup O(1) while preserving a simple forward chain.
        if index == count - 1:
            cell = program.node_links[term_id, node_id, 1]
        else:
            cell = program.node_links[term_id, node_id, 0]
            for _ in range(index):
                if cell == NO_NODE:
                    return NO_NODE
                cell = program.node_links[term_id, cell, 1]
        if cell == NO_NODE:
            return NO_NODE
        return program.node_links[term_id, cell, 0]

    if index >= 2:
        return NO_NODE
    return program.node_links[term_id, node_id, index]


def get_first_child(program, term_id, node_id):
    """Return a node's first logical child, or ``NO_NODE``."""
    return get_child_at(program, term_id, node_id, 0)


def get_query_predicate(program: ASTProgram, tid):
    """Return the predicate node below a query root, or ``NO_NODE``."""
    return get_child_at(program, tid, 0, 0)


def get_query_functor(program: ASTProgram, tid):
    """Return the functor symbol of the selected query."""
    predicate = get_query_predicate(program, tid)
    if predicate == NO_NODE:
        return get_node_symbol(program, tid, 0) * 0
    return get_node_symbol(program, tid, predicate)


def get_query_arity(program: ASTProgram, tid):
    """Return the number of arguments of the selected query."""
    predicate = get_query_predicate(program, tid)
    if predicate == NO_NODE:
        return 0
    return get_child_count(program, tid, predicate)


def get_query_argument(program: ASTProgram, tid, argument_index):
    """Return a selected query argument node, or ``NO_NODE``."""
    predicate = get_query_predicate(program, tid)
    if predicate == NO_NODE:
        return NO_NODE
    return get_child_at(program, tid, predicate, argument_index)


def is_node_allocated(program: ASTProgram, term_id: int, node_id: int) -> bool:
    """Return whether ``node_id`` is currently allocated in ``term_id``."""
    if node_id == 0:
        return True

    top = int(program.free_node_count[term_id])
    for i in range(top):
        if program.free_node_stack[term_id, i] == node_id:
            return False
    return True


def collect_head_nodes(program: ASTProgram, tid, head_root, out):
    stack = np.empty(get_node_capacity(program), dtype=np.int32)
    top = 0
    stack[top] = int(head_root)
    top += 1

    n = 0
    while top > 0:
        top -= 1
        node = stack[top]
        sym = get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION:
            left = get_child_at(program, tid, node, 0)
            right = get_child_at(program, tid, node, 1)
            if right != NO_NODE and top < stack.shape[0]:
                stack[top] = int(right)
                top += 1
            if left != NO_NODE and top < stack.shape[0]:
                stack[top] = int(left)
                top += 1
        elif sym == CoreSymID.TRUE:
            continue
        else:
            if n < out.shape[0]:
                out[n] = node
                n += 1
            else:
                break

    return n


def _count_clause_shape(program: ASTProgram):
    heads_buf = np.empty(get_node_capacity(program), dtype=np.int32)
    max_fun = -1
    max_arity = -1

    for tid in range(get_term_count(program)):
        head_root, _ = get_clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = collect_head_nodes(program, tid, head_root, heads_buf)
        for j in range(n_heads):
            head = heads_buf[j]
            fun = int(get_node_symbol(program, tid, head))
            arity = get_child_count(program, tid, head)
            if fun > max_fun:
                max_fun = fun
            if arity > max_arity:
                max_arity = arity

    return max_fun + 1, max_arity + 1


def count_clauses(program: ASTProgram):
    """Return a dense ``counts[functor_id, arity]`` table for clauses."""
    n_functors, n_arities = _count_clause_shape(program)
    counts = np.zeros((n_functors, n_arities), dtype=np.int32)
    heads_buf = np.empty(get_node_capacity(program), dtype=np.int32)

    for tid in range(get_term_count(program)):
        head_root, _ = get_clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = collect_head_nodes(program, tid, head_root, heads_buf)
        for j in range(n_heads):
            head = heads_buf[j]
            fun = int(get_node_symbol(program, tid, head))
            arity = get_child_count(program, tid, head)
            counts[fun, arity] += 1

    return counts


def terms_have_equal_structure(program: ASTProgram, term_a: int, term_b: int) -> bool:
    max_nodes = get_node_capacity(program)
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

        if get_node_symbol(program, term_a, node_a) != get_node_symbol(program, term_b, node_b):
            return False

        count_a = get_child_count(program, term_a, node_a)
        count_b = get_child_count(program, term_b, node_b)
        if count_a != count_b:
            return False
        if top + count_a > max_nodes:
            return False

        for child_index in range(count_a):
            child_a = get_child_at(program, term_a, node_a, child_index)
            child_b = get_child_at(program, term_b, node_b, child_index)
            if child_a == NO_NODE or child_b == NO_NODE:
                return False
            stack_a[top] = child_a
            stack_b[top] = child_b
            top += 1

    return True


def count_nodes(program: ASTProgram) -> np.int64:
    """Count allocated physical nodes, including internal ARGS cells."""
    max_terms_nodes = get_node_capacity(program)
    n_terms = get_term_count(program)
    total = np.int64(0)
    for term_id in range(n_terms):
        total += np.int64(max_terms_nodes) - np.int64(program.free_node_count[term_id])
    return total


def _count_clause_var_nodes(program: ASTProgram):
    """Count variable nodes in clause terms."""
    n_vars = np.int64(0)
    for term_id in range(get_term_count(program)):
        if get_node_symbol(program, term_id, 0) != np.uint16(CoreSymID.CLAUSE.value):
            continue
        for node_id in range(get_node_capacity(program)):
            sym = get_node_symbol(program, term_id, node_id)
            if is_variable_symbol(program, sym):
                n_vars += np.int64(1)
    return n_vars


def get_user_symbol_id(
    program: ASTProgram,
    user_index: int,
) -> int:
    """Return the symbol ID for a user-symbol index."""
    return program.first_user_symbol_id + user_index
