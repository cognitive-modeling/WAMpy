"""Execute queries against compiled WAM programs.

Provides APIs for textual and pre-parsed queries, including answer collection
and runtime status handling.
"""

import numpy as np
from numba import jit

from wampy.config import DEFAULT_CONFIG, WAMConfig
from wampy.status import WAMStatus
from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.answers import Answers, decode_answer, init_answers
from wampy.frontend.ast.goals import Goals
import wampy.ast as ast
from wampy.frontend.parser import build_query, SymbolTable
from wampy.compiler.compiler import (
    CodeArea,
    ENTRY_INVALID_PC,
    OP,
    compile_negated_goal,
    emit,
)
from wampy.runtime.interpreter import (
    Machine,
    init_machine,
    redo,
    run,
)
from wampy.runtime.stack import (
    Stack,
    TAG,
    deref,
    init_stack,
    make_var,
    make_const,
    make_functor,
    make_structure,
)

VAR_LAST_PLUS1 = np.int32(int(SymID.VAR_LAST) + 1)
SENTINEL = np.uint16(65535)
NONE = np.uint16(65535)


def query_from_str(
    code_area: CodeArea,
    query_str: str,
    symbol_table: SymbolTable,
    config: WAMConfig = DEFAULT_CONFIG,
) -> list[str]:
    """Parse and execute a textual query."""

    goals = build_query(query_str, symbol_table, config)
    stack = init_stack(config)
    answers = init_answers(
        config.solver.answer_max_answers,
        config.solver.answer_max_nodes,
        config.ast.num_terms,
    )
    machine = init_machine(
        code_area.code,
        code_area.entry,
        stack,
    )

    nsol, st = query(code_area, machine, goals, stack, answers, config)
    result = _build_result(query_str, st, nsol, answers, symbol_table)
    return result["answers"]


@jit(cache=False)
def query(
    code_area: CodeArea,
    machine: Machine,
    goals: Goals,
    stack: Stack,
    answers: Answers,
    config: WAMConfig = DEFAULT_CONFIG,
) -> tuple[np.int32, np.int64]:
    """Execute an already-parsed query."""

    answer_rows = answers.rows
    term_usage = answers.term_usage
    q_fun = goals.fun
    q_arity = goals.arity
    q_arg_nodes = goals.args
    q_prog = goals.program
    q_term_id = goals.term_id

    trace_enabled = config.solver.trace
    machine.trace_enabled[0] = trace_enabled

    top_level_query_pc = np.int32(ENTRY_INVALID_PC)
    if q_fun == SymID.TRUE or q_fun == SymID.NOT_PROVABLE:
        top_level_query_pc, _ = compile_top_level_control_query(code_area, goals)

    args = np.empty(q_arity, dtype=np.uint16)
    var_env = np.full(VAR_LAST_PLUS1, SENTINEL, dtype=np.uint16)

    # Build full heap terms for each query argument (supports nested structures)
    try:
        for i in range(q_arity):
            node = q_arg_nodes[i]
            addr, st_build = build_heap_term_from_query_ast(q_prog, q_term_id, node, stack, var_env)
            if st_build != WAMStatus.SUCCESS.value:
                return np.int32(0), st_build

            args[i] = np.uint16(addr)
            machine.X[i] = np.uint16(addr)
    except Exception:
        return np.int32(0), np.int64(WAMStatus.STACK_OVERFLOW.value)

    machine.cp[0] = -1
    try:
        st = run_top_level_query(machine, goals, top_level_query_pc)
    except Exception:
        return np.int32(0), np.int64(WAMStatus.STACK_OVERFLOW)

    if st != WAMStatus.SUCCESS:
        return np.int32(0), st

    n_answers = 0
    max_answers = answer_rows.shape[0]
    while True:
        ln, st2 = snapshot_answers(stack, q_fun, args, answer_rows, n_answers)
        if st2 != WAMStatus.SUCCESS.value:
            return np.int32(n_answers), st2

        if trace_enabled == 1:
            collect_path_usage(machine, term_usage[n_answers])

        n_answers += 1
        if n_answers >= max_answers:
            return np.int32(n_answers), np.int64(WAMStatus.SUCCESS.value)

        try:
            st = redo(machine)
        except Exception:
            return np.int32(n_answers), np.int64(WAMStatus.STACK_OVERFLOW.value)
        if st == WAMStatus.SUCCESS.value:
            continue

        return np.int32(n_answers), st


def _build_result(query_src: str, st, nsol, answers: Answers, symbol_table: SymbolTable):
    answers_decoded = [
        decode_answer(
            answers.rows[i],
            symbol_table,
        )
        for i in range(int(nsol))
    ]

    if st == WAMStatus.SUCCESS or st == WAMStatus.EXHAUSTED:
        per_answer_usage = [answers.term_usage[i].tolist() for i in range(int(nsol))]
        return {
            "query": query_src,
            "status": st,
            "n_answers": int(nsol),
            "answers": answers_decoded,
            "term_usage": per_answer_usage,
            "answer_term_usage": per_answer_usage,
        }

    raise RuntimeError(f"WAM error: {WAMStatus(st).name}")


@jit(cache=False)
def build_heap_term_from_query_ast(program, term_id, root_node, stack, var_env):
    """
    Convert a query AST subtree into WAM heap terms using your runtime stack layout.

    Layout must match unify/read_term_view:
      - STR cell at addr points to functor header at heap[addr]
      - functor header: heap[fun] = symbol_id, heap[fun+1] = arity
      - STR arguments are stored at addr+1 .. addr+arity as REF cells pointing to
        argument term roots (or vars/const/struct roots).
    """
    symbols = program.symbol
    children = program.children

    max_nodes_total = symbols.shape[1]
    max_child = children.shape[2]

    # Work stacks for traversal (bounded by max_nodes_total)
    dfs_stack = np.empty(max_nodes_total, dtype=np.uint16)
    order = np.empty(max_nodes_total, dtype=np.uint16)
    out_addr = np.full(max_nodes_total, NONE, dtype=np.uint16)

    sp = 0
    ord_n = 0

    dfs_stack[sp] = np.uint16(root_node)
    sp += 1

    # Preorder collect
    while sp > 0:
        sp -= 1
        n = dfs_stack[sp]
        order[ord_n] = n
        ord_n += 1
        if ord_n >= max_nodes_total:
            return np.uint16(0), np.int64(WAMStatus.STACK_OVERFLOW)

        # push children (reverse for stable left-to-right)
        for j in range(max_child - 1, -1, -1):
            ch = children[term_id, n, j]
            if ch == NONE:
                continue
            dfs_stack[sp] = np.uint16(ch)
            sp += 1
            if sp >= max_nodes_total:
                return np.uint16(0), np.int64(WAMStatus.STACK_OVERFLOW)

    # Process in reverse (postorder) to build bottom-up
    for idx in range(ord_n - 1, -1, -1):
        n = order[idx]
        sym = np.int32(symbols[term_id, n])

        # Variable
        if ast.is_sym_a_var(sym):
            if sym < 0 or sym >= VAR_LAST_PLUS1:
                return np.uint16(0), np.int64(WAMStatus.INVALID_QUERY)

            v = var_env[sym]
            if v == SENTINEL:
                v = make_var(stack)
                var_env[sym] = v
            out_addr[n] = np.uint16(v)
            continue

        # Non-variable: decide CONST vs STRUCT by child count
        ar = 0
        for j in range(max_child):
            ch = children[term_id, n, j]
            if ch == NONE:
                break
            ar += 1

        if ar == 0:
            # CONST
            c = make_const(stack, np.uint16(sym))
            out_addr[n] = np.uint16(c)
            continue

        # STRUCT
        fun_pos = make_functor(stack, np.uint16(sym), np.uint16(ar))

        args = np.empty(ar, dtype=np.uint16)
        for j in range(ar):
            ch = children[term_id, n, j]
            args[j] = out_addr[ch]

        s = make_structure(stack, fun_pos, args)
        out_addr[n] = np.uint16(s)

    return out_addr[np.uint16(root_node)], np.int64(WAMStatus.SUCCESS)


@jit(cache=False)
def collect_path_usage(machine, usage_row):
    for i in range(usage_row.shape[0]):
        usage_row[i] = 0

    depth = int(machine.path_depth[0])
    for i in range(depth):
        term_id = machine.path_terms[i]
        if 0 <= term_id < usage_row.shape[0]:
            usage_row[term_id] += 1


@jit(cache=False)
def snapshot_answers(stack, q_fun, roots, answers, sol_i):
    """
    Writes prefix encoding of the FULL query term:
      q_fun(roots[0], ..., roots[q_arity-1])
    into answers[sol_i].

    Returns (length:uint16, status:int64).
    """
    ln, st = encode_query_term_prefix(
        stack,
        q_fun,
        roots,
        answers[sol_i]["symbols"],
        answers[sol_i]["tags"],
        answers[sol_i]["aritys"],
    )
    if st != WAMStatus.SUCCESS:
        return np.uint16(0), st

    answers[sol_i]["len"] = ln
    return ln, st


@jit(cache=False)
def encode_query_term_prefix(stack, q_fun, roots, out_symbols, out_tags, out_aritys):
    """
    Prefix-encode a synthetic STRUCT root for the predicate functor, followed by
    prefix encodings of each argument term from the WAM heap.

    Root node:
      TAG.STR, symbol = q_fun, arity = len(roots)
    Then children are roots[0..] in order.
    """
    max_nodes = out_symbols.shape[0]
    q_arity = int(roots.shape[0])

    if max_nodes < 1:
        return np.uint16(0), np.int64(WAMStatus.HEAP_OVERFLOW)

    # Synthetic root: q_fun / q_arity
    out_tags[0] = np.uint8(TAG.STR)
    out_symbols[0] = np.int64(q_fun)
    out_aritys[0] = np.uint16(q_arity)

    work = np.empty(max_nodes, dtype=np.uint16)
    sp = 0

    for i in range(q_arity - 1, -1, -1):
        if sp >= max_nodes:
            return np.uint16(1), np.int64(WAMStatus.ENVIRONMENT_OVERFLOW)
        work[sp] = np.uint16(roots[i])
        sp += 1

    n = 1
    while sp > 0:
        sp -= 1
        a = deref(stack, work[sp])
        if a >= stack.heap_top[0]:
            return np.uint16(n), np.int64(WAMStatus.CORRUPT_ENVIRONMENT)

        if n >= max_nodes:
            return np.uint16(n), np.int64(WAMStatus.HEAP_OVERFLOW)

        tag = stack.tags[a]

        if tag == TAG.REF:
            out_tags[n] = np.uint8(TAG.REF)
            out_symbols[n] = np.int64(a)  # var id = heap address
            out_aritys[n] = np.uint16(0)
            n += 1
            continue

        if tag == TAG.CON:
            out_tags[n] = np.uint8(TAG.CON)
            out_symbols[n] = np.int64(stack.heap[a])  # const symbol id
            out_aritys[n] = np.uint16(0)
            n += 1
            continue

        if tag == TAG.STR:
            fun = stack.heap[a]
            symbol_id = stack.heap[fun]
            ar = stack.heap[fun + 1]

            out_tags[n] = np.uint8(TAG.STR)
            out_symbols[n] = np.int64(symbol_id)
            out_aritys[n] = np.uint16(ar)
            n += 1

            base = a + 1
            ar_i = int(ar)

            for i in range(ar_i - 1, -1, -1):
                if sp >= max_nodes:
                    return np.uint16(n), np.int64(WAMStatus.ENVIRONMENT_OVERFLOW)
                work[sp] = np.uint16(base + i)
                sp += 1

            continue

        return np.uint16(n), np.int64(WAMStatus.CORRUPT_ENVIRONMENT)

    return np.uint16(n), np.int64(WAMStatus.SUCCESS)


@jit(cache=False)
def compile_top_level_control_query(code_area: CodeArea, query: Goals):
    original_pc = code_area.pc.value

    if query.fun == SymID.TRUE and query.arity == 0:
        code_area.pc.value = original_pc
        code_area = emit(code_area, OP.PROCEED)
        query_code_size = code_area.pc.value
        code_area.pc.value = original_pc
        return np.int32(original_pc), np.int32(query_code_size)

    if query.fun != SymID.NOT_PROVABLE:
        return np.int32(ENTRY_INVALID_PC), np.int32(original_pc)

    program = query.program
    term_id = query.term_id
    children = program.children[term_id, 0]
    not_node = np.int64(ENTRY_INVALID_PC)

    for slot in range(children.shape[0]):
        c = children[slot]
        if c == NONE:
            continue
        not_node = np.int64(c)
        break

    if not_node < 0:
        return np.int32(ENTRY_INVALID_PC), np.int32(original_pc)

    var_table = np.full(VAR_LAST_PLUS1, -1, dtype=np.int16)
    code_area.pc.value = original_pc
    code_area, _ = compile_negated_goal(code_area, program, term_id, not_node, var_table, 0)
    code_area = emit(code_area, OP.PROCEED)
    query_code_size = code_area.pc.value
    code_area.pc.value = original_pc
    return np.int32(original_pc), np.int32(query_code_size)


@jit(cache=False)
def run_top_level_query(machine, query, top_level_query_pc):
    q_fun = query.fun
    q_arity = query.arity

    if q_fun == SymID.TRUE or q_fun == SymID.NOT_PROVABLE:
        if q_fun == SymID.TRUE and q_arity != 0:
            return np.int64(WAMStatus.INVALID_QUERY)
        if q_fun == SymID.NOT_PROVABLE and q_arity != 1:
            return np.int64(WAMStatus.INVALID_QUERY)
        if top_level_query_pc < 0:
            return np.int64(WAMStatus.INVALID_QUERY)
        machine.pc[0] = top_level_query_pc
        try:
            return run(machine)
        except Exception:
            return np.int64(WAMStatus.STACK_OVERFLOW)

    machine.pc[0] = machine.entry[q_fun, q_arity]
    try:
        return run(machine)
    except Exception:
        return np.int64(WAMStatus.STACK_OVERFLOW)
