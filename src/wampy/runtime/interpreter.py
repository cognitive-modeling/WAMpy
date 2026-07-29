# wam/runtime/interpreter.py
"""
Minimal WAM instruction interpreter.

Supports:
    GET_VAR, GET_VAL, GET_CONST
    PUT_VAR, PUT_VAL, PUT_CONST
    CALL, EXECUTE, PROCEED
    MARK_CUT, CUT, DROP_CUT, FAIL
    TRY, RETRY, TRUST
    JMP_RETRY, JMP_TRUST

No environments (ALLOCATE / DEALLOCATE) yet.
Nested CALLs preserve caller X registers via a small call-frame stack.
"""

from collections import namedtuple

import numpy as np
from numba import int64, jit, uint16, void

from wampy.status import WAMStatus
from wampy.compiler.compiler import OP
from wampy.runtime.stack import (
    TAG,
    make_var,
    make_functor,
    make_structure,
    deref,
)
from wampy.runtime.unify import unify, unify_const, push_struct, pop_struct

# ============================================================
# Machine state
# ============================================================

READ = 0
WRITE = 1

# The interpreter dispatch graph has crashed on cached overload reloads.
# Keep only the simple X-register copy helpers cached and compile the rest fresh.

Machine = namedtuple(
    "Machine",
    [
        "code",
        "code_size",
        "entry",
        "pc",
        "cp",
        "X",
        "stack",
        # --- WAM structure/unification state ---
        "mode",  # READ / WRITE
        "S",  # structure pointer
        "S_stack",  # saved S values
        "mode_stack",  # saved modes
        "st_top",  # structure stack top
        "fuel",  # max steps
        "fail_base",  # cp_top boundary for local failure handling
        "cut_levels",  # cp_top cut boundaries
        "cut_top",  # next free cut boundary slot
        # --- clause usage tracing ---
        "trace_enabled",  # int32[1]
        "clause_entry_term",  # int32[:], entry-pc -> term id or -1
        "path_terms",  # int32[:], term ids in the current proof path
        "path_depth",  # int32[1], current depth in path_terms
        "choice_path_depth",  # int32[:], snapshot path_depth per choice point
        "choice_cut_tops",  # int32[:], snapshot cut_top per choice point
    ],
)


# Machine constructors allocate nested runtime state and are part of the
# uncached interpreter dispatch graph.
@jit(cache=False)
def init_machine(code, entry, stack):
    max_x = stack.choice_X.shape[1]
    max_struct = stack.struct_stack_size
    step_size = stack.step_limit
    trace_enabled = np.zeros(1, dtype=np.int32)
    clause_entry_term = np.full(1, -1, dtype=np.int32)
    path_terms = np.empty(1, dtype=np.int32)
    path_depth = np.zeros(1, dtype=np.int32)
    choice_path_depth = np.zeros(stack.choice_points.shape[0], dtype=np.int32)
    fail_base = np.full(1, -1, dtype=np.int32)
    cut_levels = np.empty(stack.choice_points.shape[0], dtype=np.int32)
    cut_top = np.zeros(1, dtype=np.int32)
    choice_cut_tops = np.zeros(stack.choice_points.shape[0], dtype=np.int32)

    return Machine(
        code=code,
        code_size=code.shape[0],
        entry=entry,
        pc=np.zeros(1, dtype=np.int32),
        cp=np.full(1, -1, dtype=np.int32),
        X=np.zeros(max_x, dtype=np.uint16),
        stack=stack,
        # --- WAM structure/unification state ---
        mode=np.zeros(1, dtype=np.int32),  # READ initially
        S=np.zeros(1, dtype=np.uint16),
        S_stack=np.zeros(max_struct, dtype=np.uint16),
        mode_stack=np.zeros(max_struct, dtype=np.int32),
        st_top=np.zeros(1, dtype=np.int32),
        fuel=np.full(1, step_size, dtype=np.int64),  # set by solve()
        fail_base=fail_base,
        cut_levels=cut_levels,
        cut_top=cut_top,
        trace_enabled=trace_enabled,
        clause_entry_term=clause_entry_term,
        path_terms=path_terms,
        path_depth=path_depth,
        choice_path_depth=choice_path_depth,
        choice_cut_tops=choice_cut_tops,
    )

@jit
def reset_machine(machine, stack):
    machine.pc[0] = 0
    machine.cp[0] = -1
    machine.mode[0] = 0
    machine.S[0] = 0
    machine.st_top[0] = 0
    machine.fuel[0] = stack.step_limit
    machine.fail_base[0] = -1
    machine.cut_top[0] = 0
    machine.path_depth[0] = 0


# ============================================================
# Failure handling
# ============================================================


@jit(cache=False)
def redo(m) -> WAMStatus:
    status = _fail(m)
    if status != WAMStatus.SUCCESS:
        return status
    return run(m)


@jit(cache=False)
def _fail(m) -> WAMStatus:
    stack = m.stack

    if stack.cp_top[0] <= m.fail_base[0]:
        return WAMStatus.EXHAUSTED

    if stack.cp_top[0] == 0:
        return WAMStatus.EXHAUSTED

    i = stack.cp_top[0] - 1

    # restore argument registers for the predicate we're retrying
    copy_choice_to_X(m.X, stack.choice_X, i)

    # restore trail
    target = stack.choice_points[i]
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = stack.choice_heap_tops[i]

    # restore call-frame stack
    stack.call_top[0] = stack.choice_call_tops[i]
    stack.call_alloc_top[0] = stack.choice_call_alloc_tops[i]

    m.cut_top[0] = m.choice_cut_tops[i]

    # restore CP
    m.cp[0] = stack.choice_cps[i]

    # jump to alternative
    m.pc[0] = stack.choice_pcs[i]

    # restore proof-path depth to the branch point
    if m.trace_enabled[0] == 1:
        m.path_depth[0] = m.choice_path_depth[i]

    # reset structure-unification state
    m.mode[0] = READ
    m.S[0] = 0
    m.st_top[0] = 0

    return WAMStatus.SUCCESS


# ============================================================
# Instruction implementations
# ============================================================


@jit(void(uint16[:, :], int64, uint16[:]), cache=True)
def copy_X_to_choice(choice_X, i, X):
    for j in range(X.shape[0]):
        choice_X[i, j] = X[j]


@jit(void(uint16[:], uint16[:, :], int64), cache=True)
def copy_choice_to_X(X, choice_X, i):
    for j in range(X.shape[0]):
        X[j] = choice_X[i, j]


@jit(inline="always")
def sync_cp_from_call_stack(m) -> None:
    call_top = m.stack.call_top[0]
    if call_top < 0:
        m.cp[0] = -1
        return
    m.cp[0] = m.stack.call_pcs[call_top]


@jit(cache=False)
def exec_GET_VAR(m, x, i) -> WAMStatus:
    m.X[x] = m.X[i]
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_GET_VAL(m, x, i) -> WAMStatus:
    st = unify(m.stack, m.X[x], m.X[i])
    if st == WAMStatus.SUCCESS:
        return WAMStatus.SUCCESS
    if st == WAMStatus.EXHAUSTED:
        return _fail(m)
    return st


@jit(cache=True)
def exec_GET_CONST(m, c, i) -> WAMStatus:
    if not unify_const(m.stack, m.X[i], c):
        return _fail(m)
    return WAMStatus.SUCCESS


@jit(cache=True)
def exec_UNIFY_CONST(m, c) -> WAMStatus:
    stack = m.stack
    s = m.S[0]
    if m.mode[0] == READ:
        if not unify_const(stack, s, c):
            return _fail(m)
    else:
        stack.heap[s] = c
        stack.tags[s] = TAG.CON
    m.S[0] = s + 1
    return WAMStatus.SUCCESS


@jit(cache=True)
def exec_PUT_CONST(m, c, i) -> WAMStatus:
    # store constant directly as a tagged cell
    v = make_var(m.stack)
    m.stack.heap[v] = c
    m.stack.tags[v] = TAG.CON
    m.X[i] = v
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_PUT_VAL(m, x, i) -> WAMStatus:
    m.X[i] = m.X[x]
    return WAMStatus.SUCCESS


@jit(cache=True)
def exec_PUT_VAR(m, x, i) -> WAMStatus:
    """PUT_VAR Xx, Xi"""
    v = make_var(m.stack)
    m.X[x] = v
    m.X[i] = v
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_CALL(m, fun, arity, ret_pc) -> WAMStatus:
    stack = m.stack
    target = m.entry[fun, arity]
    if target < 0:
        # TODO decide
        # if m.strict_errors[0]:
        #     return WAMStatus.UNDEFINED_PREDICATE
        return _fail(m)

    call_i = stack.call_alloc_top[0]
    if call_i >= stack.call_pcs.shape[0]:
        return WAMStatus.STACK_OVERFLOW

    stack.call_prev[call_i] = stack.call_top[0]
    stack.call_pcs[call_i] = ret_pc
    copy_X_to_choice(stack.call_X, call_i, m.X)
    stack.call_top[0] = call_i
    stack.call_alloc_top[0] = call_i + 1
    m.cp[0] = ret_pc
    m.pc[0] = target
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_EXECUTE(m, fun, arity) -> WAMStatus:
    target = m.entry[fun, arity]
    if target < 0:
        # return WAMStatus.UNDEFINED_PREDICATE
        return _fail(m)

    m.pc[0] = target
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_MARK_CUT(m) -> WAMStatus:
    top = m.cut_top[0]
    if top >= m.cut_levels.shape[0]:
        return WAMStatus.STACK_OVERFLOW

    m.cut_levels[top] = m.stack.cp_top[0]
    m.cut_top[0] = top + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_CUT(m) -> WAMStatus:
    top = m.cut_top[0] - 1
    if top < 0:
        return WAMStatus.CORRUPT_CHOICEPOINT

    boundary = m.cut_levels[top]
    if boundary < 0 or boundary > m.stack.cp_top[0]:
        return WAMStatus.CORRUPT_CHOICEPOINT

    m.stack.cp_top[0] = boundary
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_DROP_CUT(m) -> WAMStatus:
    top = m.cut_top[0] - 1
    if top < 0:
        return WAMStatus.CORRUPT_CHOICEPOINT

    m.cut_top[0] = top
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_FAIL(m) -> WAMStatus:
    return _fail(m)


@jit(cache=False)
def exec_TRY(m, alt_pc) -> WAMStatus:
    stack = m.stack
    i = stack.cp_top[0]

    # guard: choicepoint stack overflow
    if i >= stack.choice_points.shape[0]:
        return WAMStatus.STACK_OVERFLOW

    stack.choice_points[i] = stack.trail_top[0]
    stack.choice_pcs[i] = alt_pc
    stack.choice_cps[i] = m.cp[0]
    stack.choice_call_tops[i] = stack.call_top[0]
    stack.choice_call_alloc_tops[i] = stack.call_alloc_top[0]
    stack.choice_heap_tops[i] = stack.heap_top[0]
    copy_X_to_choice(stack.choice_X, i, m.X)
    m.choice_path_depth[i] = m.path_depth[0]
    m.choice_cut_tops[i] = m.cut_top[0]

    stack.cp_top[0] = i + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_RETRY(m, alt_pc) -> WAMStatus:
    stack = m.stack
    i = stack.cp_top[0] - 1
    if i < 0:
        return WAMStatus.CORRUPT_CHOICEPOINT

    # restore argument registers for correct backtracking across goals
    copy_choice_to_X(m.X, stack.choice_X, i)

    # restore trail
    target = stack.choice_points[i]
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = stack.choice_heap_tops[i]

    # restore call-frame stack
    stack.call_top[0] = stack.choice_call_tops[i]
    stack.call_alloc_top[0] = stack.choice_call_alloc_tops[i]

    m.cut_top[0] = m.choice_cut_tops[i]

    # restore continuation pointer
    m.cp[0] = stack.choice_cps[i]

    # update retry PC
    stack.choice_pcs[i] = alt_pc

    # restore proof-path depth to the branch point
    if m.trace_enabled[0] == 1:
        m.path_depth[0] = m.choice_path_depth[i]

    # entering a new clause head: reset structure-unification state
    m.mode[0] = READ
    m.S[0] = 0
    m.st_top[0] = 0

    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_PROCEED(m) -> bool:
    """
    Return True if this PROCEED is top-level (one solution found).
    Return False if it just returns to the caller (continue execution).
    """
    stack = m.stack
    call_top = stack.call_top[0]
    if call_top < 0:
        return True  # top-level => yield one solution

    if m.trace_enabled[0] == 1 and m.path_depth[0] > 0:
        m.path_depth[0] -= 1

    m.pc[0] = stack.call_pcs[call_top]
    copy_choice_to_X(m.X, stack.call_X, call_top)
    stack.call_top[0] = stack.call_prev[call_top]
    sync_cp_from_call_stack(m)
    return False


@jit(cache=False)
def exec_TRUST(m) -> WAMStatus:
    stack = m.stack
    i = stack.cp_top[0] - 1

    # restore argument registers
    copy_choice_to_X(m.X, stack.choice_X, i)

    # restore trail
    target = stack.choice_points[i]
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = stack.choice_heap_tops[i]

    # restore call-frame stack
    stack.call_top[0] = stack.choice_call_tops[i]
    stack.call_alloc_top[0] = stack.choice_call_alloc_tops[i]

    m.cut_top[0] = m.choice_cut_tops[i]

    # restore CP
    m.cp[0] = stack.choice_cps[i]

    # restore proof-path depth to the branch point
    if m.trace_enabled[0] == 1:
        m.path_depth[0] = m.choice_path_depth[i]

    # POP choice point
    stack.cp_top[0] -= 1

    # entering a new clause head: reset structure-unification state
    m.mode[0] = READ
    m.S[0] = 0
    m.st_top[0] = 0

    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_JMP_RETRY(m, clause_pc, alt_pc) -> WAMStatus:
    """
    Same choicepoint semantics as RETRY, but jump to explicit clause_pc
    instead of falling through to pc + 1.
    """
    stack = m.stack
    i = stack.cp_top[0] - 1
    if i < 0:
        return WAMStatus.CORRUPT_CHOICEPOINT

    # restore argument registers for correct backtracking across goals
    copy_choice_to_X(m.X, stack.choice_X, i)

    # restore trail
    target = stack.choice_points[i]
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = stack.choice_heap_tops[i]

    m.cut_top[0] = m.choice_cut_tops[i]

    # restore continuation pointer
    m.cp[0] = stack.choice_cps[i]

    # update retry PC
    stack.choice_pcs[i] = alt_pc

    # restore proof-path depth to the branch point
    if m.trace_enabled[0] == 1:
        m.path_depth[0] = m.choice_path_depth[i]

    # entering a new clause head: reset structure-unification state
    m.mode[0] = READ
    m.S[0] = 0
    m.st_top[0] = 0

    # explicit jump to clause
    m.pc[0] = clause_pc
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_JMP_TRUST(m, clause_pc) -> WAMStatus:
    """
    Same choicepoint semantics as TRUST, but jump to explicit clause_pc
    instead of falling through to pc + 1.
    """
    stack = m.stack
    i = stack.cp_top[0] - 1
    if i < 0:
        return WAMStatus.CORRUPT_CHOICEPOINT

    # restore argument registers
    copy_choice_to_X(m.X, stack.choice_X, i)

    # restore trail
    target = stack.choice_points[i]
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = stack.choice_heap_tops[i]

    m.cut_top[0] = m.choice_cut_tops[i]

    # restore CP
    m.cp[0] = stack.choice_cps[i]

    # restore proof-path depth to the branch point
    if m.trace_enabled[0] == 1:
        m.path_depth[0] = m.choice_path_depth[i]

    # POP choice point
    stack.cp_top[0] -= 1

    # entering a new clause head: reset structure-unification state
    m.mode[0] = READ
    m.S[0] = 0
    m.st_top[0] = 0

    # explicit jump to clause
    m.pc[0] = clause_pc
    return WAMStatus.SUCCESS


@jit(cache=True)
def exec_PUT_STR(m, symbol_id, arity, i) -> WAMStatus:
    stack = m.stack

    # Allocate functor + arity argument cells on heap, and build a STR cell.
    fun_pos = make_functor(stack, symbol_id, arity)

    # Use a numpy array instead of a Python list
    args = np.empty(arity, dtype=np.uint16)
    for k in range(arity):
        args[k] = make_var(stack)
    str_cell = make_structure(stack, fun_pos, args)

    # PUT_STR places the structure into Xi (no unification with prior Xi).
    m.X[i] = str_cell

    # Enter WRITE mode and start unifying structure arguments.
    push_struct(m)
    m.mode[0] = WRITE
    m.S[0] = str_cell + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_GET_STR(m, symbol_id, arity, i) -> WAMStatus:
    stack = m.stack
    xi = deref(stack, m.X[i])

    # If Xi is not a valid heap address, matching must fail (and backtrack)
    if xi >= stack.heap.shape[0]:
        return _fail(m)

    if xi < stack.heap_top[0] and stack.tags[xi] == TAG.REF and stack.heap[xi] == xi:
        # Xi is unbound var: build structure and bind it
        fun_pos = make_functor(stack, symbol_id, arity)
        args = np.empty(arity, dtype=np.uint16)
        for k in range(arity):
            args[k] = make_var(stack)
        str_cell = make_structure(stack, fun_pos, args)

        st = unify(stack, xi, str_cell)
        if st == WAMStatus.EXHAUSTED:
            return _fail(m)
        if st != WAMStatus.SUCCESS:
            return st
        push_struct(m)
        m.mode[0] = WRITE
        m.S[0] = str_cell + 1
        return WAMStatus.SUCCESS

    xi = deref(stack, xi)
    if xi >= stack.heap_top[0] or stack.tags[xi] != TAG.STR:
        return _fail(m)

    fun_pos = stack.heap[xi]
    if stack.heap[fun_pos] != symbol_id or stack.heap[fun_pos + 1] != arity:
        return _fail(m)

    push_struct(m)
    m.mode[0] = READ
    m.S[0] = xi + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_UNIFY_VAR(m, j) -> WAMStatus:
    s = m.S[0]
    m.X[j] = s
    m.S[0] = s + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_UNIFY_VAL(m, j) -> WAMStatus:
    stack = m.stack
    s = m.S[0]
    if m.mode[0] == READ:
        st = unify(stack, m.X[j], s)
        if st == WAMStatus.EXHAUSTED:
            return _fail(m)
        if st != WAMStatus.SUCCESS:
            return st
    else:
        # WRITE: bind the arg cell (fresh var) to Xj's term
        stack.heap[s] = m.X[j]
        stack.tags[s] = TAG.REF
    m.S[0] = s + 1
    return WAMStatus.SUCCESS


@jit(cache=False)
def exec_END_STR(m) -> WAMStatus:
    pop_struct(m)
    return WAMStatus.SUCCESS


# ============================================================
# Interpreter loop
# ============================================================


@jit(cache=False)
def run(m) -> WAMStatus:
    code = m.code
    code_size = m.code_size

    while 0 <= m.pc[0] < code_size:
        if m.fuel[0] <= 0:
            return WAMStatus.STEP_LIMIT
        m.fuel[0] -= 1

        pc = m.pc[0]

        if m.trace_enabled[0] == 1:
            if 0 <= pc < m.clause_entry_term.shape[0]:
                term_id = m.clause_entry_term[pc]
                if term_id >= 0:
                    depth = m.path_depth[0]
                    if depth >= m.path_terms.shape[0]:
                        return WAMStatus.STACK_OVERFLOW
                    m.path_terms[depth] = term_id
                    m.path_depth[0] = depth + 1

        op, a, b, c = code[pc]

        if op == OP.GET_VAR:
            m.pc[0] = pc + 1
            exec_GET_VAR(m, a, b)
            continue  # pure, cannot fail

        elif op == OP.GET_VAL:
            m.pc[0] = pc + 1
            st = exec_GET_VAL(m, a, b)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.GET_CONST:
            m.pc[0] = pc + 1
            st = exec_GET_CONST(m, a, b)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.PUT_VAR:
            m.pc[0] = pc + 1
            exec_PUT_VAR(m, a, b)

        elif op == OP.PUT_VAL:
            m.pc[0] = pc + 1
            exec_PUT_VAL(m, a, b)

        elif op == OP.PUT_CONST:
            m.pc[0] = pc + 1
            exec_PUT_CONST(m, a, b)

        elif op == OP.CALL:
            # CALL can fail if predicate undefined
            st = exec_CALL(m, a, b, pc + 1)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.EXECUTE:
            # Tail call: jump to callee without dropping usage trace entry.
            st = exec_EXECUTE(m, a, b)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.MARK_CUT:
            m.pc[0] = pc + 1
            st = exec_MARK_CUT(m)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.CUT:
            m.pc[0] = pc + 1
            st = exec_CUT(m)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.DROP_CUT:
            m.pc[0] = pc + 1
            st = exec_DROP_CUT(m)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.FAIL:
            st = exec_FAIL(m)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.PROCEED:
            if exec_PROCEED(m):
                return WAMStatus.SUCCESS
            continue

        elif op == OP.TRY:
            st = exec_TRY(m, a)
            if st != WAMStatus.SUCCESS:
                return st
            m.pc[0] = pc + 1
            continue

        elif op == OP.RETRY:
            m.pc[0] = pc + 1
            st = exec_RETRY(m, a)  # if you make it return status
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.TRUST:
            m.pc[0] = pc + 1
            st = exec_TRUST(m)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.JMP_RETRY:
            st = exec_JMP_RETRY(m, a, b)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.JMP_TRUST:
            st = exec_JMP_TRUST(m, a)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.GET_STR:
            m.pc[0] = pc + 1
            st = exec_GET_STR(m, a, b, c)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.PUT_STR:
            m.pc[0] = pc + 1
            exec_PUT_STR(m, a, b, c)

        elif op == OP.UNIFY_VAR:
            m.pc[0] = pc + 1
            exec_UNIFY_VAR(m, a)

        elif op == OP.UNIFY_VAL:
            m.pc[0] = pc + 1
            st = exec_UNIFY_VAL(m, a)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.UNIFY_CONST:
            m.pc[0] = pc + 1
            st = exec_UNIFY_CONST(m, a)
            if st != WAMStatus.SUCCESS:
                return st

        elif op == OP.END_STR:
            m.pc[0] = pc + 1
            exec_END_STR(m)

        else:
            return WAMStatus.INVALID_OPCODE

    # If PC leaves code range:
    return WAMStatus.INVALID_PC
