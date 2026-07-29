"""
WAM compiler adapted to the rational_program AST layout.

Compiles:
    Program(children, symbol, ...)  ->  Static WAM instruction list

Predicate entry table layout:

entry[functor_id, arity] -> entry PC
"""

from typing import NamedTuple

import numpy as np
from numba import jit

import wampy.ast as ast
from numba_toolbox import MutableInt32, mutable_int32
from wampy.config import WAMConfig, DEFAULT_CONFIG
from wampy.frontend.ast.program import NONE, Program
from wampy.frontend.ast.symbol_id import SymID

from wampy.compiler.opcodes import OP

# ============================================================
# Constants
# ============================================================

OMIT_REDUNDANT_ALLOCATE = True
VAR_UNBOUND = -1
MAX_GOAL_STACK = 256  # TODO add to config # must be >= max AND nesting depth
ENTRY_INVALID_PC = -1
NO_NODE = -1  # sentinel for "no head node"
STRUCT_STACK_SIZE = 64  # TODO add to config


class CodeArea(NamedTuple):
    code: np.ndarray  # int32[:, 4]
    pc: MutableInt32
    pc_onto: MutableInt32
    entry: np.ndarray  # int32[:, :]
    clause_entry_term: np.ndarray  # int32[:], maps clause entry PC -> source term_id


@jit(cache=True)
def init_compiler(config: WAMConfig = DEFAULT_CONFIG):
    """
    Only code[:code_area.pc.value] may be read or executed.
    """

    max_instr = config.limits.max_instr
    code = np.empty((max_instr, 4), dtype=np.int32)
    pc = mutable_int32(0)
    pc_onto = mutable_int32(0)

    entry = np.full(
        (config.entry.max_functors, config.entry.max_arity + 1),
        ENTRY_INVALID_PC,
        dtype=np.int32,
    )

    clause_entry_term = np.full(max_instr, -1, dtype=np.int32)

    code_area = CodeArea(
        code=code,
        pc=pc,
        pc_onto=pc_onto,
        entry=entry,
        clause_entry_term=clause_entry_term,
    )
    return code_area


@jit(cache=True)
def reset_compiler(code_area, config: WAMConfig = DEFAULT_CONFIG):
    """Reset the compiler each time before compile."""
    # Only clear the USED portion of the clause_entry_term array
    last_pc = code_area.pc.value
    if last_pc > 0:
        code_area.clause_entry_term[:last_pc].fill(-1)

    # Reset the PC and pc_onto counters
    code_area.pc.value = 0
    code_area.pc_onto.value = 0  # (Added this: it was missing from your original code!)

    # Clear the entry table
    code_area.entry.fill(ENTRY_INVALID_PC)

    return code_area


# ============================================================
# Top-level compilation
# ============================================================


@jit(cache=True)
def compile(
    program: Program,
    code_area: CodeArea = None,
    config: WAMConfig = DEFAULT_CONFIG,
):
    if code_area == None:
        code_area = init_compiler(config)
    else:
        pass
        # TODO: reset the compiler here. before we need to clean up the compile_onto first (program as input)
        # code_area = reset_compiler(code_area, config)

    clause_counts = get_validated_clause_counts(program, config)
    emitted = np.zeros_like(clause_counts)

    # pending_patch[fun,arity] stores the TRY/RETRY instruction index
    # that must jump to the next alternative of the same predicate.
    pending_patch = np.full(clause_counts.shape, -1, dtype=np.int32)

    heads_buf = np.empty(program.symbol.shape[1], dtype=np.int32)

    for tid in range(program.symbol.shape[0]):
        head_root, body_root = ast.clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = ast.collect_head_nodes(program, tid, head_root, heads_buf)
        if n_heads == 0:
            continue

        for j in range(n_heads):
            head = heads_buf[j]

            fun = int(program.symbol[tid, head])
            arity = ast.get_arity_of_node(program, tid, head)

            total = clause_counts[fun, arity]
            index = emitted[fun, arity]

            # Patch previous alternative for this SAME predicate to current pc.
            prev_patch = pending_patch[fun, arity]
            if prev_patch >= 0:
                code_area.code[prev_patch, 1] = code_area.pc.value
                pending_patch[fun, arity] = -1

            if index == 0:
                code_area.entry[fun, arity] = code_area.pc.value

            if total > 1:
                if index == 0:
                    code_area = emit(code_area, OP.TRY, 0)
                    pending_patch[fun, arity] = code_area.pc.value - 1
                elif index < total - 1:
                    code_area = emit(code_area, OP.RETRY, 0)
                    pending_patch[fun, arity] = code_area.pc.value - 1
                else:
                    code_area = emit(code_area, OP.TRUST)

            entry_pc = code_area.pc.value
            if 0 <= entry_pc < code_area.clause_entry_term.shape[0]:
                code_area.clause_entry_term[entry_pc] = tid

            code_area = compile_clause(code_area, program, tid, head, body_root)

            emitted[fun, arity] += 1

    return code_area


@jit(cache=True)
def compile_program_onto(
    static_code_area,
    static_entry,
    static_size,
    dynamic_program,
    config: WAMConfig = DEFAULT_CONFIG,
):
    """
    Compile dynamic program on top of an already compiled (static) program

    Params:
    static_entry: original entry table of the static program (need to backup urself)
    static_size: original size of the compiled static program (last PC used +1)

    Bridge model:
    - dynamic predicates run first
    - on backtracking, dynamic-prog jumps to a bridge emitted AFTER the dynamic clause
    - bridge enters unchanged static program

    Bridge forms:
    - single-clause static:  JMP_TRUST static_entry_pc
    - multi-clause static:   JMP_RETRY (static_entry_pc + 1), static_try_next_alt
    """

    static_code_area.pc.value = static_size  ## static_compiled.code.shape[0]
    static_code_area.entry[:, :] = static_entry[:, :]  ## reset with copy

    static_code_area.clause_entry_term[static_size:].fill(-1)  ## TODO: do we need to reset "old dynamic terms?"

    clause_counts = get_validated_clause_counts(dynamic_program, config)

    emitted = np.zeros_like(clause_counts)
    pending_patch = np.full(clause_counts.shape, -1, dtype=np.int32)
    heads_buf = np.empty(dynamic_program.symbol.shape[1], dtype=np.int32)

    for tid in range(dynamic_program.symbol.shape[0]):
        head_root, body_root = ast.clause_head_body(dynamic_program, tid)
        if head_root == NO_NODE:
            continue

        n_heads = ast.collect_head_nodes(dynamic_program, tid, head_root, heads_buf)
        if n_heads == 0:
            continue

        for j in range(n_heads):
            head = heads_buf[j]

            fun = int(dynamic_program.symbol[tid, head])
            arity = ast.get_arity_of_node(dynamic_program, tid, head)

            total = clause_counts[fun, arity]
            index = emitted[fun, arity]

            static_entry_pc = int(static_entry[fun, arity])
            overlaps_static = static_entry_pc != ENTRY_INVALID_PC

            # patch previous dynamic internal alternative for this predicate
            prev_patch = pending_patch[fun, arity]
            if prev_patch >= 0:
                static_code_area.code[prev_patch, 1] = static_code_area.pc.value
                pending_patch[fun, arity] = -1

            # first dynamic clause overrides public entry
            if index == 0:
                static_code_area.entry[fun, arity] = static_code_area.pc.value

            # if >= 0, this dynamic prefix must later be patched to the bridge pc
            bridge_patch_pc = -1

            # emit dynamic prefix
            if total == 1:
                if overlaps_static:
                    # single dynamic clause with static fallback
                    static_code_area = emit(static_code_area, OP.TRY, 0)
                    bridge_patch_pc = static_code_area.pc.value - 1
            else:
                if index == 0:
                    static_code_area = emit(static_code_area, OP.TRY, 0)
                    pending_patch[fun, arity] = static_code_area.pc.value - 1
                elif index < total - 1:
                    static_code_area = emit(static_code_area, OP.RETRY, 0)
                    pending_patch[fun, arity] = static_code_area.pc.value - 1
                else:
                    # last dynamic alternative
                    if overlaps_static:
                        static_code_area = emit(static_code_area, OP.RETRY, 0)
                        bridge_patch_pc = static_code_area.pc.value - 1
                    else:
                        static_code_area = emit(static_code_area, OP.TRUST)

            entry_pc = static_code_area.pc.value
            if 0 <= entry_pc < static_code_area.clause_entry_term.shape[0]:
                static_code_area.clause_entry_term[entry_pc] = tid

            # compile dynamic clause
            static_code_area = compile_clause(static_code_area, dynamic_program, tid, head, body_root)
            emitted[fun, arity] += 1

            # append bridge AFTER dynamic clause, then patch dynamic prefix to point to it
            if bridge_patch_pc >= 0:
                bridge_pc = static_code_area.pc.value

                first_op = int(static_code_area.code[static_entry_pc, 0])
                if first_op == OP.TRY:
                    # multi-clause static:
                    # first real clause starts at static_entry_pc + 1
                    # next alt prefix comes from TRY arg
                    first_clause_pc = static_entry_pc + 1
                    next_prefix_pc = int(static_code_area.code[static_entry_pc, 1])
                    static_code_area = emit(static_code_area, OP.JMP_RETRY, first_clause_pc, next_prefix_pc)
                else:
                    # single-clause static:
                    # entry already points directly to the clause
                    static_code_area = emit(static_code_area, OP.JMP_TRUST, static_entry_pc)

                static_code_area.code[bridge_patch_pc, 1] = bridge_pc

    return static_code_area


# ============================================================
# Helpers
# ============================================================


@jit(cache=True)
def emit(code_area, op, a=0, b=0, c=0):
    pc = code_area.pc.value
    code_area.code[pc, 0] = op
    code_area.code[pc, 1] = a
    code_area.code[pc, 2] = b
    code_area.code[pc, 3] = c
    code_area.pc.value = pc + 1
    return code_area


@jit(cache=True)
def head_node(program, tid):
    if program.symbol[tid, 0] != SymID.CLAUSE:
        return NO_NODE

    children = program.children[tid, 0]

    for c in children:
        if c == NONE:
            continue
        if program.symbol[tid, c] == SymID.TRUE:
            continue
        return c

    # zero-arity fact: head exists but has no args
    return 1  # or whatever index the functor node uses


@jit(inline="always", cache=True)
def get_validated_clause_counts(program, config: WAMConfig):
    counts = ast.get_count_clauses(program)
    if counts.shape[0] > config.entry.max_functors:
        raise ValueError("ENTRY_MAX_FUNCTORS too small for program")
    if counts.shape[1] > config.entry.max_arity + 1:
        raise ValueError("ENTRY_MAX_ARITY too small for program")
    return counts


@jit(cache=True)
def emit_structure(code_area, program, tid, root_node, root_reg, var_table, next_x, is_head):
    """
    Emit GET_STR/PUT_STR + UNIFY_* + END_STR for a compound term, including nesting.
    Returns (code_area, next_x) ALWAYS.
    is_head: 1 => GET_STR, 0 => PUT_STR
    """

    node_stack = np.empty(STRUCT_STACK_SIZE, dtype=np.int32)
    reg_stack = np.empty(STRUCT_STACK_SIZE, dtype=np.int32)
    idx_stack = np.empty(STRUCT_STACK_SIZE, dtype=np.int32)
    top = 0

    # root
    node_stack[top] = root_node
    reg_stack[top] = root_reg
    idx_stack[top] = 0
    top += 1

    fun = program.symbol[tid, root_node]
    ar = ast.get_arity_of_node(program, tid, root_node)

    if is_head == 1:
        code_area = emit(code_area, OP.GET_STR, fun, ar, root_reg)
    else:
        code_area = emit(code_area, OP.PUT_STR, fun, ar, root_reg)

    while top > 0:
        node = node_stack[top - 1]
        idx = idx_stack[top - 1]

        children = program.children[tid, node]

        # find next child
        child = NONE
        k = idx
        while k < children.shape[0]:
            if children[k] != NONE:
                child = children[k]
                idx_stack[top - 1] = k + 1
                break
            k += 1

        if child == NONE:
            code_area = emit(code_area, OP.END_STR, 0, 0, 0)
            top -= 1
            continue

        sym = program.symbol[tid, child]

        # variable?
        if SymID.VAR_FIRST <= sym <= SymID.VAR_LAST:
            reg0 = var_table[sym]
            if reg0 == VAR_UNBOUND:
                var_table[sym] = next_x
                code_area = emit(code_area, OP.UNIFY_VAR, next_x, 0, 0)
                next_x += 1
            else:
                code_area = emit(code_area, OP.UNIFY_VAL, reg0, 0, 0)
            continue

        # compound?
        if ast.get_arity_of_node(program, tid, child) > 0:
            # make a temp reg holding this argument cell
            t = next_x
            next_x += 1

            code_area = emit(code_area, OP.UNIFY_VAR, t, 0, 0)

            # push nested
            if top >= STRUCT_STACK_SIZE:
                # hard fail in compiler; you can also clamp or raise a custom error
                return code_area, next_x

            node_stack[top] = child
            reg_stack[top] = t
            idx_stack[top] = 0
            top += 1

            f2 = program.symbol[tid, child]
            a2 = ast.get_arity_of_node(program, tid, child)

            if is_head == 1:
                code_area = emit(code_area, OP.GET_STR, f2, a2, t)
            else:
                code_area = emit(code_area, OP.PUT_STR, f2, a2, t)
            continue

        # atomic const
        code_area = emit(code_area, OP.UNIFY_CONST, sym, 0, 0)

    return code_area, next_x


# ============================================================
# Clause compilation (NO var_map)
# ============================================================


@jit(cache=True)
def count_goals(program, tid, root_node):
    count = 0

    stack = np.empty(MAX_GOAL_STACK, dtype=np.int32)
    top = 0

    stack[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]

        sym = program.symbol[tid, node]

        if sym == SymID.AND:
            children = program.children[tid, node]
            for c in children:
                if c != NONE:
                    stack[top] = c
                    top += 1

        elif sym == SymID.TRUE:
            continue
        else:
            count += 1

    return count


@jit(cache=True)
def compile_goals_inline(code_area, program, tid, root_node, var_table, next_x):
    remaining = count_goals(program, tid, root_node)

    stack = np.empty(MAX_GOAL_STACK, dtype=np.int32)
    top = 0

    stack[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]

        sym = program.symbol[tid, node]

        if sym == SymID.AND:
            children = program.children[tid, node]
            left = NONE
            right = NONE

            for c in children:
                if c != NONE:
                    if left == NONE:
                        left = c
                    else:
                        right = c
                        break

            # push right first, then left
            if right != NONE:
                stack[top] = right
                top += 1
            if left != NONE:
                stack[top] = left
                top += 1

        elif sym == SymID.TRUE:
            continue

        else:
            remaining -= 1
            is_last = remaining == 0

            code_area, next_x = compile_goal(code_area, program, tid, node, var_table, next_x, is_last)

            if is_last:
                return code_area, next_x

    return code_area, next_x


@jit(cache=True)
def compile_clause(code_area, program, tid, head, body_root):
    next_x = 0
    arity = ast.get_arity_of_node(program, tid, head)

    var_table = np.full((SymID.VAR_LAST + 1,), VAR_UNBOUND, dtype=np.int16)
    next_x = arity  # start after argument registers

    code_area, next_x = compile_head(code_area, program, tid, head, var_table, next_x)

    # Compile body from the positional body_root only (matches syntax_tree semantics)
    if body_root != NO_NODE:
        if program.symbol[tid, body_root] != SymID.TRUE:
            code_area, next_x = compile_goals_inline(code_area, program, tid, body_root, var_table, next_x)

    return emit(code_area, OP.PROCEED)


@jit(cache=True)
def compile_head(code_area, program, tid, node, var_table, next_x):
    args = program.children[tid, node]

    for i in range(args.shape[0]):
        a = args[i]
        if a == NONE:
            continue

        sym = program.symbol[tid, a]

        if ast.is_sym_a_var(sym):
            reg = var_table[sym]
            if reg == VAR_UNBOUND:
                var_table[sym] = next_x
                code_area = emit(code_area, OP.GET_VAR, next_x, i)
                next_x += 1
            else:
                code_area = emit(code_area, OP.GET_VAL, reg, i)

        else:
            # NEW: structure support
            if ast.is_compound(program, tid, a):
                code_area, next_x = emit_structure(code_area, program, tid, a, i, var_table, next_x, True)
            else:
                code_area = emit(code_area, OP.GET_CONST, sym, i)

    return code_area, next_x


@jit(cache=True)
def emit_goal_arguments(code_area, program, tid, node, var_table, next_x):
    args = program.children[tid, node]

    for i in range(args.shape[0]):
        a = args[i]
        if a == NONE:
            continue

        s = program.symbol[tid, a]

        if ast.is_sym_a_var(s):
            reg = var_table[s]
            if reg == VAR_UNBOUND:
                var_table[s] = next_x
                code_area = emit(code_area, OP.PUT_VAR, next_x, i)
                next_x += 1
            else:
                code_area = emit(code_area, OP.PUT_VAL, reg, i)
        else:
            if ast.is_compound(program, tid, a):
                code_area, next_x = emit_structure(code_area, program, tid, a, i, var_table, next_x, False)
            else:
                code_area = emit(code_area, OP.PUT_CONST, s, i)

    return code_area, next_x


@jit(cache=True)
def compile_negated_goal(code_area, program, tid, not_node, var_table, next_x):
    current = not_node
    depth = 0
    try_pcs = np.empty(MAX_GOAL_STACK, dtype=np.int32)

    while True:
        children = program.children[tid, current]
        child = NO_NODE
        child_count = 0
        for i in range(children.shape[0]):
            c = children[i]
            if c == NONE:
                continue
            child_count += 1
            if child == NO_NODE:
                child = int(c)

        if child_count != 1:
            raise ValueError("Malformed NOT goal: expected exactly one operand")

        child_sym = program.symbol[tid, child]
        if child_sym == SymID.NOT_PROVABLE:
            depth += 1
            if depth >= MAX_GOAL_STACK:
                raise ValueError("Negation nesting is too deep")
            current = child
            continue

        depth += 1
        target = child
        break

    target_sym = program.symbol[tid, target]
    if target_sym == SymID.AND:
        raise ValueError("Grouped negation is not supported in this phase")
    if ast.is_sym_a_var(target_sym):
        raise ValueError("Negation operand must be a callable goal")

    if target_sym != SymID.TRUE:
        code_area, next_x = emit_goal_arguments(code_area, program, tid, target, var_table, next_x)

    for i in range(depth):
        code_area = emit(code_area, OP.MARK_CUT)
        code_area = emit(code_area, OP.TRY, 0)
        try_pcs[i] = code_area.pc.value - 1

    arity = ast.get_arity_of_node(program, tid, target)
    if target_sym != SymID.TRUE:
        code_area = emit(code_area, OP.CALL, int(target_sym), arity)

    for i in range(depth - 1, -1, -1):
        code_area = emit(code_area, OP.CUT)
        code_area = emit(code_area, OP.FAIL)
        code_area.code[try_pcs[i], 1] = code_area.pc.value
        code_area = emit(code_area, OP.TRUST)
        code_area = emit(code_area, OP.DROP_CUT)

    return code_area, next_x


@jit(cache=True)
def compile_goal(code_area, program, tid, node, var_table, next_x, is_last):
    sym = program.symbol[tid, node]
    if sym == SymID.NOT_PROVABLE:
        return compile_negated_goal(code_area, program, tid, node, var_table, next_x)

    code_area, next_x = emit_goal_arguments(code_area, program, tid, node, var_table, next_x)

    arity = ast.get_arity_of_node(program, tid, node)
    if is_last:
        code_area = emit(code_area, OP.EXECUTE, sym, arity)
    else:
        code_area = emit(code_area, OP.CALL, sym, arity)

    return code_area, next_x
