"""Code generation for clause-head and goal-argument terms."""

import numpy as np

from wampy.compiler.codegen.emitter import emit
from wampy.compiler.codegen.register_allocation import allocate_x, resolve_variable_register
from wampy.compiler.opcodes import OP
from wampy.config import CompilerConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE


def emit_input_structure(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    root_node,
    root_reg,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
):
    """
    Emit a compound input term breadth-first with GET_STR.

    Clause heads match an existing term, so the parent structure must be
    opened before its children are inspected.  Nested children are deferred
    until every direct parent argument has been emitted.  In WRITE mode this
    keeps the parent's argument cells contiguous on the heap.
    """

    queue_capacity = ast.get_node_capacity(program)
    node_queue = np.empty(queue_capacity, dtype=np.int32)
    register_queue = np.empty(queue_capacity, dtype=np.int32)
    depth_queue = np.empty(queue_capacity, dtype=np.int32)
    queue_head = 0
    queue_tail = 1

    node_queue[0] = root_node
    register_queue[0] = root_reg
    depth_queue[0] = 1

    while queue_head < queue_tail:
        node = node_queue[queue_head]
        register = register_queue[queue_head]
        depth = depth_queue[queue_head]
        queue_head += 1

        fun = ast.get_node_symbol(program, tid, node)
        child_count = ast.get_child_count(program, tid, node)
        emit(compiled_program.code, code_size, OP.GET_STR, fun, child_count, register)

        for child_index in range(child_count):
            child = ast.get_child_at(program, tid, node, child_index)
            sym = ast.get_node_symbol(program, tid, child)

            if ast.is_variable_symbol(program, sym):
                var_slot = ast.variable_index(program, sym)
                reg0, is_y, first, next_x = resolve_variable_register(
                    var_table,
                    var_slot,
                    next_x,
                    max_x_registers,
                )
                if first:
                    opcode = OP.UNIFY_Y_VAR if is_y else OP.UNIFY_VAR
                else:
                    opcode = OP.UNIFY_Y_VAL if is_y else OP.UNIFY_VAL
                emit(
                    compiled_program.code,
                    code_size,
                    opcode,
                    reg0,
                    0,
                    0,
                )
                continue

            if ast.is_compound_node(program, tid, child):
                if depth >= config.max_structure_depth:
                    raise ValueError("Structure nesting is too deep")
                if queue_tail >= queue_capacity:
                    raise ValueError("Structure nesting is too large")

                child_register, next_x = allocate_x(next_x, max_x_registers)
                emit(
                    compiled_program.code,
                    code_size,
                    OP.UNIFY_VAR,
                    child_register,
                    0,
                    0,
                )
                node_queue[queue_tail] = child
                register_queue[queue_tail] = child_register
                depth_queue[queue_tail] = depth + 1
                queue_tail += 1
                continue

            emit(compiled_program.code, code_size, OP.UNIFY_CONST, sym, 0, 0)

        emit(compiled_program.code, code_size, OP.END_STR, 0, 0, 0)

    return compiler_state, next_x


def emit_output_structure(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    root_node,
    root_reg,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
):
    """Emit a compound output term bottom-up with PUT_STR and SET_*.

    This follows the classic WAM construction split: PUT_STR creates only the
    structure representation and the following SET_* instructions append its
    arguments one heap cell at a time. Nested structures are built first in
    temporary X registers and then attached with SET_VAL.
    """

    node_stack = np.empty(config.max_structure_depth, dtype=np.int32)
    child_index_stack = np.empty(config.max_structure_depth, dtype=np.int32)
    postorder = np.empty(ast.get_node_capacity(program), dtype=np.int32)
    node_register = np.full(ast.get_node_capacity(program), -1, dtype=np.int32)

    top = 0
    n_postorder = 0

    node_stack[top] = root_node
    child_index_stack[top] = 0
    top += 1

    while top > 0:
        node = node_stack[top - 1]
        child_index = child_index_stack[top - 1]
        child = NO_NODE
        child_count = ast.get_child_count(program, tid, node)
        if child_index < child_count:
            child = ast.get_child_at(program, tid, node, child_index)
            child_index_stack[top - 1] = child_index + 1

        if child != NO_NODE and ast.is_compound_node(program, tid, child):
            if top >= config.max_structure_depth:
                raise ValueError("Structure nesting is too deep")
            node_stack[top] = child
            child_index_stack[top] = 0
            top += 1
            continue

        if child != NO_NODE:
            continue

        postorder[n_postorder] = node
        n_postorder += 1
        top -= 1

    node_register[root_node] = root_reg
    for i in range(n_postorder):
        node = postorder[i]
        if node != root_node:
            node_register[node], next_x = allocate_x(next_x, max_x_registers)

    for i in range(n_postorder):
        node = postorder[i]
        fun = ast.get_node_symbol(program, tid, node)
        arity = ast.get_child_count(program, tid, node)
        destination = node_register[node]

        emit(
            compiled_program.code,
            code_size,
            OP.PUT_STR,
            fun,
            arity,
            destination,
        )

        child_count = ast.get_child_count(program, tid, node)
        for child_index in range(child_count):
            child = ast.get_child_at(program, tid, node, child_index)
            sym = ast.get_node_symbol(program, tid, child)

            if ast.is_compound_node(program, tid, child):
                emit(
                    compiled_program.code,
                    code_size,
                    OP.SET_VAL,
                    node_register[child],
                    0,
                    0,
                )
                continue

            if ast.is_variable_symbol(program, sym):
                var_slot = ast.variable_index(program, sym)
                reg, is_y, first, next_x = resolve_variable_register(
                    var_table,
                    var_slot,
                    next_x,
                    max_x_registers,
                )
                if first:
                    emit(
                        compiled_program.code,
                        code_size,
                        OP.SET_Y_VAR if is_y else OP.SET_VAR,
                        reg,
                        0,
                        0,
                    )
                else:
                    emit(
                        compiled_program.code,
                        code_size,
                        OP.SET_Y_VAL if is_y else OP.SET_VAL,
                        reg,
                        0,
                        0,
                    )
                continue

            emit(
                compiled_program.code,
                code_size,
                OP.SET_CONST,
                sym,
                0,
                0,
            )

    return compiler_state, next_x


def compile_head(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    node,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
):
    arity = ast.get_child_count(program, tid, node)
    for i in range(arity):
        a = ast.get_child_at(program, tid, node, i)

        sym = ast.get_node_symbol(program, tid, a)

        if ast.is_variable_symbol(program, sym):
            var_slot = ast.variable_index(program, sym)
            reg, is_y, first, next_x = resolve_variable_register(
                var_table,
                var_slot,
                next_x,
                max_x_registers,
            )
            if first:
                op = OP.GET_Y_VAR if is_y else OP.GET_VAR
            else:
                op = OP.GET_Y_VAL if is_y else OP.GET_VAL
            emit(compiled_program.code, code_size, op, reg, i)

        else:
            if ast.is_compound_node(program, tid, a):
                compiler_state, next_x = emit_input_structure(
                    compiled_program,
                    compiler_state,
                    code_size,
                    program,
                    tid,
                    a,
                    i,
                    var_table,
                    next_x,
                    config,
                    max_x_registers,
                )
            else:
                emit(compiled_program.code, code_size, OP.GET_CONST, sym, i)

    return compiler_state, next_x


def emit_goal_arguments(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    node,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
):
    arity = ast.get_child_count(program, tid, node)
    for i in range(arity):
        a = ast.get_child_at(program, tid, node, i)

        s = ast.get_node_symbol(program, tid, a)

        if ast.is_variable_symbol(program, s):
            var_slot = ast.variable_index(program, s)
            reg, is_y, first, next_x = resolve_variable_register(
                var_table,
                var_slot,
                next_x,
                max_x_registers,
            )
            if first:
                op = OP.PUT_Y_VAR if is_y else OP.PUT_VAR
            else:
                op = OP.PUT_Y_VAL if is_y else OP.PUT_VAL
            emit(compiled_program.code, code_size, op, reg, i)
        else:
            if ast.is_compound_node(program, tid, a):
                compiler_state, next_x = emit_output_structure(
                    compiled_program,
                    compiler_state,
                    code_size,
                    program,
                    tid,
                    a,
                    i,
                    var_table,
                    next_x,
                    config,
                    max_x_registers,
                )
            else:
                emit(compiled_program.code, code_size, OP.PUT_CONST, s, i)

    return compiler_state, next_x
