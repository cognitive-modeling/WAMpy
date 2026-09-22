"""Analysis and code generation for clause-body goals."""

import numpy as np

from wampy.compiler.codegen.emitter import emit, patch_instruction
from wampy.compiler.codegen.term import emit_goal_arguments
from wampy.compiler.compiler_state import NO_PREDICATE_SLOT
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.config import CompilerConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.runtime.machine.stack import ChoicePointKind


def unwrap_negated_goal(program, tid, not_node, config: CompilerConfig):
    """Return ``(target_node, negation_depth)`` for a NAF chain."""

    current = not_node
    depth = 0

    while True:
        child_count = ast.get_child_count(program, tid, current)
        child = ast.get_child_at(program, tid, current, 0)

        if child_count != 1:
            raise ValueError("Malformed NOT goal: expected exactly one operand")

        child_sym = ast.get_node_symbol(program, tid, child)
        depth += 1
        if child_sym == CoreSymID.NEGATION_AS_FAILURE:
            if depth >= config.max_goal_depth:
                raise ValueError("Negation nesting is too deep")
            current = child
            continue

        return child, depth


def is_body_control_conjunction(program, tid, body_root, node):
    """Return whether ``node`` is a control conjunction below ``body_root``.

    Conjunction nodes are followed only through conjunction children, so a
    comma term used as an argument of an ordinary callable is not mistaken for
    body control.
    """

    stack = np.empty(ast.get_node_capacity(program), dtype=np.int32)
    top = 0
    stack[top] = body_root
    top += 1

    while top > 0:
        top -= 1
        current = stack[top]
        if current == node:
            return ast.get_node_symbol(program, tid, current) == CoreSymID.CONJUNCTION

        if ast.get_node_symbol(program, tid, current) != CoreSymID.CONJUNCTION:
            continue

        child_count = ast.get_child_count(program, tid, current)
        for child_index in range(child_count):
            child = ast.get_child_at(program, tid, current, child_index)
            if child == NO_NODE:
                continue
            if top >= stack.shape[0]:
                raise ValueError("Goal nesting is too deep")
            stack[top] = child
            top += 1

    return False


def max_goal_arity(program, tid, root_node, config: CompilerConfig):
    """Return the largest callable arity in one executable goal expression."""

    if root_node == NO_NODE:
        return 0

    maximum = 0
    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    roots = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0
    stack[top] = root_node
    roots[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        control_root = roots[top]
        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, control_root, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                roots[top] = control_root
                top += 1
            continue

        if sym == CoreSymID.TRUE or sym == CoreSymID.CUT:
            continue

        if sym == CoreSymID.NEGATION_AS_FAILURE:
            target, _ = unwrap_negated_goal(program, tid, node, config)
            target_sym = ast.get_node_symbol(program, tid, target)
            if target_sym == CoreSymID.CONJUNCTION:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = target
                roots[top] = target
                top += 1
                continue
            if target_sym == CoreSymID.TRUE:
                continue
            arity = ast.get_child_count(program, tid, target)
        else:
            arity = ast.get_child_count(program, tid, node)

        if arity > maximum:
            maximum = arity

    return maximum


def max_callable_arity(program, tid, head, body_root, config: CompilerConfig):
    """Return the argument-register window required by one clause.

    Body-level AND nodes are traversed as conjunction control nodes.  Ordinary
    goals contribute only their own callable arity; their argument subtrees
    are intentionally not traversed.
    """

    argument_register_count = ast.get_child_count(program, tid, head)
    if body_root == NO_NODE:
        return argument_register_count

    body_arity = max_goal_arity(program, tid, body_root, config)
    if body_arity > argument_register_count:
        argument_register_count = body_arity

    return argument_register_count


def count_goals(program, tid, root_node, config: CompilerConfig):
    count = 0

    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0

    stack[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]

        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, root_node, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                top += 1

        elif sym == CoreSymID.TRUE:
            continue
        else:
            count += 1

    return count


def count_execution_goals(program, tid, root_node, config: CompilerConfig):
    """Count callable execution events, including calls inside grouped NAF."""

    if root_node == NO_NODE:
        return 0

    count = 0
    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    roots = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0
    stack[top] = root_node
    roots[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        control_root = roots[top]
        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, control_root, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                roots[top] = control_root
                top += 1
            continue

        if sym == CoreSymID.TRUE:
            continue

        if sym == CoreSymID.NEGATION_AS_FAILURE:
            target, _ = unwrap_negated_goal(program, tid, node, config)
            if ast.get_node_symbol(program, tid, target) == CoreSymID.CONJUNCTION:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = target
                roots[top] = target
                top += 1
                continue

        count += 1

    return count


def count_naf_levels(program, tid, root_node, config: CompilerConfig):
    """Count every NAF control level in an executable goal expression."""

    if root_node == NO_NODE:
        return 0

    count = 0
    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    roots = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0
    stack[top] = root_node
    roots[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        control_root = roots[top]
        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, control_root, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                roots[top] = control_root
                top += 1
            continue

        if sym != CoreSymID.NEGATION_AS_FAILURE:
            continue

        target, depth = unwrap_negated_goal(program, tid, node, config)
        count += depth
        if ast.get_node_symbol(program, tid, target) == CoreSymID.CONJUNCTION:
            if top >= config.max_goal_depth:
                raise ValueError("Goal nesting is too deep")
            stack[top] = target
            roots[top] = target
            top += 1

    return count


def has_later_cut(program, tid, root_node, config: CompilerConfig):
    """Return whether a cut occurs after the first executable goal."""

    if root_node == NO_NODE:
        return False

    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    roots = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0
    stack[top] = root_node
    roots[top] = root_node
    top += 1
    phase = 0

    while top > 0:
        top -= 1
        node = stack[top]
        control_root = roots[top]
        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, control_root, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                roots[top] = control_root
                top += 1
            continue

        if sym == CoreSymID.TRUE:
            continue

        if sym == CoreSymID.NEGATION_AS_FAILURE:
            target, _ = unwrap_negated_goal(program, tid, node, config)
            if ast.get_node_symbol(program, tid, target) == CoreSymID.CONJUNCTION:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = target
                roots[top] = target
                top += 1
                continue

        phase += 1
        if sym == CoreSymID.CUT and phase > 1:
            return True

    return False


def contains_control_cut(program, tid, root_node, config: CompilerConfig):
    """Return whether a goal expression contains a control-level cut."""

    if root_node == NO_NODE:
        return False

    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    roots = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0
    stack[top] = root_node
    roots[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]
        control_root = roots[top]
        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CUT:
            return True

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, control_root, node
        ):
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = child
                roots[top] = control_root
                top += 1
            continue

        if sym == CoreSymID.NEGATION_AS_FAILURE:
            target, _ = unwrap_negated_goal(program, tid, node, config)
            if ast.get_node_symbol(program, tid, target) == CoreSymID.CONJUNCTION:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = target
                roots[top] = target
                top += 1

    return False


def compile_goals_inline(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    root_node,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
    tail_call,
    has_environment,
    environment_sizes,
    cut_y=-1,
    naf_y_base=-1,
    environment_size_override=-1,
):
    remaining = count_goals(program, tid, root_node, config)
    phase = 0
    naf_level_offset = 0

    stack = np.empty(config.max_goal_depth, dtype=np.int32)
    top = 0

    stack[top] = root_node
    top += 1

    while top > 0:
        top -= 1
        node = stack[top]

        sym = ast.get_node_symbol(program, tid, node)

        if sym == CoreSymID.CONJUNCTION and is_body_control_conjunction(
            program, tid, root_node, node
        ):
            left = NO_NODE
            right = NO_NODE

            child_count = ast.get_child_count(program, tid, node)
            if child_count > 0:
                left = ast.get_child_at(program, tid, node, 0)
            if child_count > 1:
                right = ast.get_child_at(program, tid, node, 1)

            # push right first, then left
            if right != NO_NODE:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = right
                top += 1
            if left != NO_NODE:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                stack[top] = left
                top += 1

        elif sym == CoreSymID.TRUE:
            continue

        else:
            remaining -= 1
            phase += 1
            is_last = remaining == 0

            compiler_state, next_x, naf_levels_used = compile_goal(
                compiled_program,
                compiler_state,
                code_size,
                program,
                tid,
                node,
                var_table,
                next_x,
                is_last and tail_call,
                config,
                max_x_registers,
                has_environment,
                (
                    environment_size_override
                    if environment_size_override >= 0
                    else environment_sizes[phase]
                ),
                cut_y,
                phase == 1,
                naf_y_base + naf_level_offset if naf_y_base >= 0 else -1,
            )
            naf_level_offset += naf_levels_used

            if is_last:
                return compiler_state, next_x

    return compiler_state, next_x


def compile_grouped_goals_inline(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    root_node,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
    environment_size,
    naf_y_base,
    analysis_validated=False,
):
    """Compile a grouped goal without recursing through the normal goal loop."""

    task_nodes = np.empty(config.max_goal_depth, dtype=np.int32)
    task_kinds = np.empty(config.max_goal_depth, dtype=np.uint8)
    task_frames = np.empty(config.max_goal_depth, dtype=np.int32)
    frame_bases = np.empty(config.max_goal_depth, dtype=np.int32)
    frame_depths = np.empty(config.max_goal_depth, dtype=np.int32)
    frame_try_pcs = np.empty(
        (config.max_goal_depth, config.max_goal_depth),
        dtype=np.int32,
    )

    top = 0
    task_nodes[top] = root_node
    task_kinds[top] = 0
    task_frames[top] = -1
    top += 1
    frame_top = 0
    naf_level_offset = 0

    while top > 0:
        top -= 1
        node = task_nodes[top]
        kind = task_kinds[top]
        frame = task_frames[top]

        if kind == 1:
            base = frame_bases[frame]
            depth = frame_depths[frame]
            for i in range(depth - 1, -1, -1):
                emit(compiled_program.code, code_size, OP.CUT, base + i)
                emit(compiled_program.code, code_size, OP.FAIL)
                patch_instruction(
                    compiled_program.code,
                    frame_try_pcs[frame, i],
                    code_size[0],
                )
                emit(compiled_program.code, code_size, OP.TRUST_ME_ELSE_FAIL)
            frame_top -= 1
            continue

        sym = ast.get_node_symbol(program, tid, node)
        # This stack starts at a NAF conjunction and only expands conjunction
        # children, so every conjunction it contains is body-level control.
        if sym == CoreSymID.CONJUNCTION:
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count - 1, -1, -1):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                task_nodes[top] = child
                task_kinds[top] = 0
                task_frames[top] = -1
                top += 1
            continue

        if sym == CoreSymID.TRUE:
            continue
        if sym == CoreSymID.CUT:
            raise ValueError("Cut inside grouped negation is not supported yet")

        if sym == CoreSymID.NEGATION_AS_FAILURE:
            target, depth = unwrap_negated_goal(program, tid, node, config)
            target_sym = ast.get_node_symbol(program, tid, target)
            if not analysis_validated:
                if target_sym == CoreSymID.CUT or ast.is_variable_symbol(program, target_sym):
                    raise ValueError("Negation operand must be a callable goal")
                if target_sym == CoreSymID.CONJUNCTION and contains_control_cut(
                    program, tid, target, config
                ):
                    raise ValueError("Cut inside grouped negation is not supported yet")

            if frame_top >= config.max_goal_depth:
                raise ValueError("Negation nesting is too deep")
            if naf_y_base < 0:
                raise ValueError("Negation is missing saved choice-point levels")

            base = naf_y_base + naf_level_offset
            frame_bases[frame_top] = base
            frame_depths[frame_top] = depth
            for i in range(depth):
                emit(compiled_program.code, code_size, OP.GET_LEVEL, base + i)
                emit(
                    compiled_program.code,
                    code_size,
                    OP.TRY_ME_ELSE,
                    0,
                    ChoicePointKind.NAF,
                )
                frame_try_pcs[frame_top, i] = code_size[0] - 1
            frame = frame_top
            frame_top += 1
            naf_level_offset += depth

            if top >= config.max_goal_depth:
                raise ValueError("Goal nesting is too deep")
            task_nodes[top] = NO_NODE
            task_kinds[top] = 1
            task_frames[top] = frame
            top += 1

            if target_sym != CoreSymID.TRUE:
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                task_nodes[top] = target
                task_kinds[top] = 0
                task_frames[top] = -1
                top += 1
            continue

        compiler_state, next_x = emit_goal_arguments(
            compiled_program,
            compiler_state,
            code_size,
            program,
            tid,
            node,
            var_table,
            next_x,
            config,
            max_x_registers,
        )
        arity = ast.get_child_count(program, tid, node)
        predicate_slot = find_predicate_slot(compiler_state, int(sym))
        if predicate_slot == NO_PREDICATE_SLOT:
            raise ValueError("Predicate slot was not registered")
        emit(
            compiled_program.code,
            code_size,
            OP.CALL,
            predicate_slot,
            arity,
            environment_size,
        )

    return compiler_state, next_x, naf_level_offset


def compile_negated_goal(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    not_node,
    var_table,
    next_x,
    config: CompilerConfig,
    max_x_registers,
    environment_size,
    naf_y_base=-1,
    analysis_validated=False,
):
    try_pcs = np.empty(config.max_goal_depth, dtype=np.int32)

    target, depth = unwrap_negated_goal(program, tid, not_node, config)
    target_sym = ast.get_node_symbol(program, tid, target)
    if not analysis_validated:
        if target_sym == CoreSymID.CUT:
            raise ValueError("Negation operand must be a callable goal")
        if ast.is_variable_symbol(program, target_sym):
            raise ValueError("Negation operand must be a callable goal")
        if target_sym == CoreSymID.CONJUNCTION and contains_control_cut(
            program, tid, target, config
        ):
            raise ValueError("Cut inside grouped negation is not supported yet")

        if target_sym == CoreSymID.CONJUNCTION:
            target_arity = max_goal_arity(program, tid, target, config)
        else:
            target_arity = ast.get_child_count(program, tid, target)
        if target_arity > max_x_registers:
            raise ValueError("Predicate arity exceeds X register capacity")
        if next_x < target_arity:
            next_x = target_arity

    for i in range(depth):
        if naf_y_base < 0:
            raise ValueError("Negation is missing saved choice-point levels")
        emit(compiled_program.code, code_size, OP.GET_LEVEL, naf_y_base + i)
        emit(
            compiled_program.code,
            code_size,
            OP.TRY_ME_ELSE,
            0,
            ChoicePointKind.NAF,
        )
        try_pcs[i] = code_size[0] - 1

    if target_sym == CoreSymID.CONJUNCTION:
        compiler_state, next_x, compiled_naf_levels = compile_grouped_goals_inline(
            compiled_program,
            compiler_state,
            code_size,
            program,
            tid,
            target,
            var_table,
            next_x,
            config,
            max_x_registers,
            environment_size,
            naf_y_base + depth,
            analysis_validated,
        )
        nested_naf_levels = compiled_naf_levels
    elif target_sym != CoreSymID.TRUE:
        compiler_state, next_x = emit_goal_arguments(
            compiled_program,
            compiler_state,
            code_size,
            program,
            tid,
            target,
            var_table,
            next_x,
            config,
            max_x_registers,
        )
        arity = ast.get_child_count(program, tid, target)
        predicate_slot = find_predicate_slot(compiler_state, int(target_sym))
        if predicate_slot == NO_PREDICATE_SLOT:
            raise ValueError("Predicate slot was not registered")
        emit(
            compiled_program.code,
            code_size,
            OP.CALL,
            predicate_slot,
            arity,
            environment_size,
        )
        nested_naf_levels = 0
    else:
        nested_naf_levels = 0

    for i in range(depth - 1, -1, -1):
        emit(compiled_program.code, code_size, OP.CUT, naf_y_base + i)
        emit(compiled_program.code, code_size, OP.FAIL)
        patch_instruction(compiled_program.code, try_pcs[i], code_size[0])
        emit(compiled_program.code, code_size, OP.TRUST_ME_ELSE_FAIL)

    return compiler_state, next_x, depth + nested_naf_levels


def compile_goal(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    node,
    var_table,
    next_x,
    is_last,
    config: CompilerConfig,
    max_x_registers,
    has_environment,
    environment_size,
    cut_y=-1,
    is_first_goal=False,
    naf_y_base=-1,
    analysis_validated=False,
):
    sym = ast.get_node_symbol(program, tid, node)
    if sym == CoreSymID.NEGATION_AS_FAILURE:
        compiler_state, next_x, naf_levels_used = compile_negated_goal(
            compiled_program,
            compiler_state,
            code_size,
            program,
            tid,
            node,
            var_table,
            next_x,
            config,
            max_x_registers,
            environment_size,
            naf_y_base,
            analysis_validated,
        )
        return compiler_state, next_x, naf_levels_used

    if sym == CoreSymID.CUT:
        if is_first_goal:
            emit(compiled_program.code, code_size, OP.NECK_CUT)
        elif cut_y >= 0:
            emit(compiled_program.code, code_size, OP.CUT, cut_y)
        else:
            raise ValueError("Non-neck cut is missing a saved choice-point level")
        return compiler_state, next_x, 0

    compiler_state, next_x = emit_goal_arguments(
        compiled_program,
        compiler_state,
        code_size,
        program,
        tid,
        node,
        var_table,
        next_x,
        config,
        max_x_registers,
    )

    arity = ast.get_child_count(program, tid, node)
    predicate_slot = find_predicate_slot(compiler_state, int(sym))
    if predicate_slot == NO_PREDICATE_SLOT:
        raise ValueError("Predicate slot was not registered")
    if is_last:
        if has_environment:
            emit(compiled_program.code, code_size, OP.DEALLOCATE)
        emit(
            compiled_program.code,
            code_size,
            OP.EXECUTE,
            predicate_slot,
            arity,
        )
    else:
        emit(
            compiled_program.code,
            code_size,
            OP.CALL,
            predicate_slot,
            arity,
            environment_size,
        )

    return compiler_state, next_x, 0
