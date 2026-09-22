"""Analysis and code generation for complete clauses."""

from typing import NamedTuple

import numpy as np

from wampy.compiler.codegen.emitter import emit
from wampy.compiler.codegen.goal import (
    compile_goal,
    unwrap_negated_goal,
)
from wampy.compiler.codegen.register_allocation import (
    VAR_UNBOUND,
    encode_unbound_y,
    record_variable_occurrences,
)
from wampy.compiler.codegen.term import compile_head
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import resolve_predicate_slot
from wampy.config import CompilerConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE
from wampy.frontend.ast.symbol_id import CoreSymID


class ClauseAnalysis(NamedTuple):
    """Control and liveness information needed to compile one clause."""

    var_table: np.ndarray
    goal_nodes: np.ndarray
    goal_count: int
    call_environment_sizes: np.ndarray
    argument_register_count: int
    has_environment: bool
    tail_execute: bool
    cut_y: int
    naf_y: int


class ClauseAnalysisScratch(NamedTuple):
    """Reusable fixed-capacity buffers for one compile_program() invocation."""

    var_table: np.ndarray
    first: np.ndarray
    last: np.ndarray
    permanent: np.ndarray
    variable_stack: np.ndarray
    control_nodes: np.ndarray
    control_grouped: np.ndarray
    control_owner: np.ndarray
    goal_nodes: np.ndarray
    event_owner: np.ndarray
    event_always_returns: np.ndarray
    call_environment_sizes: np.ndarray


def init_clause_analysis_scratch(
    node_capacity,
    config: CompilerConfig,
    max_variables_per_term,
):
    """Allocate fixed buffers shared by every clause in one compilation."""

    return ClauseAnalysisScratch(
        var_table=np.empty(max_variables_per_term, dtype=np.int16),
        first=np.empty(max_variables_per_term, dtype=np.int32),
        last=np.empty(max_variables_per_term, dtype=np.int32),
        permanent=np.empty(max_variables_per_term, dtype=np.uint8),
        variable_stack=np.empty(node_capacity, dtype=np.int32),
        control_nodes=np.empty(config.max_goal_depth, dtype=np.int32),
        control_grouped=np.empty(config.max_goal_depth, dtype=np.uint8),
        control_owner=np.empty(config.max_goal_depth, dtype=np.int32),
        goal_nodes=np.empty(node_capacity, dtype=np.int32),
        event_owner=np.empty(node_capacity + 1, dtype=np.int32),
        event_always_returns=np.empty(node_capacity + 1, dtype=np.uint8),
        call_environment_sizes=np.empty(node_capacity + 1, dtype=np.int32),
    )


def emit_clause_prefix(
    compiled_program,
    compiler_state,
    is_first,
    is_last,
    has_base_fallback,
):
    """Emit choice-prefix instructions for base or extension clauses."""

    next_patch_pc = -1
    bridge_patch_pc = -1

    if is_last and is_first:
        if has_base_fallback:
            emit(compiled_program.code, compiler_state.pc, OP.TRY_ME_ELSE, 0)
            bridge_patch_pc = compiler_state.pc[0] - 1

    elif is_first:
        emit(compiled_program.code, compiler_state.pc, OP.TRY_ME_ELSE, 0)
        next_patch_pc = compiler_state.pc[0] - 1

    elif not is_last:
        emit(compiled_program.code, compiler_state.pc, OP.RETRY_ME_ELSE, 0)
        next_patch_pc = compiler_state.pc[0] - 1

    elif has_base_fallback:
        # Last extension alternative falls through to the persistent base.
        emit(compiled_program.code, compiler_state.pc, OP.RETRY_ME_ELSE, 0)
        bridge_patch_pc = compiler_state.pc[0] - 1

    else:
        emit(compiled_program.code, compiler_state.pc, OP.TRUST_ME_ELSE_FAIL)

    return compiler_state, next_patch_pc, bridge_patch_pc


def emit_base_bridge(compiled_program, compiler_state, bridge_patch_pc, base_entry_pc):
    """Append a bridge from the last extension clause into base code."""

    bridge_pc = compiler_state.pc[0]
    first_op = int(compiled_program.code[base_entry_pc, 0])

    if first_op == OP.TRY_ME_ELSE:
        # Multi-clause base predicate: enter the first real clause while
        # preserving the first choice instruction's next-alternative target.
        first_clause_pc = base_entry_pc + 1
        next_prefix_pc = int(compiled_program.code[base_entry_pc, 1])
        emit(
            compiled_program.code,
            compiler_state.pc,
            OP.JMP_RETRY,
            first_clause_pc,
            next_prefix_pc,
        )
    else:
        # Single-clause base predicate: its entry already is the clause PC.
        emit(
            compiled_program.code,
            compiler_state.pc,
            OP.JMP_TRUST,
            base_entry_pc,
        )

    compiled_program.code[bridge_patch_pc, 1] = bridge_pc
    return compiler_state


def analyze_clause(
    program,
    tid,
    head,
    body_root,
    compiler_state,
    scratch: ClauseAnalysisScratch,
    config: CompilerConfig,
):
    """Analyze one clause and register its callable body predicates."""

    var_table = scratch.var_table
    var_table.fill(VAR_UNBOUND)
    argument_register_count = ast.get_child_count(program, tid, head)

    if body_root == NO_NODE or ast.get_node_symbol(program, tid, body_root) == CoreSymID.TRUE:
        return ClauseAnalysis(
            var_table=var_table,
            goal_nodes=scratch.goal_nodes,
            goal_count=0,
            call_environment_sizes=scratch.call_environment_sizes,
            argument_register_count=argument_register_count,
            has_environment=False,
            tail_execute=False,
            cut_y=-1,
            naf_y=-1,
        )

    first = scratch.first
    last = scratch.last
    permanent = scratch.permanent
    first.fill(-1)
    last.fill(-1)
    permanent.fill(0)

    record_variable_occurrences(
        program,
        tid,
        head,
        0,
        first,
        last,
        scratch.variable_stack,
    )

    control_nodes = scratch.control_nodes
    control_grouped = scratch.control_grouped
    control_owner = scratch.control_owner
    goal_nodes = scratch.goal_nodes
    event_owner = scratch.event_owner
    event_always_returns = scratch.event_always_returns

    top = 0
    control_nodes[top] = body_root
    control_grouped[top] = 0
    control_owner[top] = 0
    top += 1

    goal_count = 0
    event_count = 0
    has_non_neck_cut = False
    has_naf = False

    while top > 0:
        top -= 1
        node = control_nodes[top]
        is_grouped = control_grouped[top] == 1
        owner = control_owner[top]
        symbol = ast.get_node_symbol(program, tid, node)

        # Every node on this stack is known to be body-level control, so a
        # conjunction can be expanded directly without rescanning its body.
        if symbol == CoreSymID.CONJUNCTION:
            child_count = ast.get_child_count(program, tid, node)
            for child_index in range(child_count - 1, -1, -1):
                child = ast.get_child_at(program, tid, node, child_index)
                if child == NO_NODE:
                    continue
                if top >= config.max_goal_depth:
                    raise ValueError("Goal nesting is too deep")
                control_nodes[top] = child
                control_grouped[top] = 1 if is_grouped else 0
                control_owner[top] = owner
                top += 1
            continue

        if symbol == CoreSymID.TRUE:
            continue

        # A non-grouped node corresponds to one compile_goal() invocation.
        if not is_grouped:
            goal_count += 1
            if goal_count > goal_nodes.shape[0]:
                raise ValueError("Too many goals in clause")
            goal_nodes[goal_count - 1] = node
            owner = goal_count

        if symbol == CoreSymID.CUT:
            if is_grouped:
                raise ValueError("Cut inside grouped negation is not supported yet")
            if owner > 1:
                has_non_neck_cut = True
            continue

        if symbol == CoreSymID.NEGATION_AS_FAILURE:
            target, _ = unwrap_negated_goal(program, tid, node, config)
            has_naf = True
            target_symbol = ast.get_node_symbol(program, tid, target)

            if target_symbol == CoreSymID.CUT or ast.is_variable_symbol(program, target_symbol):
                raise ValueError("Negation operand must be a callable goal")

            if target_symbol == CoreSymID.CONJUNCTION:
                if top >= config.max_goal_depth:
                    raise ValueError("Negation nesting is too deep")
                control_nodes[top] = target
                control_grouped[top] = 1
                control_owner[top] = owner
                top += 1
                continue

            if target_symbol == CoreSymID.TRUE:
                continue

            resolve_predicate_slot(compiler_state, int(target_symbol))
            target_arity = ast.get_child_count(program, tid, target)
            if target_arity > argument_register_count:
                argument_register_count = target_arity

            event_count += 1
            event_owner[event_count] = owner
            event_always_returns[event_count] = 1
            record_variable_occurrences(
                program,
                tid,
                target,
                event_count,
                first,
                last,
                scratch.variable_stack,
            )
            continue

        resolve_predicate_slot(compiler_state, int(symbol))
        arity = ast.get_child_count(program, tid, node)
        if arity > argument_register_count:
            argument_register_count = arity

        event_count += 1
        event_owner[event_count] = owner
        event_always_returns[event_count] = 1 if is_grouped else 0

        record_variable_occurrences(
            program,
            tid,
            node,
            event_count,
            first,
            last,
            scratch.variable_stack,
        )

    tail_execute = False
    if goal_count > 0:
        last_symbol = ast.get_node_symbol(program, tid, goal_nodes[goal_count - 1])
        tail_execute = last_symbol != CoreSymID.CUT and last_symbol != CoreSymID.NEGATION_AS_FAILURE

    call_environment_sizes = scratch.call_environment_sizes
    for goal_index in range(goal_count + 1):
        call_environment_sizes[goal_index] = 0

    has_environment = False

    for phase in range(1, event_count + 1):
        owner = event_owner[phase]
        returning_call = event_always_returns[phase] != 0 or owner < goal_count

        if not returning_call:
            continue

        has_environment = True
        live_count = 0
        for variable_slot in range(var_table.shape[0]):
            variable_first = first[variable_slot]
            if variable_first < 0:
                continue
            if variable_first <= phase and phase < last[variable_slot]:
                permanent[variable_slot] = 1
                live_count += 1

        if owner > 0 and live_count > call_environment_sizes[owner]:
            call_environment_sizes[owner] = live_count

    y_count = 0
    for variable_slot in range(var_table.shape[0]):
        if permanent[variable_slot] == 0:
            continue
        var_table[variable_slot] = encode_unbound_y(y_count)
        y_count += 1

    cut_y = -1
    if has_non_neck_cut:
        cut_y = y_count
        y_count += 1
        has_environment = True

    naf_y = y_count
    if has_naf:
        has_environment = True

    return ClauseAnalysis(
        var_table=var_table,
        goal_nodes=goal_nodes,
        goal_count=goal_count,
        call_environment_sizes=call_environment_sizes,
        argument_register_count=argument_register_count,
        has_environment=has_environment,
        tail_execute=tail_execute,
        cut_y=cut_y,
        naf_y=naf_y,
    )


def compile_analyzed_goals(
    compiled_program,
    compiler_state,
    code_size,
    program,
    tid,
    analysis: ClauseAnalysis,
    next_x,
    config,
    max_x_registers,
):
    """Emit the top-level body goals found during clause analysis."""

    naf_level_offset = 0

    for goal_index in range(analysis.goal_count):
        phase = goal_index + 1
        node = analysis.goal_nodes[goal_index]
        is_last = goal_index == analysis.goal_count - 1

        compiler_state, next_x, naf_levels_used = compile_goal(
            compiled_program,
            compiler_state,
            code_size,
            program,
            tid,
            node,
            analysis.var_table,
            next_x,
            is_last and analysis.tail_execute,
            config,
            max_x_registers,
            analysis.has_environment,
            analysis.call_environment_sizes[phase],
            analysis.cut_y,
            phase == 1,
            analysis.naf_y + naf_level_offset if analysis.naf_y >= 0 else -1,
            True,
        )
        naf_level_offset += naf_levels_used

    return compiler_state, next_x


def compile_clause(
    compiled_program,
    compiler_state,
    program,
    tid,
    head,
    body_root,
    config: CompilerConfig,
    analysis_scratch: ClauseAnalysisScratch,
    max_x_registers,
):
    analysis = analyze_clause(
        program,
        tid,
        head,
        body_root,
        compiler_state,
        analysis_scratch,
        config,
    )
    if analysis.argument_register_count > max_x_registers:
        raise ValueError("Predicate arity exceeds X register capacity")
    next_x = analysis.argument_register_count

    if analysis.has_environment:
        emit(compiled_program.code, compiler_state.pc, OP.ALLOCATE)

    compiler_state, next_x = compile_head(
        compiled_program,
        compiler_state,
        compiler_state.pc,
        program,
        tid,
        head,
        analysis.var_table,
        next_x,
        config,
        max_x_registers,
    )

    if analysis.cut_y >= 0:
        emit(compiled_program.code, compiler_state.pc, OP.GET_LEVEL, analysis.cut_y, 1)

    compiler_state, next_x = compile_analyzed_goals(
        compiled_program,
        compiler_state,
        compiler_state.pc,
        program,
        tid,
        analysis,
        next_x,
        config,
        max_x_registers,
    )

    if analysis.tail_execute:
        return compiler_state

    if analysis.has_environment:
        emit(compiled_program.code, compiler_state.pc, OP.DEALLOCATE)

    emit(compiled_program.code, compiler_state.pc, OP.PROCEED)
    return compiler_state
