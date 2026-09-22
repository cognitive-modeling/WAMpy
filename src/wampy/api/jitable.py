"""Numba integration for JIT-compatible WAMpy functions."""

__all__ = [
    "CompiledProgram",
    "CompiledQuery",
    "clear_compiled_program",
    "clear_compiled_query",
    "compile_program",
    "compile_query",
    "display",
    "init_compiled_program",
    "init_compiled_query",
    "init_compiler_state",
    "init_symbol_table",
    "render_term_line_jit",
    "reset_machine",
    "term_label_jit",
    "undo_hypothesis",
]

from numba.extending import register_jitable

from wampy.compiler.codegen.clause import (
    analyze_clause,
    compile_analyzed_goals,
    compile_clause,
    emit_base_bridge,
    emit_clause_prefix,
    init_clause_analysis_scratch,
)
from wampy.compiler.codegen.emitter import emit, patch_instruction
from wampy.compiler.codegen.goal import (
    compile_goal,
    compile_goals_inline,
    compile_grouped_goals_inline,
    compile_negated_goal,
    contains_control_cut,
    count_execution_goals,
    count_goals,
    count_naf_levels,
    has_later_cut,
    is_body_control_conjunction,
    max_callable_arity,
    max_goal_arity,
    unwrap_negated_goal,
)
from wampy.compiler.codegen.register_allocation import (
    allocate_x,
    encode_unbound_y,
    record_variable_occurrences,
    resolve_variable_register,
)
from wampy.compiler.codegen.term import (
    compile_head,
    emit_goal_arguments,
    emit_input_structure,
    emit_output_structure,
)
from wampy.compiler.compiled_program import (
    CompiledProgram,
    clear_compiled_program,
    init_compiled_program,
)
from wampy.compiler.compiled_query import (
    CompiledQuery,
    clear_compiled_query,
    init_compiled_query,
)
from wampy.compiler.compiler import compile_predicate_group, compile_program
from wampy.compiler.compiler_state import (
    clear_compiler_state,
    init_compiler_state,
    record_hypothesis_predicate_entry_undo,
    undo_hypothesis,
)
from wampy.compiler.diagnostics import display_compiler_state, render_compiler_state
from wampy.compiler.predicates import (
    find_predicate_slot,
    predicate_lookup_position,
    resolve_predicate_slot,
)
from wampy.compiler.program_plan import (
    _count_and_register_plan_clauses,
    build_program_compile_plan,
)
from wampy.compiler.query import _query_control_slots, compile_query
from wampy.config import ASTConfig
from wampy.frontend.ast import symbol_id as ast_symbol_id
from wampy.frontend.render import (
    display,
    has_term_content_jit,
    render,
    render_display_term_jit,
    render_term_line_jit,
    right_align_str_jit,
    symbol_for_id_jit,
    symbol_to_str_jit,
    term_label_jit,
    term_to_str_jit,
    usage_for_term_jit,
)
from wampy.frontend.symbol_table import (
    SymbolTable,
    _allocate_next_user_symbol_id,
    _check_layout_compatibility,
    _empty_symbol_table_resolved,
    intern_symbol,
    merge_symbol_tables,
)
from wampy.runtime.debug import record_path_usage
from wampy.runtime.interpreter.choice import (
    exec_JMP_RETRY,
    exec_JMP_TRUST,
    exec_RETRY_ME_ELSE,
    exec_TRUST_ME_ELSE_FAIL,
    exec_TRY_ME_ELSE,
    fail,
    mark_enclosing_naf_cutoff,
)
from wampy.runtime.interpreter.control import (
    _update_hb_after_cut,
    exec_ALLOCATE,
    exec_CALL,
    exec_CUT,
    exec_DEALLOCATE,
    exec_EXECUTE,
    exec_FAIL,
    exec_GET_LEVEL,
    exec_NECK_CUT,
    exec_PROCEED,
)
from wampy.runtime.interpreter.execution import (
    emulate,
    redo,
    run,
)
from wampy.runtime.interpreter.term import (
    exec_END_STR,
    exec_GET_CONST,
    exec_GET_STR,
    exec_GET_VAL,
    exec_GET_VAR,
    exec_GET_Y_VAL,
    exec_GET_Y_VAR,
    exec_PUT_CONST,
    exec_PUT_STR,
    exec_PUT_VAL,
    exec_PUT_VAR,
    exec_PUT_Y_VAL,
    exec_PUT_Y_VAR,
    exec_SET_CONST,
    exec_SET_VAL,
    exec_SET_VAR,
    exec_SET_Y_VAL,
    exec_SET_Y_VAR,
    exec_UNIFY_CONST,
    exec_UNIFY_VAL,
    exec_UNIFY_VAR,
    exec_UNIFY_Y_VAL,
    exec_UNIFY_Y_VAR,
)
from wampy.runtime.machine import (
    init_machine,
    init_machine_registers,
    init_pdl,
    reset_machine,
    reset_machine_registers,
)
from wampy.runtime.machine.choice_point import (
    copy_A_to_choice,
    copy_choice_to_A,
    discard_choice,
    restore_choice,
    save_choice,
)
from wampy.runtime.machine.environment import (
    allocate_environment,
    deallocate_environment,
    environment_size_at_continuation,
    environment_size_at_entry,
    environment_y_address,
)
from wampy.runtime.machine.heap import (
    deref,
    init_heap,
    make_const,
    make_functor,
    make_structure,
    make_var,
)
from wampy.runtime.machine.stack import init_stack, is_choice_point_address
from wampy.runtime.machine.trail import init_trail, trail_var, unwind_trail
from wampy.runtime.term_decode import read_term_view
from wampy.runtime.unification import (
    _u_pop,
    _u_push,
    bind_var,
    pop_struct,
    push_struct,
    unify,
    unify_const,
)


def init_symbol_table(config: ASTConfig) -> SymbolTable:
    """Create a symbol table from an AST config inside a Numba boundary."""

    _, first_user_symbol_id = ast_symbol_id.calculate_symbol_id_positions(
        config.max_variables_per_term,
    )
    return _empty_symbol_table_resolved(
        first_user_symbol_id,
        first_user_symbol_id + config.max_user_symbols,
    )


# Compiler
register_jitable(compile_program)
register_jitable(_query_control_slots)
register_jitable(compile_query)
register_jitable(init_compiled_program)
register_jitable(clear_compiled_program)
register_jitable(init_compiled_query)
register_jitable(clear_compiled_query)
register_jitable(init_compiler_state)
register_jitable(clear_compiler_state)
register_jitable(record_hypothesis_predicate_entry_undo)
register_jitable(emit_clause_prefix)
register_jitable(emit_base_bridge)
register_jitable(predicate_lookup_position)
register_jitable(find_predicate_slot)
register_jitable(resolve_predicate_slot)
register_jitable(_count_and_register_plan_clauses)
register_jitable(build_program_compile_plan)
register_jitable(compile_predicate_group)
register_jitable(inline="always")(allocate_x)
register_jitable(inline="always")(encode_unbound_y)
register_jitable(resolve_variable_register)
register_jitable(emit)
register_jitable(patch_instruction)
register_jitable(emit_input_structure)
register_jitable(emit_output_structure)
register_jitable(unwrap_negated_goal)
register_jitable(inline="always")(is_body_control_conjunction)
register_jitable(max_callable_arity)
register_jitable(count_goals)
register_jitable(record_variable_occurrences)
register_jitable(init_clause_analysis_scratch)
register_jitable(analyze_clause)
register_jitable(compile_analyzed_goals)
register_jitable(compile_goals_inline)
register_jitable(compile_grouped_goals_inline)
register_jitable(compile_clause)
register_jitable(compile_head)
register_jitable(emit_goal_arguments)
register_jitable(compile_negated_goal)
register_jitable(compile_goal)
register_jitable(count_execution_goals)
register_jitable(count_naf_levels)
register_jitable(contains_control_cut)
register_jitable(has_later_cut)
register_jitable(max_goal_arity)

# Symbol table
register_jitable(init_symbol_table)
register_jitable(_empty_symbol_table_resolved)
register_jitable(_allocate_next_user_symbol_id)
register_jitable(intern_symbol)
register_jitable(_check_layout_compatibility)
register_jitable(merge_symbol_tables)


# Rendering
register_jitable(symbol_for_id_jit)
register_jitable(term_label_jit)
register_jitable(right_align_str_jit)
register_jitable(usage_for_term_jit)
register_jitable(has_term_content_jit)
register_jitable(symbol_to_str_jit)
register_jitable(term_to_str_jit)
register_jitable(render_term_line_jit)
register_jitable(render_display_term_jit)
register_jitable(render)
register_jitable(display)
register_jitable(render_compiler_state)
register_jitable(display_compiler_state)


# Runtime machine
register_jitable(init_heap)
register_jitable(init_trail)
register_jitable(init_stack)
register_jitable(is_choice_point_address)
register_jitable(init_machine_registers)
register_jitable(init_pdl)
register_jitable(reset_machine_registers)
register_jitable(init_machine)
register_jitable(reset_machine)
register_jitable(deref)
register_jitable(trail_var)
register_jitable(make_var)
register_jitable(make_const)
register_jitable(make_functor)
register_jitable(make_structure)
register_jitable(unwind_trail)
register_jitable(allocate_environment)
register_jitable(deallocate_environment)
register_jitable(save_choice)
register_jitable(discard_choice)


# Runtime unification
register_jitable(unify)
register_jitable(unify_const)
register_jitable(bind_var)
register_jitable(push_struct)
register_jitable(pop_struct)
register_jitable(_u_push)
register_jitable(_u_pop)


# Runtime interpreter
register_jitable(redo)
register_jitable(fail)
register_jitable(mark_enclosing_naf_cutoff)
register_jitable(restore_choice)
register_jitable(copy_A_to_choice)
register_jitable(copy_choice_to_A)
register_jitable(environment_size_at_continuation)
register_jitable(environment_size_at_entry)
register_jitable(environment_y_address)
register_jitable(exec_ALLOCATE)
register_jitable(exec_DEALLOCATE)
register_jitable(exec_GET_VAR)
register_jitable(exec_GET_Y_VAR)
register_jitable(exec_GET_Y_VAL)
register_jitable(exec_GET_VAL)
register_jitable(exec_GET_CONST)
register_jitable(exec_UNIFY_CONST)
register_jitable(exec_PUT_CONST)
register_jitable(exec_PUT_VAL)
register_jitable(exec_PUT_VAR)
register_jitable(exec_PUT_Y_VAR)
register_jitable(exec_PUT_Y_VAL)
register_jitable(exec_CALL)
register_jitable(exec_EXECUTE)
register_jitable(exec_GET_LEVEL)
register_jitable(exec_CUT)
register_jitable(exec_NECK_CUT)
register_jitable(exec_FAIL)
register_jitable(_update_hb_after_cut)
register_jitable(exec_TRY_ME_ELSE)
register_jitable(exec_RETRY_ME_ELSE)
register_jitable(exec_PROCEED)
register_jitable(exec_TRUST_ME_ELSE_FAIL)
register_jitable(exec_JMP_RETRY)
register_jitable(exec_JMP_TRUST)
register_jitable(exec_PUT_STR)
register_jitable(exec_SET_VAR)
register_jitable(exec_SET_Y_VAR)
register_jitable(exec_SET_VAL)
register_jitable(exec_SET_Y_VAL)
register_jitable(exec_SET_CONST)
register_jitable(exec_GET_STR)
register_jitable(exec_UNIFY_VAR)
register_jitable(exec_UNIFY_VAL)
register_jitable(exec_UNIFY_Y_VAR)
register_jitable(exec_UNIFY_Y_VAL)
register_jitable(exec_END_STR)
register_jitable(run)
register_jitable(emulate)


# Runtime term decoding
register_jitable(read_term_view)
register_jitable(record_path_usage)
register_jitable(undo_hypothesis)
