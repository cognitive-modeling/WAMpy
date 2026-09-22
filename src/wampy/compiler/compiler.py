"""Top-level orchestration for compiling AST programs to WAM code."""

from wampy.compiler import program_plan
from wampy.compiler.codegen import clause
from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiler_state import (
    ENTRY_INVALID_PC,
    CompilerState,
    record_hypothesis_predicate_entry_undo,
)
from wampy.config import DEFAULT_CONFIG, WAMConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import ASTProgram


def compile_program(
    program: ASTProgram,
    compiled_program: CompiledProgram,
    compiler_state: CompilerState,
    config: WAMConfig = DEFAULT_CONFIG,
) -> None:
    """Compile ``program`` into the supplied reusable compiled program.

    Callers must clear fresh compiler/program state before a new base compile
    and undo an installed hypothesis before compiling its replacement.

    With a fresh state, ``program`` becomes the persistent base and ``pc_onto``
    is set to the end of that base.

    With an existing base, ``program`` becomes the new hypothesis. The base
    prefix remains unchanged and remains available as a backtracking fallback.
    """

    compiler_config = config.compiler
    max_x_registers = config.runtime.max_x_registers

    if compiler_config.max_arity > max_x_registers:
        raise ValueError("Predicate arity exceeds X register capacity")

    fresh_base = (
        compiler_state.pc_onto[0] == 0
        and compiler_state.pc[0] == 0
        and compiler_state.predicate_count[0] == 0
    )

    if not fresh_base and compiler_state.hypothesis_predicate_entry_undo_count[0] != 0:
        raise ValueError("undo_hypothesis() must be called before compiling another hypothesis")

    plan = program_plan.build_program_compile_plan(
        program,
        compiler_state,
        compiler_config,
        max_x_registers,
    )
    node_capacity = ast.get_node_capacity(program)
    analysis_scratch = clause.init_clause_analysis_scratch(
        node_capacity,
        compiler_config,
        config.frontend.ast.max_variables_per_term,
    )

    for signature_index in range(plan.signature_count):
        compiler_state = compile_predicate_group(
            compiled_program,
            compiler_state,
            program,
            plan,
            signature_index,
            compiler_config,
            analysis_scratch,
            max_x_registers,
            fresh_base,
        )

    if fresh_base:
        compiler_state.pc_onto[0] = compiler_state.pc[0]

    compiled_program.code_size[0] = compiler_state.pc[0]


def compile_predicate_group(
    compiled_program,
    compiler_state,
    program,
    plan: program_plan.ProgramCompilePlan,
    signature_index,
    config,
    analysis_scratch,
    max_x_registers,
    fresh_base,
):
    """Emit one complete predicate-signature group from a compile plan."""

    signatures = plan.signatures
    clauses = plan.clauses
    predicate_slot = signatures[signature_index, program_plan.PLAN_SIGNATURE_SLOT]
    arity = signatures[signature_index, program_plan.PLAN_SIGNATURE_ARITY]
    clause_index = signatures[signature_index, program_plan.PLAN_SIGNATURE_FIRST_CLAUSE]

    base_entry_pc = compiled_program.predicate_entry[predicate_slot, arity]
    if not fresh_base:
        record_hypothesis_predicate_entry_undo(
            compiler_state,
            predicate_slot,
            arity,
            base_entry_pc,
        )
    compiled_program.predicate_entry[predicate_slot, arity] = compiler_state.pc[0]

    has_base_fallback = base_entry_pc != ENTRY_INVALID_PC
    previous_patch = ENTRY_INVALID_PC
    is_first = True

    while clause_index != program_plan.PLAN_INVALID_INDEX:
        tid = clauses[clause_index, program_plan.PLAN_CLAUSE_TID]
        head = clauses[clause_index, program_plan.PLAN_CLAUSE_HEAD]
        body_root = clauses[clause_index, program_plan.PLAN_CLAUSE_BODY]
        next_clause = clauses[clause_index, program_plan.PLAN_CLAUSE_NEXT]
        is_last = next_clause == program_plan.PLAN_INVALID_INDEX

        if previous_patch != ENTRY_INVALID_PC:
            compiled_program.code[previous_patch, 1] = compiler_state.pc[0]

        compiler_state, next_patch, bridge_patch = clause.emit_clause_prefix(
            compiled_program,
            compiler_state,
            is_first,
            is_last,
            has_base_fallback,
        )
        compiler_state = clause.compile_clause(
            compiled_program,
            compiler_state,
            program,
            tid,
            head,
            body_root,
            config,
            analysis_scratch,
            max_x_registers,
        )

        if bridge_patch != ENTRY_INVALID_PC:
            compiler_state = clause.emit_base_bridge(
                compiled_program,
                compiler_state,
                bridge_patch,
                base_entry_pc,
            )

        previous_patch = next_patch
        is_first = False
        clause_index = next_clause

    return compiler_state
