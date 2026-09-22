"""Compilation of executable query terms."""

import numpy as np

from wampy.compiler.codegen.emitter import emit
from wampy.compiler.codegen.goal import (
    compile_goals_inline,
    count_goals,
    count_naf_levels,
    has_later_cut,
    max_callable_arity,
)
from wampy.compiler.codegen.register_allocation import (
    VAR_UNBOUND,
    encode_unbound_y,
    record_variable_occurrences,
    resolve_variable_register,
)
from wampy.compiler.codegen.term import emit_output_structure
from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiled_query import CompiledQuery, clear_compiled_query
from wampy.compiler.compiler_state import (
    ENTRY_INVALID_PC,
    NO_PREDICATE_SLOT,
    CompilerState,
)
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.config import DEFAULT_CONFIG, WAMConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID


def _query_control_slots(program, term_id, root_node, config):
    """Return ``(has_later_cut, naf_level_count)`` for a query body."""

    return (
        has_later_cut(program, term_id, root_node, config),
        count_naf_levels(program, term_id, root_node, config),
    )


def compile_query(
    program: ASTProgram,
    compiled_query: CompiledQuery,
    compiled_program: CompiledProgram,
    compiler_state: CompilerState,
    config: WAMConfig = DEFAULT_CONFIG,
    term_id: int = 0,
) -> None:
    """Compile one query term into an independent WAM instruction block.

    ``compiled_query`` is cleared before emission so its arrays can be reused.
    Predicate slots are read from ``compiler_state`` but are never allocated or
    changed here. Query control-flow targets are relocated above the immutable
    program code so the interpreter can use one integer PC across both arrays.
    The supplied artifact is mutated in place and this function returns ``None``.
    """

    compiler_config = config.compiler
    max_x_registers = config.runtime.max_x_registers
    query_root = ast.get_query_predicate(program, term_id)

    if query_root == NO_NODE:
        raise ValueError("Query has no callable goal")

    clear_compiled_query(compiled_query)
    code = compiled_query.code
    variable_registers = compiled_query.variable_registers
    variable_name_ids = compiled_query.variable_name_ids
    query_arity = ast.get_child_count(program, term_id, query_root)
    argument_registers = compiled_query.argument_registers

    variable_count = ast.get_variable_capacity(program)
    if variable_count > variable_registers.shape[0]:
        raise ValueError("CompiledQuery variable register capacity is too small")
    if query_arity > argument_registers.shape[0]:
        raise ValueError("CompiledQuery argument register capacity is too small")

    query_symbol = ast.get_node_symbol(program, term_id, query_root)
    if query_symbol == CoreSymID.TRUE:
        if query_arity != 0:
            raise ValueError("true must have arity 0")
    else:
        has_later_cut, naf_level_count = _query_control_slots(
            program, term_id, query_root, compiler_config
        )
        argument_register_count = max_callable_arity(
            program,
            term_id,
            query_root,
            query_root,
            compiler_config,
        )
        if argument_register_count > max_x_registers:
            raise ValueError("Predicate arity exceeds X register capacity")

        if query_symbol not in (
            CoreSymID.CONJUNCTION,
            CoreSymID.NEGATION_AS_FAILURE,
            CoreSymID.CUT,
        ):
            predicate_slot = find_predicate_slot(compiler_state, int(query_symbol))
            if predicate_slot == NO_PREDICATE_SLOT:
                raise RuntimeError("WAM error: UNDEFINED_PREDICATE")
            if compiled_program.predicate_entry[predicate_slot, query_arity] == ENTRY_INVALID_PC:
                raise RuntimeError("WAM error: INVALID_PC")

        # A query is itself a returning caller.  Its variables and displayed
        # argument roots therefore live in Y slots until the answer is copied
        # back to stable X registers immediately before DEALLOCATE.
        var_table = np.full(variable_registers.shape[0], VAR_UNBOUND, dtype=np.int16)
        y_by_variable = np.full(variable_registers.shape[0], -1, dtype=np.int16)
        first = np.full(variable_registers.shape[0], -1, dtype=np.int32)
        last = np.full(variable_registers.shape[0], -1, dtype=np.int32)
        variable_stack = np.empty(ast.get_node_capacity(program), dtype=np.int32)
        record_variable_occurrences(
            program,
            term_id,
            query_root,
            0,
            first,
            last,
            variable_stack,
        )
        query_variable_count = 0
        for variable_slot in range(variable_count):
            if first[variable_slot] < 0:
                continue
            y_by_variable[variable_slot] = query_variable_count
            var_table[variable_slot] = encode_unbound_y(query_variable_count)
            if (
                program.variable_name_ids[term_id, variable_slot]
                != np.iinfo(program.variable_name_ids.dtype).max
            ):
                variable_name_ids[variable_slot] = program.variable_name_ids[
                    term_id,
                    variable_slot,
                ]
            query_variable_count += 1

        output_top = query_arity
        for variable_slot in range(variable_count):
            if y_by_variable[variable_slot] < 0:
                continue
            output_register = -1
            if (
                query_symbol != CoreSymID.CONJUNCTION
                and query_symbol != CoreSymID.NEGATION_AS_FAILURE
            ):
                for argument_index in range(query_arity):
                    argument_node = ast.get_child_at(
                        program,
                        term_id,
                        query_root,
                        argument_index,
                    )
                    argument_symbol = ast.get_node_symbol(program, term_id, argument_node)
                    if (
                        ast.is_variable_symbol(program, argument_symbol)
                        and ast.variable_index(program, argument_symbol) == variable_slot
                    ):
                        output_register = argument_index
                        break

            if output_register < 0:
                if output_top >= max_x_registers:
                    raise ValueError("X register capacity exceeded")
                output_register = output_top
                output_top += 1
            variable_registers[variable_slot] = output_register

        num_permanent = query_variable_count + query_arity
        cut_y = -1
        if has_later_cut:
            cut_y = num_permanent
            num_permanent += 1
        naf_y_base = num_permanent
        num_permanent += naf_level_count
        emit(compiled_query.code, compiled_query.code_size, OP.ALLOCATE)
        if cut_y >= 0:
            emit(compiled_query.code, compiled_query.code_size, OP.GET_LEVEL, cut_y, 1)

        if output_top > argument_register_count:
            next_x = output_top
        else:
            next_x = argument_register_count

        argument_index = 0
        for argument_index in range(query_arity):
            argument_node = ast.get_child_at(
                program,
                term_id,
                query_root,
                argument_index,
            )

            argument_registers[argument_index] = argument_index
            argument_symbol = ast.get_node_symbol(program, term_id, argument_node)
            if ast.is_variable_symbol(program, argument_symbol):
                variable_slot = ast.variable_index(program, argument_symbol)
                reg, is_y, first, next_x = resolve_variable_register(
                    var_table,
                    variable_slot,
                    next_x,
                    max_x_registers,
                )
                if first:
                    op = OP.PUT_Y_VAR if is_y else OP.PUT_VAR
                else:
                    op = OP.PUT_Y_VAL if is_y else OP.PUT_VAL
                emit(
                    compiled_query.code,
                    compiled_query.code_size,
                    op,
                    reg,
                    argument_index,
                )
            elif ast.is_compound_node(program, term_id, argument_node):
                compiler_state, next_x = emit_output_structure(
                    compiled_query,
                    compiler_state,
                    compiled_query.code_size,
                    program,
                    term_id,
                    argument_node,
                    argument_index,
                    var_table,
                    next_x,
                    compiler_config,
                    max_x_registers,
                )
            else:
                emit(
                    compiled_query.code,
                    compiled_query.code_size,
                    OP.PUT_CONST,
                    argument_symbol,
                    argument_index,
                )

            emit(
                compiled_query.code,
                compiled_query.code_size,
                OP.GET_Y_VAR,
                query_variable_count + argument_index,
                argument_index,
            )

        environment_sizes = np.full(
            count_goals(program, term_id, query_root, compiler_config) + 1,
            num_permanent,
            dtype=np.int32,
        )
        compiler_state, _ = compile_goals_inline(
            compiled_query,
            compiler_state,
            compiled_query.code_size,
            program,
            term_id,
            query_root,
            var_table,
            next_x,
            compiler_config,
            max_x_registers,
            False,
            True,
            environment_sizes,
            cut_y,
            naf_y_base,
        )

        for argument_index in range(query_arity):
            emit(
                compiled_query.code,
                compiled_query.code_size,
                OP.PUT_Y_VAL,
                query_variable_count + argument_index,
                argument_registers[argument_index],
            )
        for variable_slot in range(variable_count):
            if y_by_variable[variable_slot] < 0:
                continue
            emit(
                compiled_query.code,
                compiled_query.code_size,
                OP.PUT_Y_VAL,
                y_by_variable[variable_slot],
                variable_registers[variable_slot],
            )
        emit(compiled_query.code, compiled_query.code_size, OP.DEALLOCATE)

    emit(
        compiled_query.code,
        compiled_query.code_size,
        OP.PROCEED,
    )
    query_code_size = int(compiled_query.code_size[0])
    query_base = int(compiled_program.code_size[0])

    # Program targets are already absolute. Only query-local control transfers
    # require relocation into the shared virtual PC space.
    for pc in range(query_code_size):
        op = int(code[pc, 0])
        if op == OP.TRY_ME_ELSE or op == OP.RETRY_ME_ELSE:
            code[pc, 1] += query_base
        elif op == OP.JMP_RETRY:
            code[pc, 1] += query_base
            code[pc, 2] += query_base
        elif op == OP.JMP_TRUST:
            code[pc, 1] += query_base
