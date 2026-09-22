import numpy as np
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiled_query import (
    clear_compiled_query,
    init_compiled_query,
)
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.query import compile_query
from wampy.config import load_config_toml
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


def test_init_and_clear_compiled_query_retain_allocations():
    config = load_config_toml("""
        [frontend.ast]
        max_variables_per_term = 7

        [compiler]
        max_instructions = 11
        max_arity = 3
        """)
    compiled_query = init_compiled_query(config)

    assert compiled_query.code.shape == (11, 4)
    assert compiled_query.code_size.shape == (1,)
    assert compiled_query.code_size[0] == 0
    assert compiled_query.variable_registers.shape == (7,)
    assert compiled_query.variable_name_ids.shape == (7,)
    assert compiled_query.variable_name_ids.dtype == config.frontend.ast.symbol_id_dtype
    assert compiled_query.argument_registers.shape == (3,)
    assert np.all(compiled_query.variable_registers == -1)
    assert np.all(
        compiled_query.variable_name_ids == np.iinfo(config.frontend.ast.symbol_id_dtype).max
    )
    assert np.all(compiled_query.argument_registers == -1)

    compiled_query.code[0] = (1, 2, 3, 4)
    compiled_query.variable_registers[0] = 2
    compiled_query.variable_name_ids[0] = 123
    compiled_query.argument_registers[0] = 1
    compiled_query.code_size[0] = 1

    code = compiled_query.code
    code_size = compiled_query.code_size
    variable_registers = compiled_query.variable_registers
    variable_name_ids = compiled_query.variable_name_ids
    argument_registers = compiled_query.argument_registers
    result = clear_compiled_query(compiled_query)

    assert result is None
    assert compiled_query.code is code
    assert compiled_query.code_size is code_size
    assert compiled_query.variable_registers is variable_registers
    assert compiled_query.variable_name_ids is variable_name_ids
    assert compiled_query.argument_registers is argument_registers
    assert compiled_query.code_size[0] == 0
    assert np.all(compiled_query.code[0] == 0)
    assert np.all(compiled_query.variable_registers == -1)
    assert np.all(
        compiled_query.variable_name_ids == np.iinfo(compiled_query.variable_name_ids.dtype).max
    )
    assert np.all(compiled_query.argument_registers == -1)


def test_compile_query_clears_and_reuses_the_supplied_artifact():
    program, symbol_table = parse("p(a).", empty_symbol_table())
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    assert (
        compile_program(
            program,
            compiled_program,
            compiler_state,
        )
        is None
    )
    buffer = init_compiled_query()
    variable_registers = buffer.variable_registers
    variable_name_ids = buffer.variable_name_ids
    argument_registers = buffer.argument_registers
    code_size = buffer.code_size

    query, _ = parse("?- p(X).", symbol_table)
    assert (
        compile_query(
            query,
            buffer,
            compiled_program,
            compiler_state,
        )
        is None
    )
    first_code_size = int(buffer.code_size[0])
    code = buffer.code

    true_query, _ = parse("?- true.", symbol_table)
    assert (
        compile_query(
            true_query,
            buffer,
            compiled_program,
            compiler_state,
        )
        is None
    )

    assert buffer.code is code
    assert buffer.code_size is code_size
    assert buffer.variable_registers is variable_registers
    assert buffer.variable_name_ids is variable_name_ids
    assert buffer.argument_registers is argument_registers
    assert buffer.code_size[0] == 1
    assert np.all(buffer.code[1:first_code_size] == 0)
    assert np.all(buffer.variable_registers == -1)
    assert np.all(buffer.variable_name_ids == np.iinfo(buffer.variable_name_ids.dtype).max)
    assert np.all(buffer.argument_registers == -1)


def test_compile_program_mutates_the_supplied_artifact_in_place():
    program, _ = parse("p(a).", empty_symbol_table())
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    code = compiled_program.code
    code_size = compiled_program.code_size
    predicate_entry = compiled_program.predicate_entry

    assert compile_program(program, compiled_program, compiler_state) is None

    assert compiled_program.code is code
    assert compiled_program.code_size is code_size
    assert compiled_program.predicate_entry is predicate_entry
    assert compiled_program.code_size.shape == (1,)
    assert compiled_program.code_size.dtype == np.int32
    assert compiled_program.code_size[0] > 0
