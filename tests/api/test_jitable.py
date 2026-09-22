"""Tests for functions registered through the Numba-jitable WAMpy API."""

import importlib
import inspect

import numba.extending
import pytest
import wampy.api.jitable as jitable
from numba import jit
from pytest import MonkeyPatch
from wampy.api.jitable import (
    _empty_symbol_table_resolved,
    clear_compiled_program,
    compile_program,
    compile_query,
    init_compiled_program,
    init_compiled_query,
    init_compiler_state,
    init_machine,
    redo,
    render_compiler_state,
    run,
    undo_hypothesis,
)
from wampy.compiler.opcodes import OP
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table
from wampy.status import WAMStatus


@pytest.mark.requires_numba_jit
def test_render_compiler_state_is_callable_from_numba() -> None:
    @jit
    def call_render() -> str:
        compiler_state = init_compiler_state()
        compiled_program = init_compiled_program()
        symbol_table = _empty_symbol_table_resolved(100, 200)
        symbol_table.symbol_by_id[7] = "parent"

        compiler_state.predicate_count[0] = 1
        compiler_state.symbol_by_predicate_slot[0] = 7
        compiled_program.predicate_entry[0, 2] = 0
        compiled_program.code[0, 0] = OP.GET_VAR
        compiled_program.code[0, 1] = 1
        compiled_program.code[0, 2] = 0
        compiled_program.code[0, 3] = 0
        compiled_program.code_size[0] = 1
        compiler_state.pc[0] = 1

        return render_compiler_state(compiled_program, compiler_state, symbol_table)

    assert call_render() == (
        "\n=== Predicate entry points ===\n"
        "parent/2 -> PC 0\n"
        "\n=== WAM instructions ===\n"
        f"0000: {OP.GET_VAR.name:<10} {1:3d} {0:3d} {0:3d}"
    )
    assert call_render.nopython_signatures


@pytest.mark.requires_numba_jit
def test_compile_program_accepts_python_created_state_from_numba() -> None:
    base_program, symbol_table = parse("p(a).", empty_symbol_table())
    extension_program, _ = parse("p(b).", symbol_table)
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()

    compile_program(base_program, compiled_program, compiler_state)
    base_code_size = compiled_program.code_size[0]

    @jit
    def jit_compile_program(program, compiled_program, state) -> None:
        compile_program(program, compiled_program, state)

    jit_compile_program(extension_program, compiled_program, compiler_state)

    assert compiled_program.code_size[0] > base_code_size
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 1
    undo_hypothesis(compiler_state, compiled_program)
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 0
    jit_compile_program(extension_program, compiled_program, compiler_state)
    assert compiled_program.code_size[0] > base_code_size
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 1
    assert jit_compile_program.nopython_signatures


@pytest.mark.requires_numba_jit
def test_clear_compiled_program_is_callable_from_numba() -> None:
    compiled_program = init_compiled_program()
    compiled_program.code[0] = (OP.GET_CONST, 7, 0, 0)
    compiled_program.code_size[0] = 1
    compiled_program.predicate_entry[0, 1] = 0

    @jit
    def jit_clear_compiled_program(program):
        return clear_compiled_program(program)

    result = jit_clear_compiled_program(compiled_program)

    assert result.code_size[0] == 0
    assert compiled_program.code_size[0] == 0
    assert tuple(compiled_program.code[0]) == (OP.GET_CONST, 7, 0, 0)
    assert (compiled_program.predicate_entry == -1).all()
    assert jit_clear_compiled_program.nopython_signatures


@pytest.mark.requires_numba_jit
def test_compile_query_is_callable_from_numba() -> None:
    program, symbol_table = parse("p(a).", empty_symbol_table())

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()

    compile_program(program, compiled_program, compiler_state)
    query_program, _ = parse("?- p(X).", symbol_table)

    @jit
    def jit_compile_query(query_ast, compiled_query, compiled_program, state) -> None:
        compile_query(query_ast, compiled_query, compiled_program, state)

    compiled_query = init_compiled_query()
    jit_compile_query(query_program, compiled_query, compiled_program, compiler_state)

    assert OP.CALL in compiled_query.code[: compiled_query.code_size[0], 0]
    assert OP(compiled_query.code[compiled_query.code_size[0] - 1, 0]) is OP.PROCEED
    assert compiled_query.variable_name_ids[0] == symbol_table.id_by_symbol["X"]
    assert jit_compile_query.nopython_signatures


@pytest.mark.requires_numba_jit
def test_runtime_execution_is_callable_from_numba() -> None:
    program, symbol_table = parse("p(a).", empty_symbol_table())
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    compile_program(program, compiled_program, compiler_state)

    query_program, _ = parse("?- p(a).", symbol_table)
    compiled_query = init_compiled_query()
    compile_query(
        query_program,
        compiled_query,
        compiled_program,
        compiler_state,
    )
    machine = init_machine()

    @jit
    def run_jitted(machine, compiled_program, compiled_query):
        return run(machine, compiled_program, compiled_query)

    @jit
    def redo_jitted(machine, compiled_program, compiled_query):
        return redo(machine, compiled_program, compiled_query)

    assert (
        WAMStatus(int(run_jitted(machine, compiled_program, compiled_query))) is WAMStatus.SUCCESS
    )
    assert run_jitted.nopython_signatures
    assert (
        WAMStatus(int(redo_jitted(machine, compiled_program, compiled_query)))
        is WAMStatus.EXHAUSTED
    )
    assert redo_jitted.nopython_signatures


def test_all_jitable_module_functions_are_registered(monkeypatch: MonkeyPatch) -> None:
    """Every function imported by wampy.api.jitable should be registered with Numba."""

    registered = set()

    def record_register_jitable(function=None, **_options):
        if function is None:
            return record_register_jitable

        registered.add(function)
        return function

    try:
        with monkeypatch.context() as registration_patch:
            registration_patch.setattr(
                numba.extending,
                "register_jitable",
                record_register_jitable,
            )
            importlib.reload(jitable)

            module_functions = {
                name: function
                for name, function in inspect.getmembers(jitable, inspect.isfunction)
                if name != "register_jitable"
            }
    finally:
        importlib.reload(jitable)

    missing = {name for name, function in module_functions.items() if function not in registered}

    assert not missing, f"Functions are not registered jitable: {sorted(missing)}"
    assert registered == set(module_functions.values())
