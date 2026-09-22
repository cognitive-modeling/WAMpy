"""Focused tests for the compiled-query solution decoding boundary."""

from wampy.api import (
    compile_program,
    compile_query,
    decode_bindings,
    decode_variables,
    empty_symbol_table,
    init_compiled_program,
    init_compiled_query,
    init_compiler_state,
    init_machine,
    parse,
    redo,
    run,
)
from wampy.status import WAMStatus


def _compile(program_source: str, query_source: str):
    symbols = empty_symbol_table()
    program, symbols = parse(program_source, symbols)
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    compile_program(program, compiled_program, compiler_state)

    query, symbols = parse(query_source, symbols)
    compiled_query = init_compiled_query()
    compile_query(query, compiled_query, compiled_program, compiler_state)

    return symbols, compiled_program, compiled_query


def test_low_level_decoders_use_compiled_query_variable_name_ids():
    symbols, compiled_program, compiled_query = _compile(
        "parent(anakin, luke).",
        "?- parent(X, luke).",
    )
    machine = init_machine()

    assert WAMStatus(int(run(machine, compiled_program, compiled_query))) is WAMStatus.SUCCESS
    assert decode_variables(machine, compiled_query, symbols) == ["anakin"]
    assert decode_bindings(machine, compiled_query, symbols) == {"X": "anakin"}


def test_low_level_decoder_returns_two_named_variables():
    symbols, compiled_program, compiled_query = _compile(
        "parent(anakin, luke).",
        "?- parent(X, Y).",
    )
    machine = init_machine()

    assert run(machine, compiled_program, compiled_query) == WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {
        "X": "anakin",
        "Y": "luke",
    }


def test_repeated_query_variable_is_returned_once():
    symbols, compiled_program, compiled_query = _compile(
        "same(a, a).",
        "?- same(X, X).",
    )
    machine = init_machine()

    assert run(machine, compiled_program, compiled_query) == WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {"X": "a"}


def test_anonymous_query_variables_are_not_named_bindings():
    symbols, compiled_program, compiled_query = _compile(
        "parent(anakin, luke).",
        "?- parent(_, X).",
    )
    machine = init_machine()

    assert run(machine, compiled_program, compiled_query) == WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {"X": "luke"}


def test_unresolved_aliases_render_named_variables():
    symbols, compiled_program, compiled_query = _compile(
        "same(X, X).",
        "?- same(A, B).",
    )
    machine = init_machine()

    assert run(machine, compiled_program, compiled_query) == WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {
        "A": "A",
        "B": "A",
    }


def test_low_level_decoder_follows_backtracking():
    symbols, compiled_program, compiled_query = _compile(
        "parent(anakin, luke). parent(padme, luke).",
        "?- parent(X, luke).",
    )
    machine = init_machine()

    status = WAMStatus(int(run(machine, compiled_program, compiled_query)))
    assert status is WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {"X": "anakin"}

    status = WAMStatus(int(redo(machine, compiled_program, compiled_query)))
    assert status is WAMStatus.SUCCESS
    assert decode_bindings(machine, compiled_query, symbols) == {"X": "padme"}


def test_reusing_compiled_query_does_not_leak_variable_names():
    symbols = empty_symbol_table()
    program, symbols = parse("parent(anakin, luke).", symbols)
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    compile_program(program, compiled_program, compiler_state)

    query_x, symbols = parse("?- parent(X, luke).", symbols)
    query_y, symbols = parse("?- parent(Y, luke).", symbols)
    compiled_query = init_compiled_query()

    compile_query(query_x, compiled_query, compiled_program, compiler_state)
    assert int(compiled_query.variable_name_ids[0]) == symbols.id_by_symbol["X"]

    compile_query(query_y, compiled_query, compiled_program, compiler_state)
    assert int(compiled_query.variable_name_ids[0]) == symbols.id_by_symbol["Y"]
    invalid_name_id = int(compiled_query.variable_name_ids.max())
    assert all(int(name_id) == invalid_name_id for name_id in compiled_query.variable_name_ids[1:])
