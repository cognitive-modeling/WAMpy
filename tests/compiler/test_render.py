import pytest
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.diagnostics import display_compiler_state, render_compiler_state
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import resolve_predicate_slot
from wampy.frontend.symbol_table import (
    empty_symbol_table,
    symbol_table_from_mapping,
)


def test_display_compiler_state_empty(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    symbol_table = empty_symbol_table()

    display_compiler_state(compiled_program, compiler_state, symbol_table)

    assert capsys.readouterr().out == (
        "\n=== Predicate entry points ===\n\n=== WAM instructions ===\n(no instructions)\n"
    )


def test_render_compiler_state_empty_does_not_print(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    symbol_table = empty_symbol_table()

    text = render_compiler_state(compiled_program, compiler_state, symbol_table)

    assert text == (
        "\n=== Predicate entry points ===\n\n=== WAM instructions ===\n(no instructions)"
    )
    assert capsys.readouterr().out == ""


def test_display_compiler_state_prints_entries_and_instructions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    symbol_table = symbol_table_from_mapping({7: "parent"})

    predicate_slot = resolve_predicate_slot(compiler_state, 7)
    compiled_program.predicate_entry[predicate_slot, 2] = 0
    compiled_program.code[0] = [OP.GET_VAR, 1, 0, 0]
    compiled_program.code[1] = [OP.PROCEED, 0, 0, 0]
    compiled_program.code_size[0] = 2
    compiler_state.pc[0] = 2

    display_compiler_state(compiled_program, compiler_state, symbol_table)

    assert capsys.readouterr().out == (
        "\n=== Predicate entry points ===\n"
        "parent/2 -> PC 0\n"
        "\n=== WAM instructions ===\n"
        f"0000: {OP.GET_VAR.name:<10} {1:3d} {0:3d} {0:3d}\n"
        f"0001: {OP.PROCEED.name:<10} {0:3d} {0:3d} {0:3d}\n"
    )


def test_display_compiler_state_uses_symbol_id_as_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    symbol_table = empty_symbol_table()

    predicate_slot = resolve_predicate_slot(compiler_state, 12)
    compiled_program.predicate_entry[predicate_slot, 1] = 0

    display_compiler_state(compiled_program, compiler_state, symbol_table)

    output = capsys.readouterr().out

    assert "12/1 -> PC 0" in output


def test_display_compiler_state_only_prints_emitted_instructions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    symbol_table = empty_symbol_table()

    compiled_program.code[0] = [OP.PROCEED, 0, 0, 0]
    compiled_program.code[1] = [OP.GET_CONST, 99, 0, 0]
    compiled_program.code_size[0] = 1
    compiler_state.pc[0] = 1

    display_compiler_state(compiled_program, compiler_state, symbol_table)

    output = capsys.readouterr().out

    assert "PROCEED" in output
    assert "GET_CONST" not in output
