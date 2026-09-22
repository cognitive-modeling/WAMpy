import numpy as np
import pytest
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.diagnostics import opcode_trace
from wampy.compiler.opcodes import OP


@pytest.fixture
def compiled_program():
    compiled_program = init_compiled_program()
    emitted_opcodes = (OP.GET_VAR, OP.PUT_VAL, OP.PROCEED)

    compiled_program.code[: len(emitted_opcodes), 0] = np.array(
        emitted_opcodes,
        dtype=np.int32,
    )
    compiled_program.code[len(emitted_opcodes) :, 0] = -1

    compiled_program.code_size[0] = len(emitted_opcodes)
    return compiled_program


def test_opcode_trace_default_range_returns_all_emitted_opcodes(compiled_program):
    assert opcode_trace(compiled_program) == (OP.GET_VAR, OP.PUT_VAL, OP.PROCEED)


def test_opcode_trace_valid_subrange_returns_selected_opcodes(compiled_program):
    assert opcode_trace(compiled_program, start=1, stop=3) == (OP.PUT_VAL, OP.PROCEED)


def test_opcode_trace_empty_range_returns_empty_tuple(compiled_program):
    assert opcode_trace(compiled_program, start=2, stop=2) == ()


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (-1, None),
        (0, -1),
        (2, 1),
        (4, 4),
        (0, 4),
    ],
)
def test_opcode_trace_rejects_invalid_ranges(compiled_program, start, stop):
    emitted = compiled_program.code_size[0]
    effective_stop = emitted if stop is None else stop

    with pytest.raises(
        ValueError,
        match=f"start={start}, stop={effective_stop}, emitted={emitted}",
    ):
        opcode_trace(compiled_program, start=start, stop=stop)


def test_opcode_trace_does_not_read_unused_instruction_cells(compiled_program):
    assert opcode_trace(compiled_program, start=0, stop=compiled_program.code_size[0]) == (
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.PROCEED,
    )
