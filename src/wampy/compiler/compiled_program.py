"""Reusable runtime artifacts produced by the WAM compiler."""

from typing import NamedTuple

import numpy as np

from wampy.config import DEFAULT_CONFIG, WAMConfig


class CompiledProgram(NamedTuple):
    """Executable WAM program image consumed by the interpreter.

            ``code`` is a preallocated instruction area whose valid prefix is
            ``code[:code_size[0]]``. ``predicate_entry`` maps each compiled predicate
            slot and arity to the instruction address at which execution of that
            predicate begins.

            A newly initialized program has ``code_size[0] == 0`` and no valid
            predicate entries. The allocated arrays are retained so the compiler
            can fill or extend the program without reallocating the underlying
            storage.


                      arity
                 0    1    2    3
               ┌───────────────────
    slot 0     │ -1   -1   10   -1
    slot 1     │ -1   25   -1   -1
    slot 2     │ -1   -1   -1   40

    """

    code: np.ndarray  # int32[:, 4]
    code_size: np.ndarray  # int32[1], valid emitted code length
    predicate_entry: np.ndarray  # int32[:, :]


INVALID_PC = -1


def init_compiled_program(
    config: WAMConfig = DEFAULT_CONFIG,
) -> CompiledProgram:
    """Allocate an empty reusable WAM program image."""

    compiler_config = config.compiler

    code = np.empty(
        (compiler_config.max_instructions, 4),
        dtype=np.int32,
    )

    predicate_entry = np.full(
        (
            compiler_config.max_predicates,
            compiler_config.max_arity + 1,
        ),
        INVALID_PC,
        dtype=np.int32,
    )

    return CompiledProgram(
        code=code,
        code_size=np.zeros(1, dtype=np.int32),
        predicate_entry=predicate_entry,
    )


def clear_compiled_program(
    compiled_program: CompiledProgram,
) -> CompiledProgram:
    """Reset a compiled program while retaining its allocated buffers."""

    compiled_program.code_size[0] = 0
    compiled_program.predicate_entry.fill(INVALID_PC)

    return compiled_program
