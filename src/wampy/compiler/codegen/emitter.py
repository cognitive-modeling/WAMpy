"""Low-level WAM instruction emission and patching."""

import numpy as np


def emit(
    code: np.ndarray,
    code_size: np.ndarray,
    op: int,
    a: int = 0,
    b: int = 0,
    c: int = 0,
) -> None:
    pc = code_size[0]
    code[pc, 0] = op
    code[pc, 1] = a
    code[pc, 2] = b
    code[pc, 3] = c
    code_size[0] = pc + 1


def patch_instruction(code: np.ndarray, pc: int, operand: int) -> None:
    code[pc, 1] = operand
