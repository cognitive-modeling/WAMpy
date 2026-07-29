"""Higher-level AST rewrites helpers for the array-backed AST."""

__all__ = [
    "copy_and_override_term",
]

import numpy as np
from numba import jit

from wampy.frontend.ast.program import Program, NONE


@jit(cache=True)
def copy_and_override_term(program, dst_term, src_term):
    max_nodes = program.symbol.shape[1]
    max_slots = program.children.shape[2]
    max_free = program.free_blocks.shape[1]

    for n in range(max_nodes):
        program.symbol[dst_term, n] = program.symbol[src_term, n]

    for n in range(max_nodes):
        for s in range(max_slots):
            program.children[dst_term, n, s] = program.children[src_term, n, s]

    for i in range(max_free):
        program.free_blocks[dst_term, i] = program.free_blocks[src_term, i]

    program.free_top[dst_term] = program.free_top[src_term]
