"""Functional WAMpy API."""

from wampy.frontend.parser import load_prolog, parse  # TODO: Make jitable
from wampy.frontend.solution import (
    decode_bindings,
    decode_term,
    decode_variables,
)  # TODO: Make jitable
from wampy.frontend.symbol_table import empty_symbol_table  # TODO: Make jitable
from wampy.status import WAMStatus

from . import ast
from .jitable import (
    CompiledProgram,
    CompiledQuery,
    clear_compiled_program,
    clear_compiled_query,
    compile_program,
    compile_query,
    display,
    init_compiled_program,
    init_compiled_query,
    init_compiler_state,
    init_machine,
    init_symbol_table,
    redo,
    render_term_line_jit,
    reset_machine,
    run,
    term_label_jit,
    undo_hypothesis,
)

__all__ = [
    "CompiledProgram",
    "CompiledQuery",
    "WAMStatus",
    "ast",
    "clear_compiled_program",
    "clear_compiled_query",
    "compile_program",
    "compile_query",
    "decode_bindings",
    "decode_term",
    "decode_variables",
    "display",
    "empty_symbol_table",
    "init_compiled_program",
    "init_compiled_query",
    "init_compiler_state",
    "init_machine",
    "init_symbol_table",
    "load_prolog",
    "parse",
    "redo",
    "render_term_line_jit",
    "reset_machine",
    "run",
    "term_label_jit",
    "undo_hypothesis",
]
