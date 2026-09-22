"""


CompilerState           --writes->          CompiledProgram
─────────────                               ───────────────
pc                                          code
compiler metadata                           code_size
predicate lookup                            predicate_entry
predicate_count
pc_onto
hypothesis_predicate_entry_undo
hypothesis_predicate_entry_undo_count



0                              base_code_size                    code_size
│────────────────────────────────────│──────────────────────────────│
.             base                            onto program


CompiledProgram ──┐
                  │
                  ▼
                    run() + Machine
                  ▲
                  │
CompiledQuery ────┘
      │
      └── entry_pc / result metadata

"""

from .compiled_program import (
    CompiledProgram,
    clear_compiled_program,
    init_compiled_program,
)
from .compiled_query import (
    CompiledQuery,
    clear_compiled_query,
    init_compiled_query,
)
from .compiler import compile_program
from .compiler_state import (
    ENTRY_INVALID_PC,
    CompilerState,
    clear_compiler_state,
    init_compiler_state,
    undo_hypothesis,
)
from .query import compile_query

__all__ = [
    "ENTRY_INVALID_PC",
    "CompiledProgram",
    "CompiledQuery",
    "CompilerState",
    "clear_compiled_program",
    "clear_compiled_query",
    "clear_compiler_state",
    "compile_program",
    "compile_query",
    "init_compiled_program",
    "init_compiled_query",
    "init_compiler_state",
    "undo_hypothesis",
]
