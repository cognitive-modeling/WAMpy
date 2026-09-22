"""Reusable WAM instruction buffers produced for executable queries.


parse("?- parent(X, luke).")
        │
        ├─ symbols: symbol ID for "X"
        │
        └─ AST:
             slot 0 -> name ID 103
                    │
                    ▼
compile_query()
        │
        └─ CompiledQuery:
             slot 0 -> name ID 103
             slot 0 -> X register 0
                    │
                    ▼
run()
                    │
                    ▼
decode_bindings(machine, compiled_query, symbols)
                    │
                    ▼
             {"X": "anakin"}

"""

from typing import NamedTuple

import numpy as np

from wampy.config import DEFAULT_CONFIG, WAMConfig

INVALID_REGISTER = -1
# Symbol IDs use an unsigned, configuration-selected dtype.  The arrays use
# that dtype's maximum value as the actual invalid sentinel; this signed value
# remains the conceptual sentinel for callers that need a Python scalar.
INVALID_SYMBOL_ID = -1


class CompiledQuery(NamedTuple):
    """Executable query code and its source-variable register mapping.

    Query instructions use local addresses while being compiled.  Control-flow
    targets are relocated into the machine's virtual code address space before
    compilation completes.  Execution therefore always starts at the
    compiled program's ``code_size[0]``.
    """

    code: np.ndarray  # int32[:, 4], valid prefix is code[:code_size[0]]
    code_size: np.ndarray  # int32[1], also the next instruction while compiling
    variable_registers: np.ndarray  # int16[:],  # variable slot -> result X register
    variable_name_ids: np.ndarray  # variable slot -> symbol-table ID of source name
    argument_registers: np.ndarray  # int16[:], displayed query arg -> X register


def init_compiled_query(
    config: WAMConfig = DEFAULT_CONFIG,
) -> CompiledQuery:
    """Allocate an empty reusable query compilation artifact."""

    max_query_arity = config.compiler.max_arity

    if max_query_arity < 2:
        max_query_arity = 2

    max_variables = config.frontend.ast.max_variables_per_term
    symbol_dtype = config.frontend.ast.symbol_id_dtype
    invalid_symbol_id = np.iinfo(symbol_dtype).max

    return CompiledQuery(
        code=np.empty(
            (config.compiler.max_instructions, 4),
            dtype=np.int32,
        ),
        code_size=np.zeros(1, dtype=np.int32),
        variable_registers=np.full(max_variables, INVALID_REGISTER, dtype=np.int16),
        variable_name_ids=np.full(max_variables, invalid_symbol_id, dtype=symbol_dtype),
        argument_registers=np.full(max_query_arity, INVALID_REGISTER, dtype=np.int16),
    )


def clear_compiled_query(compiled_query: CompiledQuery) -> None:
    """Clear a query artifact while retaining its allocated arrays."""

    compiled_query.code[: compiled_query.code_size[0]].fill(0)
    compiled_query.code_size[0] = 0

    compiled_query.variable_registers.fill(INVALID_REGISTER)
    compiled_query.variable_name_ids.fill(
        np.iinfo(compiled_query.variable_name_ids.dtype).max,
    )
    compiled_query.argument_registers.fill(INVALID_REGISTER)
