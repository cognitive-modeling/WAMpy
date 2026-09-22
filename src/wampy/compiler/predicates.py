"""Stable compact predicate-slot lookup and allocation."""

import numpy as np

from wampy.compiler.compiler_state import (
    NO_PREDICATE_SLOT,
    CompilerState,
)


def predicate_lookup_position(compiler_state: CompilerState, symbol_id: int) -> int:
    """Return the insertion position for a raw predicate symbol ID."""

    raw_symbol_id = np.int64(symbol_id)
    count = int(compiler_state.predicate_count[0])

    low = 0
    high = count
    while low < high:
        middle = (low + high) // 2
        middle_symbol = compiler_state.predicate_symbols_sorted[middle]

        if middle_symbol < raw_symbol_id:
            low = middle + 1
        else:
            high = middle

    return low


def find_predicate_slot(compiler_state: CompilerState, symbol_id: int) -> int:
    """Return a callable symbol's existing slot without allocating one."""

    raw_symbol_id = np.int64(symbol_id)
    count = int(compiler_state.predicate_count[0])
    position = predicate_lookup_position(compiler_state, raw_symbol_id)

    if position < count and compiler_state.predicate_symbols_sorted[position] == raw_symbol_id:
        return int(compiler_state.predicate_slots_sorted[position])

    return NO_PREDICATE_SLOT


def resolve_predicate_slot(compiler_state: CompilerState, symbol_id: int) -> int:
    """Return or allocate the stable compact slot for a callable symbol."""

    raw_symbol_id = np.int64(symbol_id)
    count = int(compiler_state.predicate_count[0])
    position = predicate_lookup_position(compiler_state, raw_symbol_id)

    if position < count and compiler_state.predicate_symbols_sorted[position] == raw_symbol_id:
        return int(compiler_state.predicate_slots_sorted[position])

    if count >= compiler_state.symbol_by_predicate_slot.shape[0]:
        raise ValueError("max_predicates is too small for the program")

    index = count
    while index > position:
        compiler_state.predicate_symbols_sorted[index] = compiler_state.predicate_symbols_sorted[
            index - 1
        ]
        compiler_state.predicate_slots_sorted[index] = compiler_state.predicate_slots_sorted[
            index - 1
        ]
        index -= 1

    compiler_state.predicate_symbols_sorted[position] = raw_symbol_id
    compiler_state.predicate_slots_sorted[position] = count
    compiler_state.symbol_by_predicate_slot[count] = raw_symbol_id
    compiler_state.predicate_count[0] = count + 1
    return count
