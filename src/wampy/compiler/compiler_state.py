"""Mutable state used while compiling WAM programs."""

from typing import NamedTuple

import numpy as np

from wampy.compiler.compiled_program import CompiledProgram
from wampy.config import DEFAULT_CONFIG, WAMConfig

ENTRY_INVALID_PC = -1
NO_PREDICATE_SLOT = -1


class CompilerState(NamedTuple):
    predicate_symbols_sorted: np.ndarray  # int64[:], sorted raw symbol IDs
    predicate_slots_sorted: np.ndarray  # int32[:], slots for sorted symbols
    symbol_by_predicate_slot: np.ndarray  # int64[:], slot -> raw symbol ID
    predicate_count: np.ndarray  # int32[1]
    pc: np.ndarray  # int32[1]

    # State required to remove the currently compiled hypothesis.
    pc_onto: np.ndarray  # int32[1] # TODO rename to hypothesis_pc
    hypothesis_predicate_entry_undo: np.ndarray  # int32[:, 3]
    hypothesis_predicate_entry_undo_count: np.ndarray  # int32[1]


def init_compiler_state(
    config: WAMConfig = DEFAULT_CONFIG,
) -> CompilerState:
    """Allocate reusable mutable state for compiling a WAM program."""

    compiler_config = config.compiler

    pc = np.zeros(1, dtype=np.int32)
    pc_onto = np.zeros(1, dtype=np.int32)

    predicate_symbols_sorted = np.full(
        compiler_config.max_predicates,
        -1,
        dtype=np.int64,
    )
    predicate_slots_sorted = np.full(
        compiler_config.max_predicates,
        NO_PREDICATE_SLOT,
        dtype=np.int32,
    )
    symbol_by_predicate_slot = np.full(
        compiler_config.max_predicates,
        -1,
        dtype=np.int64,
    )
    predicate_count = np.zeros(1, dtype=np.int32)

    hypothesis_predicate_entry_undo = np.empty(
        (compiler_config.max_hypothesis_predicate_changes, 3),
        dtype=np.int32,
    )
    hypothesis_predicate_entry_undo_count = np.zeros(1, dtype=np.int32)

    return CompilerState(
        predicate_symbols_sorted=predicate_symbols_sorted,
        predicate_slots_sorted=predicate_slots_sorted,
        symbol_by_predicate_slot=symbol_by_predicate_slot,
        predicate_count=predicate_count,
        pc=pc,
        pc_onto=pc_onto,
        hypothesis_predicate_entry_undo=hypothesis_predicate_entry_undo,
        hypothesis_predicate_entry_undo_count=hypothesis_predicate_entry_undo_count,
    )


def clear_compiler_state(
    compiler_state: CompilerState,
    start_pc: int = 0,
) -> CompilerState:
    """Clear compiler state in place and retain all allocated buffers.

    ``start_pc`` is the instruction address at which the next compilation
    starts. The onto boundary is owned by the caller and is intentionally
    left unchanged.
    """

    compiler_state.pc[0] = start_pc

    compiler_state.predicate_symbols_sorted.fill(-1)
    compiler_state.predicate_slots_sorted.fill(NO_PREDICATE_SLOT)
    compiler_state.symbol_by_predicate_slot.fill(-1)
    compiler_state.predicate_count[0] = 0

    compiler_state.hypothesis_predicate_entry_undo_count[0] = 0

    return compiler_state


def keep_hypothesis(
    compiler_state: CompilerState,
) -> None:
    """Keep the current hypothesis as part of the persistent program."""

    compiler_state.hypothesis_predicate_entry_undo_count[0] = 0
    compiler_state.pc_onto[0] = compiler_state.pc[0]


def undo_hypothesis(
    compiler_state: CompilerState,
    compiled_program: CompiledProgram,
) -> None:
    """Remove the currently compiled hypothesis."""

    undo = compiler_state.hypothesis_predicate_entry_undo
    count = compiler_state.hypothesis_predicate_entry_undo_count[0]

    for index in range(count):
        predicate_slot = undo[index, 0]
        arity = undo[index, 1]
        old_pc = undo[index, 2]
        compiled_program.predicate_entry[
            predicate_slot,
            arity,
        ] = old_pc

    compiler_state.hypothesis_predicate_entry_undo_count[0] = 0

    base_pc = compiler_state.pc_onto[0]
    compiler_state.pc[0] = base_pc
    compiled_program.code_size[0] = base_pc


def record_hypothesis_predicate_entry_undo(
    compiler_state: CompilerState,
    predicate_slot: int,
    arity: int,
    old_pc: int,
) -> None:
    """Append one predicate-entry value to the current hypothesis undo log."""

    count = compiler_state.hypothesis_predicate_entry_undo_count[0]
    undo = compiler_state.hypothesis_predicate_entry_undo

    if count >= undo.shape[0]:
        raise ValueError("max_hypothesis_predicate_changes is too small")

    undo[count, 0] = predicate_slot
    undo[count, 1] = arity
    undo[count, 2] = old_pc
    compiler_state.hypothesis_predicate_entry_undo_count[0] = count + 1
