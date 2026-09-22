"""WAM trail allocation, recording, and unwinding."""

import numpy as np

from wampy.runtime.tags import TAG


def init_trail(size):
    """Allocate trail storage."""

    return np.empty(size, dtype=np.uint16)


def trail_var(state, memory, address) -> None:
    """Record a variable address so backtracking can unbind it."""

    trail = memory.trail
    top = state.TR[0]
    if top >= trail.shape[0]:
        raise IndexError("TRAIL_OVERFLOW")
    trail[top] = address
    state.TR[0] = top + 1


def unwind_trail(state, memory, target) -> None:
    """Undo bindings down to ``target`` and restore self references."""

    heap = memory.heap
    trail = memory.trail
    while state.TR[0] > target:
        state.TR[0] -= 1
        address = trail[state.TR[0]]
        heap.cells[address] = address
        heap.tags[address] = TAG.REF
