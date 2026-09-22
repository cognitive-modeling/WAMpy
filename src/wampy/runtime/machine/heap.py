"""WAM heap storage and cell-allocation primitives."""

from typing import NamedTuple

import numpy as np

from wampy.runtime.tags import TAG


class HeapMemory(NamedTuple):
    """Heap cells and their tags, stored as parallel arrays."""

    cells: np.ndarray
    tags: np.ndarray


def init_heap(size):
    """Allocate a self-referencing REF-initialized heap."""

    return HeapMemory(
        cells=np.arange(size, dtype=np.uint16),
        tags=np.full(size, TAG.REF, dtype=np.uint8),
    )


def deref(state, memory, address):
    """Follow reference cells until reaching an unbound or non-reference cell."""

    heap = memory.heap

    if address >= state.H[0]:
        return address

    seen = 0
    limit = state.H[0]
    while heap.tags[address] == TAG.REF and heap.cells[address] != address:
        address = heap.cells[address]
        if address >= state.H[0]:
            return address
        seen += 1
        if seen > limit:
            return state.H[0]

    return address


def make_var(state, memory):
    """Allocate a fresh self-referencing variable cell."""

    heap = memory.heap
    h = state.H[0]
    if h == np.uint16(65535):
        raise IndexError("HEAP_POINTER_OVERFLOW uint16")
    if h >= heap.cells.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_var")
    heap.cells[h] = h
    heap.tags[h] = TAG.REF
    state.H[0] = h + 1
    return h


def make_const(state, memory, value):
    """Allocate a constant cell."""

    heap = memory.heap
    h = state.H[0]
    if h == np.uint16(65535):
        raise IndexError("HEAP_POINTER_OVERFLOW uint16")
    if h >= heap.cells.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_const")
    heap.cells[h] = value
    heap.tags[h] = TAG.CON
    state.H[0] = h + 1
    return h


def make_functor(state, memory, symbol_id, arity):
    """Allocate a functor descriptor and its arity cell."""

    heap = memory.heap
    h = state.H[0]
    if h + 1 >= heap.cells.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_functor")
    heap.cells[h] = symbol_id
    heap.tags[h] = TAG.FUN
    heap.cells[h + 1] = arity
    heap.tags[h + 1] = TAG.FUN
    state.H[0] = h + 2
    return h


def make_structure(state, memory, functor_pos, args):
    """Allocate a structure header followed by its argument cells."""

    heap = memory.heap
    h = state.H[0]
    need = 1 + len(args)
    if h + need - 1 >= heap.cells.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_structure")
    heap.cells[h] = functor_pos
    heap.tags[h] = TAG.STR
    state.H[0] = h + 1
    for arg in args:
        heap.cells[state.H[0]] = arg
        heap.tags[state.H[0]] = TAG.REF
        state.H[0] += 1
    return h
