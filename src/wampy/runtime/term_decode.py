from enum import IntEnum, unique

import numpy as np

from wampy.runtime.machine.heap import deref
from wampy.runtime.tags import TAG
from wampy.status import WAMStatus


@unique
class TermKind(IntEnum):
    VAR = 0
    CONST = 1
    STRUCT = 2


def read_term_view(state, memory, addr):
    heap = memory.heap
    addr = deref(state, memory, addr)
    if addr >= state.H[0]:
        return (
            np.int64(WAMStatus.CORRUPT_ENVIRONMENT),
            np.uint8(255),
            np.int64(0),
            np.int64(0),
            np.int64(0),
        )

    tag = heap.tags[addr]

    if tag == TAG.REF:
        return (
            np.int64(WAMStatus.SUCCESS),
            np.uint8(TermKind.VAR.value),
            np.int64(addr),
            np.int64(0),
            np.int64(0),
        )

    if tag == TAG.CON:
        return (
            np.int64(WAMStatus.SUCCESS),
            np.uint8(TermKind.CONST.value),
            np.int64(heap.cells[addr]),
            np.int64(0),
            np.int64(0),
        )

    if tag == TAG.STR:
        fun = heap.cells[addr]
        symbol = heap.cells[fun]
        ar = heap.cells[fun + 1]
        base = addr + 1
        return (
            np.int64(WAMStatus.SUCCESS),
            np.uint8(TermKind.STRUCT.value),
            np.int64(symbol),
            np.int64(base),
            np.int64(ar),
        )

    return (
        np.int64(WAMStatus.CORRUPT_ENVIRONMENT),
        np.uint8(255),  # sentinel kind value
        np.int64(0),
        np.int64(0),
        np.int64(0),
    )


def read_term(state, memory, addr):
    st, kind, a, b, ar = read_term_view(state, memory, addr)
    st = WAMStatus(int(st))
    if st != WAMStatus.SUCCESS:
        raise RuntimeError(f"read_term failed: {st.name}")

    kind = int(kind)
    if kind == int(TermKind.VAR):
        return ("VAR", int(a))

    if kind == int(TermKind.CONST):
        return ("CONST", int(a))

    # STRUCT
    symbol_id = int(a)
    base = int(b)
    arity = int(ar)
    args = [read_term(state, memory, base + i) for i in range(arity)]
    return ("STRUCT", symbol_id, args)
