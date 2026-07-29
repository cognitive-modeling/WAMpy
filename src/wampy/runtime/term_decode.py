from enum import IntEnum, unique

import numpy as np
from numba import jit, types
from numba.typed import List

from wampy.status import WAMStatus
from wampy.runtime.stack import TAG, deref


@unique
class TermKind(IntEnum):
    VAR = 0
    CONST = 1
    STRUCT = 2


@jit(cache=True)
def read_term_view(stack, addr):
    addr = deref(stack, addr)
    if addr >= stack.heap_top[0]:
        return (
            np.int64(WAMStatus.CORRUPT_ENVIRONMENT),
            np.uint8(255),
            np.int64(0),
            np.int64(0),
            np.int64(0),
        )

    tag = stack.tags[addr]

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
            np.int64(stack.heap[addr]),
            np.int64(0),
            np.int64(0),
        )

    if tag == TAG.STR:
        fun = stack.heap[addr]
        symbol = stack.heap[fun]
        ar = stack.heap[fun + 1]
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


def read_term(stack, addr):
    st, kind, a, b, ar = read_term_view(stack, addr)
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
    args = [read_term(stack, base + i) for i in range(arity)]
    return ("STRUCT", symbol_id, args)


TERM = types.DeferredType()
TERM_LIST = types.ListType(TERM)
TERM_TUPLE = types.Tuple((types.int8, types.int64, TERM_LIST))
TERM.define(TERM_TUPLE)


@jit(cache=True)
def _empty_term_list():
    return List.empty_list(TERM)


@jit(cache=True)
def read_term_jit(stack, addr):
    """
    Nopython deep-copy term reconstruction.
    Returns: (kind:int8, value:int64, args: List[Term])
    """
    status, kind, a, b, ar = read_term_view(stack, addr)
    print(status, kind, a, b)

    # Should not happen in normal operation; return a sentinel var if it does.
    if status != WAMStatus.SUCCESS:
        return (np.int8(TermKind.VAR.value), np.int64(-1), _empty_term_list())

    k = kind

    if k == TermKind.VAR.value:
        return (np.int8(TermKind.VAR.value), np.int64(a), _empty_term_list())

    if k == TermKind.CONST.value:
        return (np.int8(TermKind.CONST.value), np.int64(a), _empty_term_list())

    # STRUCT
    # a = symbol_id, b = base addr of args, ar = arity
    args = _empty_term_list()
    base = int(b)
    arity = int(ar)
    for i in range(arity):
        args.append(read_term_jit(stack, base + i))

    return (np.int8(TermKind.STRUCT.value), np.int64(a), args)
