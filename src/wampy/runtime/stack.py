"""
WAM-style Unificationn (NumPy Implementation)

This module implements a minimal Prolog-like trailing and backtracking mechanism
using NumPy. It models core concepts from the Warren Abstract Machine (WAM),
including heap cells, logical variables, unification via destructive update,
a trail stack, choice points, and backtracking (untrailing).


## Example

Suppose you create the term:

    f(X, Y)

The heap will contain something like this:

    STR → FUN(f, 2)
    -----------------------------------------------------------------
    i:   TAG.STR   heap[i]   = j
    j:   TAG.FUN   heap[j]   = "f" (ID)  "symbol" of the function as id
    j+1: TAG.FUN   heap[j+1] = 2         "args" of the structure
    j+2: TAG.REF   heap[j+2] = X
    j+3: TAG.REF   heap[j+3] = Y

Visually:

    0   i ──STR: j
    1   j    ├─ FUN: "f"
    2   j+1  └─ FUN: arity = 2
    3        j+2 ├─ REF: ──► X
    4        j+3 └─ REF: ──► Y


"""

from collections import namedtuple
from enum import IntEnum, unique

import numpy as np
from numba import jit

from wampy.config import WAMConfig, DEFAULT_CONFIG


@unique
class TAG(IntEnum):
    REF = 0  # variable reference
    CON = 1  # constant (atom, number)
    STR = 2  # structure pointer
    FUN = 3  # functor header (arity + symbol)


Stack = namedtuple(
    "Stack",
    [
        "heap",
        "tags",
        "trail",
        "trail_top",
        "choice_points",  # trail snapshot
        "choice_pcs",  # alternative PC
        "choice_cps",  # saved cp
        "choice_call_tops",  # saved call-frame head
        "choice_call_alloc_tops",  # saved call-frame allocator top
        "choice_heap_tops",  # saved heap_top
        "choice_X",
        "cp_top",
        "call_pcs",  # return PCs for persistent call frames
        "call_prev",  # parent frame index
        "call_X",  # caller X snapshots
        "call_top",  # current call-frame head
        "call_alloc_top",  # next free persistent call-frame slot
        "heap_top",
        "u_stack",
        "u_top",
        "struct_stack_size",
        "step_limit",
        "unify_step_limit",
    ],
)


@jit(cache=True)
def init_stack(config: WAMConfig = DEFAULT_CONFIG):
    """
    Initialize and return a fresh WAM stack.

    Note:
        - We store ``(t1, t2)`` pairs flattened:
        u_stack[2*i]     = t1
        u_stack[2*i + 1] = t2

    """
    max_x = config.limits.max_x

    return Stack(
        heap=np.arange(config.stack.heap_size, dtype=np.uint16),
        tags=np.full(config.stack.heap_size, TAG.REF, dtype=np.uint8),
        trail=np.empty(config.stack.trail_size, dtype=np.uint16),
        trail_top=np.zeros(1, dtype=np.uint16),
        choice_points=np.empty(config.stack.cp_size, dtype=np.uint16),
        choice_pcs=np.empty(config.stack.cp_size, dtype=np.int32),
        choice_cps=np.empty(config.stack.cp_size, dtype=np.int32),
        choice_call_tops=np.empty(config.stack.cp_size, dtype=np.int32),
        choice_call_alloc_tops=np.empty(config.stack.cp_size, dtype=np.int32),
        choice_heap_tops=np.empty(config.stack.cp_size, dtype=np.uint16),
        choice_X=np.empty((config.stack.cp_size, max_x), dtype=np.uint16),
        cp_top=np.zeros(1, dtype=np.int32),
        call_pcs=np.empty(config.stack.cp_size, dtype=np.int32),
        call_prev=np.empty(config.stack.cp_size, dtype=np.int32),
        call_X=np.empty((config.stack.cp_size, max_x), dtype=np.uint16),
        call_top=np.full(1, -1, dtype=np.int32),
        call_alloc_top=np.zeros(1, dtype=np.int32),
        heap_top=np.zeros(1, dtype=np.uint16),
        u_stack=np.empty(config.stack.unify_stack_size * 2, dtype=np.uint16),
        u_top=np.zeros(1, dtype=np.uint16),
        struct_stack_size=config.limits.max_struct,
        step_limit=config.solver.max_steps,
        unify_step_limit=config.solver.max_unify_steps,
    )


@jit(cache=True)
def reset_stack(stack):
    stack.heap_top[0] = 0
    stack.trail_top[0] = 0
    stack.u_top[0] = 0
    stack.cp_top[0] = 0
    stack.call_top[0] = -1
    stack.call_alloc_top[0] = 0


# ---------------------------------------------------------------
# Heap primitives
# ---------------------------------------------------------------


@jit(cache=True)
def deref(stack, x):
    # GUARD: invalid heap address
    if x >= stack.heap_top[0]:
        return x

    seen = 0
    limit = stack.heap_top[0]
    while stack.tags[x] == TAG.REF and stack.heap[x] != x:
        x = stack.heap[x]
        if x >= stack.heap_top[0]:
            return x
        seen += 1
        if seen > limit:
            return stack.heap_top[0]

    return x


@jit(cache=True)
def trail_var(stack, x):
    t = stack.trail_top[0]
    if t >= stack.trail.shape[0]:
        raise IndexError("TRAIL_OVERFLOW")
    stack.trail[t] = x
    stack.trail_top[0] = t + 1


@jit(cache=True)
def make_var(stack):
    h = stack.heap_top[0]
    if h == np.uint16(65535):
        raise IndexError("HEAP_POINTER_OVERFLOW uint16")
    if h >= stack.heap.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_var")
    stack.heap[h] = h
    stack.tags[h] = TAG.REF
    stack.heap_top[0] = h + 1
    return h


@jit(cache=True)
def make_const(stack, value):
    h = stack.heap_top[0]
    if h == np.uint16(65535):
        raise IndexError("HEAP_POINTER_OVERFLOW uint16")
    if h >= stack.heap.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_const")
    stack.heap[h] = value
    stack.tags[h] = TAG.CON
    stack.heap_top[0] = h + 1
    return h


@jit(cache=True)
def make_functor(stack, symbol_id, arity):
    h = stack.heap_top[0]
    if h + 1 >= stack.heap.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_functor")
    stack.heap[h] = symbol_id
    stack.tags[h] = TAG.FUN
    stack.heap[h + 1] = arity
    stack.tags[h + 1] = TAG.FUN
    stack.heap_top[0] = h + 2
    return h


@jit(cache=True)
def make_structure(stack, functor_pos, args):
    h = stack.heap_top[0]
    need = 1 + len(args)
    if h + need - 1 >= stack.heap.shape[0]:
        raise IndexError("HEAP_OVERFLOW make_structure")
    stack.heap[h] = functor_pos
    stack.tags[h] = TAG.STR
    stack.heap_top[0] = h + 1
    for arg in args:
        stack.heap[stack.heap_top[0]] = arg
        stack.tags[stack.heap_top[0]] = TAG.REF
        stack.heap_top[0] += 1
    return h
