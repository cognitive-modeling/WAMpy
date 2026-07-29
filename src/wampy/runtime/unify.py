"""
Unification algorithm.

Implements first-order term unification. Given two terms and an existing
substitution, attempts to produce a consistent substitution that makes the
terms equal. The implementation is deterministic and has no knowledge of
backtracking or search.

This module is intended to be used by the solver.
"""

from numba import jit

from wampy.status import WAMStatus
from wampy.runtime.stack import TAG, deref, trail_var


@jit(cache=True)
def unify(stack, t1, t2):
    stack.u_top[0] = 0
    _u_push(stack, t1, t2)

    max_steps = stack.unify_step_limit
    steps = 0

    while stack.u_top[0] > 0:
        if steps >= max_steps:
            return WAMStatus.UNIFY_STEP_LIMIT
        steps += 1

        t1, t2 = _u_pop(stack)

        t1 = deref(stack, t1)
        t2 = deref(stack, t2)

        if t1 >= stack.heap_top[0] or t2 >= stack.heap_top[0]:
            return WAMStatus.CORRUPT_ENVIRONMENT

        if t1 == t2:
            continue

        tag1 = stack.tags[t1]
        tag2 = stack.tags[t2]

        if tag1 == TAG.REF:
            if not bind_var(stack, t1, t2):
                return WAMStatus.EXHAUSTED
            continue

        if tag2 == TAG.REF:
            if not bind_var(stack, t2, t1):
                return WAMStatus.EXHAUSTED
            continue

        if tag1 == TAG.CON and tag2 == TAG.CON:
            if stack.heap[t1] != stack.heap[t2]:
                return WAMStatus.EXHAUSTED
            continue

        if tag1 == TAG.STR and tag2 == TAG.STR:
            f1 = stack.heap[t1]
            f2 = stack.heap[t2]

            if stack.heap[f1] != stack.heap[f2] or stack.heap[f1 + 1] != stack.heap[f2 + 1]:
                return WAMStatus.EXHAUSTED

            arity = stack.heap[f1 + 1]
            for i in range(arity):
                _u_push(stack, t1 + 1 + i, t2 + 1 + i)
            continue

        return WAMStatus.EXHAUSTED

    return WAMStatus.SUCCESS


@jit(cache=True)
def unify_const(stack, x, c) -> bool:
    x = deref(stack, x)

    # invalid / not yet allocated address => fail
    if x >= stack.heap_top[0]:
        return False

    # bind unbound var to constant (in-place, no new heap cell)
    if stack.tags[x] == TAG.REF and stack.heap[x] == x:
        trail_var(stack, x)
        stack.heap[x] = c
        stack.tags[x] = TAG.CON
        return True

    # otherwise must already be same constant
    return (stack.tags[x] == TAG.CON) and (stack.heap[x] == c)


@jit(cache=True)
def bind_var(stack, var, term):
    var = deref(stack, var)
    term = deref(stack, term)

    if var == term:
        return True

    if var >= stack.heap_top[0]:
        return False
    if term >= stack.heap_top[0]:
        return False
    if stack.tags[var] != TAG.REF:
        return False

    if term < stack.heap_top[0]:
        if stack.tags[term] == TAG.REF and stack.heap[term] == term:
            if var < term:
                tmp = var
                var = term
                term = tmp

    if stack.cp_top[0] > 0:
        cp_index = stack.cp_top[0] - 1
        if var < stack.choice_heap_tops[cp_index]:
            trail_var(stack, var)

    stack.heap[var] = term
    stack.tags[var] = TAG.REF
    return True


@jit(cache=True)
def push_struct(m):
    t = m.st_top[0]
    m.S_stack[t] = m.S[0]
    m.mode_stack[t] = m.mode[0]
    m.st_top[0] = t + 1


@jit(cache=True)
def pop_struct(m):
    t = m.st_top[0] - 1
    m.st_top[0] = t
    m.S[0] = m.S_stack[t]
    m.mode[0] = m.mode_stack[t]


@jit(cache=True)
def _u_push(stack, a, b):
    top = stack.u_top[0]
    cap_pairs = stack.u_stack.shape[0] // 2
    if top >= cap_pairs:
        raise IndexError("UNIFY_STACK_OVERFLOW")
    stack.u_stack[2 * top] = a
    stack.u_stack[2 * top + 1] = b
    stack.u_top[0] = top + 1


@jit(cache=True)
def _u_pop(stack):
    stack.u_top[0] -= 1
    top = stack.u_top[0]
    a = stack.u_stack[2 * top]
    b = stack.u_stack[2 * top + 1]
    return a, b
