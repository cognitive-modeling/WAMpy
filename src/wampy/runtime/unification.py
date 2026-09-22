"""First-order term unification over a split WAM machine."""

from wampy.runtime.machine.heap import deref
from wampy.runtime.machine.stack import ChoiceSlot
from wampy.runtime.machine.trail import trail_var
from wampy.runtime.tags import TAG
from wampy.status import WAMStatus


def unify(state, memory, t1, t2):
    heap = memory.heap
    state.pdl_top[0] = 0
    _u_push(state, memory, t1, t2)

    max_steps = state.unify_step_limit
    steps = 0

    while state.pdl_top[0] > 0:
        if steps >= max_steps:
            return WAMStatus.UNIFY_STEP_LIMIT
        steps += 1

        t1, t2 = _u_pop(state, memory)
        t1 = deref(state, memory, t1)
        t2 = deref(state, memory, t2)

        if t1 >= state.H[0] or t2 >= state.H[0]:
            return WAMStatus.CORRUPT_ENVIRONMENT

        if t1 == t2:
            continue

        tag1 = heap.tags[t1]
        tag2 = heap.tags[t2]

        if tag1 == TAG.REF:
            if not bind_var(state, memory, t1, t2):
                return WAMStatus.EXHAUSTED
            continue

        if tag2 == TAG.REF:
            if not bind_var(state, memory, t2, t1):
                return WAMStatus.EXHAUSTED
            continue

        if tag1 == TAG.CON and tag2 == TAG.CON:
            if heap.cells[t1] != heap.cells[t2]:
                return WAMStatus.EXHAUSTED
            continue

        if tag1 == TAG.STR and tag2 == TAG.STR:
            f1 = heap.cells[t1]
            f2 = heap.cells[t2]
            if heap.cells[f1] != heap.cells[f2] or heap.cells[f1 + 1] != heap.cells[f2 + 1]:
                return WAMStatus.EXHAUSTED

            arity = heap.cells[f1 + 1]
            for index in range(arity):
                _u_push(state, memory, t1 + 1 + index, t2 + 1 + index)
            continue

        return WAMStatus.EXHAUSTED

    return WAMStatus.SUCCESS


def unify_const(state, memory, address, constant) -> bool:
    heap = memory.heap
    address = deref(state, memory, address)

    if address >= state.H[0]:
        return False

    if heap.tags[address] == TAG.REF and heap.cells[address] == address:
        trail_var(state, memory, address)
        heap.cells[address] = constant
        heap.tags[address] = TAG.CON
        return True

    return (heap.tags[address] == TAG.CON) and (heap.cells[address] == constant)


def bind_var(state, memory, var, term):
    heap = memory.heap
    stack = memory.stack
    var = deref(state, memory, var)
    term = deref(state, memory, term)

    if var == term:
        return True
    if var >= state.H[0] or term >= state.H[0] or heap.tags[var] != TAG.REF:
        return False

    if heap.tags[term] == TAG.REF and heap.cells[term] == term and var < term:
        var, term = term, var

    if state.B[0] >= 0:
        heap_boundary = stack.cells[state.B[0] + ChoiceSlot.H]
        if var < heap_boundary:
            trail_var(state, memory, var)

    heap.cells[var] = term
    heap.tags[var] = TAG.REF
    return True


def push_struct(state, memory):
    pdl = memory.pdl
    top = state.structure_top[0]
    pdl.structure_S[top] = state.S[0]
    pdl.structure_modes[top] = state.mode[0]
    state.structure_top[0] = top + 1


def pop_struct(state, memory):
    pdl = memory.pdl
    top = state.structure_top[0] - 1
    state.structure_top[0] = top
    state.S[0] = pdl.structure_S[top]
    state.mode[0] = pdl.structure_modes[top]


def _u_push(state, memory, a, b):
    cells = memory.pdl.cells
    top = state.pdl_top[0]
    capacity_pairs = cells.shape[0] // 2
    if top >= capacity_pairs:
        raise IndexError("PDL_OVERFLOW")
    cells[2 * top] = a
    cells[2 * top + 1] = b
    state.pdl_top[0] = top + 1


def _u_pop(state, memory):
    cells = memory.pdl.cells
    state.pdl_top[0] -= 1
    top = state.pdl_top[0]
    a = cells[2 * top]
    b = cells[2 * top + 1]
    return a, b
