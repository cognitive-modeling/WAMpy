"""Term construction, matching, and structure instructions."""

from wampy.runtime.interpreter.choice import fail
from wampy.runtime.machine.environment import environment_y_address
from wampy.runtime.machine.heap import deref, make_functor, make_var
from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.machine_registers import UnificationMode
from wampy.runtime.tags import TAG
from wampy.runtime.unification import pop_struct, push_struct, unify, unify_const
from wampy.status import WAMStatus


def exec_GET_VAR(
    machine: Machine,
    x,
    i,
) -> WAMStatus:
    memory = machine
    memory.X[x] = memory.X[i]
    return WAMStatus.SUCCESS


def exec_GET_Y_VAR(
    machine: Machine,
    y,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    memory.stack.cells[environment_y_address(state, y)] = memory.X[i]
    return WAMStatus.SUCCESS


def exec_GET_Y_VAL(
    machine: Machine,
    y,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    value = memory.stack.cells[environment_y_address(state, y)]
    status = unify(state, memory, value, memory.X[i])
    if status == WAMStatus.SUCCESS:
        return WAMStatus.SUCCESS
    if status == WAMStatus.EXHAUSTED:
        return fail(machine)
    return status


def exec_GET_VAL(machine: Machine, x, i) -> WAMStatus:
    state = machine.registers
    memory = machine
    status = unify(state, memory, memory.X[x], memory.X[i])
    if status == WAMStatus.SUCCESS:
        return WAMStatus.SUCCESS
    if status == WAMStatus.EXHAUSTED:
        return fail(machine)
    return status


def exec_GET_CONST(
    machine: Machine,
    constant,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    if not unify_const(state, memory, memory.X[i], constant):
        return fail(machine)
    return WAMStatus.SUCCESS


def exec_PUT_CONST(
    machine: Machine,
    constant,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.heap.cells[address] = constant
    memory.heap.tags[address] = TAG.CON
    memory.X[i] = address
    return WAMStatus.SUCCESS


def exec_PUT_VAL(
    machine: Machine,
    x,
    i,
) -> WAMStatus:
    memory = machine
    memory.X[i] = memory.X[x]
    return WAMStatus.SUCCESS


def exec_PUT_VAR(
    machine: Machine,
    x,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.X[x] = address
    memory.X[i] = address
    return WAMStatus.SUCCESS


def exec_PUT_Y_VAR(
    machine: Machine,
    y,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.stack.cells[environment_y_address(state, y)] = address
    memory.X[i] = address
    return WAMStatus.SUCCESS


def exec_PUT_Y_VAL(
    machine: Machine,
    y,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    memory.X[i] = memory.stack.cells[environment_y_address(state, y)]
    return WAMStatus.SUCCESS


def exec_PUT_STR(
    machine: Machine,
    symbol_id,
    arity,
    i,
) -> WAMStatus:
    """put_structure: allocate only the structure representation.

    With WAMpy's split functor representation this is three heap cells:
    two FUN cells followed by the STR cell.  Arguments are appended by the
    following SET_* instructions, one heap cell at a time.
    """

    state = machine.registers
    memory = machine
    heap = memory.heap

    if int(state.H[0]) + 3 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW

    functor = make_functor(state, memory, symbol_id, arity)
    structure = state.H[0]
    heap.cells[structure] = functor
    heap.tags[structure] = TAG.STR
    state.H[0] = structure + 1
    memory.X[i] = structure
    return WAMStatus.SUCCESS


def exec_SET_VAR(
    machine: Machine,
    register,
) -> WAMStatus:
    """set_variable: append a fresh REF cell and expose it in X[register]."""

    state = machine.registers
    memory = machine
    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.X[register] = address
    return WAMStatus.SUCCESS


def exec_SET_Y_VAR(
    machine: Machine,
    y,
) -> WAMStatus:
    """Permanent-variable form of set_variable."""

    state = machine.registers
    memory = machine
    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.stack.cells[environment_y_address(state, y)] = address
    return WAMStatus.SUCCESS


def exec_SET_VAL(
    machine: Machine,
    register,
) -> WAMStatus:
    """set_value: append a reference to X[register]."""

    state = machine.registers
    memory = machine
    heap = memory.heap
    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = memory.X[register]
    heap.tags[address] = TAG.REF
    state.H[0] = address + 1
    return WAMStatus.SUCCESS


def exec_SET_Y_VAL(
    machine: Machine,
    y,
) -> WAMStatus:
    """Permanent-variable form of set_value."""

    state = machine.registers
    memory = machine
    heap = memory.heap
    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = memory.stack.cells[environment_y_address(state, y)]
    heap.tags[address] = TAG.REF
    state.H[0] = address + 1
    return WAMStatus.SUCCESS


def exec_SET_CONST(
    machine: Machine,
    constant,
) -> WAMStatus:
    """set_constant: append one constant cell."""

    state = machine.registers
    memory = machine
    heap = memory.heap
    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = constant
    heap.tags[address] = TAG.CON
    state.H[0] = address + 1
    return WAMStatus.SUCCESS


def exec_GET_STR(
    machine: Machine,
    symbol_id,
    arity,
    i,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    heap = memory.heap
    address = deref(state, memory, memory.X[i])

    if address >= heap.cells.shape[0]:
        return fail(machine)

    if address < state.H[0] and heap.tags[address] == TAG.REF and heap.cells[address] == address:
        # get_structure in WRITE mode allocates only the structure header.
        # The following UNIFY_* instructions append its arguments at H.
        if int(state.H[0]) + 3 > heap.cells.shape[0]:
            return WAMStatus.HEAP_OVERFLOW

        functor = make_functor(state, memory, symbol_id, arity)
        structure = state.H[0]
        heap.cells[structure] = functor
        heap.tags[structure] = TAG.STR
        state.H[0] = structure + 1

        status = unify(state, memory, address, structure)
        if status == WAMStatus.EXHAUSTED:
            return fail(machine)
        if status != WAMStatus.SUCCESS:
            return status

        push_struct(state, memory)
        state.mode[0] = UnificationMode.WRITE
        state.S[0] = state.H[0]
        return WAMStatus.SUCCESS

    address = deref(state, memory, address)
    if address >= state.H[0] or heap.tags[address] != TAG.STR:
        return fail(machine)

    functor = heap.cells[address]
    if heap.cells[functor] != symbol_id or heap.cells[functor + 1] != arity:
        return fail(machine)

    push_struct(state, memory)
    state.mode[0] = UnificationMode.READ
    state.S[0] = address + 1
    return WAMStatus.SUCCESS


def exec_UNIFY_VAR(
    machine: Machine,
    register,
) -> WAMStatus:
    state = machine.registers
    memory = machine

    if state.mode[0] == UnificationMode.READ:
        address = state.S[0]
        memory.X[register] = address
        state.S[0] = address + 1
        return WAMStatus.SUCCESS

    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.X[register] = address
    state.S[0] = state.H[0]
    return WAMStatus.SUCCESS


def exec_UNIFY_Y_VAR(
    machine: Machine,
    y,
) -> WAMStatus:
    state = machine.registers
    memory = machine

    if state.mode[0] == UnificationMode.READ:
        address = state.S[0]
        memory.stack.cells[environment_y_address(state, y)] = address
        state.S[0] = address + 1
        return WAMStatus.SUCCESS

    if int(state.H[0]) + 1 > memory.heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = make_var(state, memory)
    memory.stack.cells[environment_y_address(state, y)] = address
    state.S[0] = state.H[0]
    return WAMStatus.SUCCESS


def exec_UNIFY_VAL(
    machine: Machine,
    register,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    heap = memory.heap

    if state.mode[0] == UnificationMode.READ:
        address = state.S[0]
        status = unify(state, memory, memory.X[register], address)
        if status == WAMStatus.EXHAUSTED:
            return fail(machine)
        if status != WAMStatus.SUCCESS:
            return status
        state.S[0] = address + 1
        return WAMStatus.SUCCESS

    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = memory.X[register]
    heap.tags[address] = TAG.REF
    state.H[0] = address + 1
    state.S[0] = state.H[0]
    return WAMStatus.SUCCESS


def exec_UNIFY_Y_VAL(
    machine: Machine,
    y,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    heap = memory.heap
    value = memory.stack.cells[environment_y_address(state, y)]

    if state.mode[0] == UnificationMode.READ:
        address = state.S[0]
        status = unify(state, memory, value, address)
        if status == WAMStatus.EXHAUSTED:
            return fail(machine)
        if status != WAMStatus.SUCCESS:
            return status
        state.S[0] = address + 1
        return WAMStatus.SUCCESS

    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = value
    heap.tags[address] = TAG.REF
    state.H[0] = address + 1
    state.S[0] = state.H[0]
    return WAMStatus.SUCCESS


def exec_UNIFY_CONST(
    machine: Machine,
    constant,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    heap = memory.heap

    if state.mode[0] == UnificationMode.READ:
        address = state.S[0]
        if not unify_const(state, memory, address, constant):
            return fail(machine)
        state.S[0] = address + 1
        return WAMStatus.SUCCESS

    if int(state.H[0]) + 1 > heap.cells.shape[0]:
        return WAMStatus.HEAP_OVERFLOW
    address = state.H[0]
    heap.cells[address] = constant
    heap.tags[address] = TAG.CON
    state.H[0] = address + 1
    state.S[0] = state.H[0]
    return WAMStatus.SUCCESS


def exec_END_STR(
    machine: Machine,
) -> WAMStatus:
    state = machine.registers
    memory = machine
    pop_struct(state, memory)
    return WAMStatus.SUCCESS
