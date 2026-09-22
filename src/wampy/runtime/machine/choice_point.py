"""Choice-point snapshot and restoration primitives."""

from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.machine_registers import NO_CHOICE_POINT, UnificationMode
from wampy.runtime.machine.stack import (
    ChoicePointKind,
    ChoiceSlot,
    is_choice_point_address,
)
from wampy.runtime.machine.trail import unwind_trail
from wampy.status import WAMStatus


def copy_A_to_choice(cells, address, A) -> None:
    """Copy argument registers into choice-point storage."""

    for register in range(A.shape[0]):
        cells[address + ChoiceSlot.A + register] = A[register]


def copy_choice_to_A(A, cells, address) -> None:
    """Restore argument registers from choice-point storage."""

    for register in range(A.shape[0]):
        A[register] = cells[address + ChoiceSlot.A + register]


def save_choice(
    machine: Machine,
    bp,
    kind=ChoicePointKind.NORMAL,
) -> WAMStatus:
    """Push a choice point whose next alternative is ``bp``."""

    state = machine.registers
    memory = machine
    stack = memory.stack
    address = state.local_top[0]
    if address + stack.choice_point_frame_size > stack.cells.shape[0]:
        return WAMStatus.STACK_OVERFLOW

    cells = stack.cells
    cells[address + ChoiceSlot.TR] = state.TR[0]
    cells[address + ChoiceSlot.BP] = bp
    cells[address + ChoiceSlot.CP] = state.CP[0]
    cells[address + ChoiceSlot.CE] = state.E[0]
    cells[address + ChoiceSlot.PREVIOUS_B] = state.B[0]
    cells[address + ChoiceSlot.H] = state.H[0]
    cells[address + ChoiceSlot.B0] = state.B0[0]
    cells[address + ChoiceSlot.KIND] = kind
    copy_A_to_choice(cells, address, memory.X)
    x_end = address + ChoiceSlot.A + memory.X.shape[0]
    cells[x_end] = state.path_depth[0]
    cells[address + ChoiceSlot.DEPTH] = memory.depth[0]
    cells[address + ChoiceSlot.RETURN_DEPTH] = memory.return_depth[0]

    state.B[0] = address
    state.local_top[0] = address + stack.choice_point_frame_size
    state.HB[0] = state.H[0]
    return WAMStatus.SUCCESS


def restore_choice(machine: Machine, address) -> None:
    """Restore registers and memory to a choice point address."""

    state = machine.registers
    memory = machine
    stack = memory.stack
    cells = stack.cells
    copy_choice_to_A(memory.X, cells, address)
    unwind_trail(state, memory, cells[address + ChoiceSlot.TR])

    state.H[0] = cells[address + ChoiceSlot.H]
    state.HB[0] = cells[address + ChoiceSlot.H]
    state.E[0] = cells[address + ChoiceSlot.CE]
    state.B0[0] = cells[address + ChoiceSlot.B0]
    x_end = address + ChoiceSlot.A + memory.X.shape[0]
    state.CP[0] = cells[address + ChoiceSlot.CP]
    state.local_top[0] = address + stack.choice_point_frame_size
    memory.depth[0] = cells[address + ChoiceSlot.DEPTH]
    memory.return_depth[0] = cells[address + ChoiceSlot.RETURN_DEPTH]

    if state.trace_enabled[0] == 1:
        state.path_depth[0] = cells[x_end]

    state.mode[0] = UnificationMode.READ
    state.S[0] = 0
    state.structure_top[0] = 0


def discard_choice(machine: Machine) -> None:
    """Remove the newest choice point and update the heap boundary."""

    state = machine.registers
    memory = machine
    stack = memory.stack
    cells = stack.cells
    address = state.B[0]
    if not is_choice_point_address(stack, address):
        return

    previous = cells[address + ChoiceSlot.PREVIOUS_B]
    if previous != NO_CHOICE_POINT:
        state.B[0] = previous
        state.HB[0] = cells[previous + ChoiceSlot.H]
    else:
        state.B[0] = NO_CHOICE_POINT
        state.HB[0] = 0
    state.local_top[0] = address
