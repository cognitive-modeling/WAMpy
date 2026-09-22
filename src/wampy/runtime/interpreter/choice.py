"""Choice instructions, failure, and backtracking."""

from wampy.runtime.machine.choice_point import (
    discard_choice,
    restore_choice,
    save_choice,
)
from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.stack import (
    ChoicePointKind,
    ChoiceSlot,
    is_choice_point_address,
)
from wampy.status import WAMStatus


def mark_enclosing_naf_cutoff(machine: Machine) -> WAMStatus:
    """Record that the nearest active NAF computation was truncated."""

    state = machine.registers
    stack = machine.stack
    address = state.B[0]

    while address > state.fail_base[0]:
        if not is_choice_point_address(stack, address):
            return WAMStatus.CORRUPT_CHOICEPOINT
        kind = stack.cells[address + ChoiceSlot.KIND]
        if kind == ChoicePointKind.NAF:
            stack.cells[address + ChoiceSlot.KIND] = ChoicePointKind.NAF_TRUNCATED
            break
        if kind == ChoicePointKind.NAF_TRUNCATED:
            break
        address = stack.cells[address + ChoiceSlot.PREVIOUS_B]

    return WAMStatus.SUCCESS


def fail(machine: Machine) -> WAMStatus:
    """Backtrack to the newest choice point and select its alternative."""

    state = machine.registers
    stack = machine.stack
    if state.B[0] <= state.fail_base[0] or not is_choice_point_address(stack, state.B[0]):
        return WAMStatus.EXHAUSTED

    address = state.B[0]
    if stack.cells[address + ChoiceSlot.KIND] == ChoicePointKind.NAF_TRUNCATED:
        restore_choice(machine, address)
        discard_choice(machine)
        status = mark_enclosing_naf_cutoff(machine)
        if status != WAMStatus.SUCCESS:
            return status
        return WAMStatus.DEPTH_LIMIT

    restore_choice(machine, address)
    state.P[0] = stack.cells[address + ChoiceSlot.BP]
    return WAMStatus.SUCCESS


def exec_TRY_ME_ELSE(
    machine: Machine,
    bp,
    kind=ChoicePointKind.NORMAL,
) -> WAMStatus:
    return save_choice(machine, bp, kind)


def exec_RETRY_ME_ELSE(machine: Machine, bp) -> WAMStatus:
    state = machine.registers
    memory = machine
    stack = memory.stack
    address = state.B[0]
    if not is_choice_point_address(stack, address):
        return WAMStatus.CORRUPT_CHOICEPOINT

    restore_choice(machine, address)
    stack.cells[address + ChoiceSlot.BP] = bp
    return WAMStatus.SUCCESS


def exec_TRUST_ME_ELSE_FAIL(machine: Machine) -> WAMStatus:
    state = machine.registers
    address = state.B[0]
    if not is_choice_point_address(machine.stack, address):
        return WAMStatus.CORRUPT_CHOICEPOINT

    restore_choice(machine, address)
    discard_choice(machine)
    return WAMStatus.SUCCESS


def exec_JMP_RETRY(machine: Machine, clause_pc, bp) -> WAMStatus:
    """Retry a choice and jump to an explicit clause address."""

    state = machine.registers
    memory = machine
    stack = memory.stack
    address = state.B[0]
    if not is_choice_point_address(stack, address):
        return WAMStatus.CORRUPT_CHOICEPOINT

    restore_choice(machine, address)
    stack.cells[address + ChoiceSlot.BP] = bp
    state.P[0] = clause_pc
    return WAMStatus.SUCCESS


def exec_JMP_TRUST(machine: Machine, clause_pc) -> WAMStatus:
    """Trust the final choice and jump to an explicit clause address."""

    state = machine.registers
    address = state.B[0]
    if not is_choice_point_address(machine.stack, address):
        return WAMStatus.CORRUPT_CHOICEPOINT

    restore_choice(machine, address)
    discard_choice(machine)
    state.P[0] = clause_pc
    return WAMStatus.SUCCESS
