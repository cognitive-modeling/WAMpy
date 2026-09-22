"""Control-flow, environment, cut, and failure instructions."""

from wampy.runtime.interpreter.choice import fail, mark_enclosing_naf_cutoff
from wampy.runtime.machine.environment import (
    ENV_HEADER_SIZE,
    allocate_environment,
    deallocate_environment,
    environment_y_address,
)
from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.machine_registers import HALT_CONTINUATION, NO_CHOICE_POINT
from wampy.runtime.machine.stack import ChoiceSlot, is_choice_point_address
from wampy.status import WAMStatus


def exec_ALLOCATE(
    machine: Machine,
    compiled_program,
    compiled_query,
    entry_pc,
) -> WAMStatus:
    return allocate_environment(
        machine,
        compiled_program,
        compiled_query,
        entry_pc,
    )


def exec_DEALLOCATE(machine: Machine) -> WAMStatus:
    return deallocate_environment(machine)


def exec_CALL(
    machine: Machine,
    predicate_entry,
    predicate_slot,
    arity,
    environment_size,
    ret_pc,
) -> WAMStatus:
    """Install the continuation and jump; X registers remain caller-saved."""

    state = machine.registers
    if machine.depth_limit[0] > 0 and machine.depth[0] >= machine.depth_limit[0]:
        status = mark_enclosing_naf_cutoff(machine)
        if status != WAMStatus.SUCCESS:
            return status
        return WAMStatus.DEPTH_LIMIT

    target = predicate_entry[predicate_slot, arity]
    if target < 0:
        return fail(machine)

    state.B0[0] = state.B[0]

    # The continuation carries the caller's environment size for the next
    # allocation refactor; allocation still consumes its existing operand.
    _ = environment_size
    state.CP[0] = ret_pc
    machine.return_depth[0] = machine.depth[0]
    machine.depth[0] += 1
    state.P[0] = target
    return WAMStatus.SUCCESS


def exec_EXECUTE(
    machine: Machine,
    predicate_entry,
    predicate_slot,
    arity,
) -> WAMStatus:
    state = machine.registers
    if machine.depth_limit[0] > 0 and machine.depth[0] >= machine.depth_limit[0]:
        status = mark_enclosing_naf_cutoff(machine)
        if status != WAMStatus.SUCCESS:
            return status
        return WAMStatus.DEPTH_LIMIT

    state.B0[0] = state.B[0]
    target = predicate_entry[predicate_slot, arity]
    if target < 0:
        return fail(machine)

    machine.depth[0] += 1
    state.P[0] = target
    return WAMStatus.SUCCESS


def exec_GET_LEVEL(machine: Machine, y: int, save_b0: int = 0) -> WAMStatus:
    """Save the current choice level, or the clause entry level when flagged."""

    state = machine.registers
    stack = machine.stack
    if state.E[0] < 0:
        return WAMStatus.CORRUPT_ENVIRONMENT

    address = environment_y_address(state, y)
    if address < state.E[0] + ENV_HEADER_SIZE or address >= stack.cells.shape[0]:
        return WAMStatus.CORRUPT_ENVIRONMENT

    if address >= state.local_top[0]:
        state.local_top[0] = address + 1
    stack.cells[address] = state.B0[0] if save_b0 else state.B[0]
    return WAMStatus.SUCCESS


def _update_hb_after_cut(machine: Machine, boundary: int) -> None:
    state = machine.registers
    if boundary == NO_CHOICE_POINT:
        state.HB[0] = 0
    else:
        state.HB[0] = machine.stack.cells[boundary + ChoiceSlot.H]


def exec_CUT(machine: Machine, y: int) -> WAMStatus:
    state = machine.registers
    stack = machine.stack
    if state.E[0] < 0:
        return WAMStatus.CORRUPT_ENVIRONMENT

    address = environment_y_address(state, y)
    if address < state.E[0] + ENV_HEADER_SIZE or address >= stack.cells.shape[0]:
        return WAMStatus.CORRUPT_ENVIRONMENT

    boundary = stack.cells[address]
    if boundary != NO_CHOICE_POINT and not is_choice_point_address(stack, boundary):
        return WAMStatus.CORRUPT_CHOICEPOINT
    if boundary > state.B[0]:
        return WAMStatus.CORRUPT_CHOICEPOINT

    state.B[0] = boundary
    _update_hb_after_cut(machine, boundary)
    return WAMStatus.SUCCESS


def exec_NECK_CUT(machine: Machine) -> WAMStatus:
    state = machine.registers
    boundary = state.B0[0]
    if boundary != NO_CHOICE_POINT and not is_choice_point_address(machine.stack, boundary):
        return WAMStatus.CORRUPT_CHOICEPOINT
    if boundary > state.B[0]:
        return WAMStatus.CORRUPT_CHOICEPOINT

    state.B[0] = boundary
    _update_hb_after_cut(machine, boundary)
    return WAMStatus.SUCCESS


def exec_FAIL(machine: Machine) -> WAMStatus:
    return fail(machine)


def exec_PROCEED(machine: Machine) -> bool:
    """Jump to CP, without restoring X, or report a top-level solution."""

    state = machine.registers

    machine.depth[0] = machine.return_depth[0]
    state.P[0] = state.CP[0]
    if state.CP[0] == HALT_CONTINUATION:
        return True

    if state.trace_enabled[0] == 1 and state.path_depth[0] > 0:
        state.path_depth[0] -= 1

    return False
