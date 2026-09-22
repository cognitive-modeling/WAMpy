"""Environment-frame addressing and lifecycle operations."""

from enum import IntEnum, unique

from wampy.compiler.opcodes import OP
from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.machine_registers import HALT_CONTINUATION
from wampy.status import WAMStatus


@unique
class EnvSlot(IntEnum):
    CE = 0
    CP = 1
    RETURN_DEPTH = 2


ENV_HEADER_SIZE = len(EnvSlot)


def environment_y_address(
    state,
    y,
):
    """Return the flat environment-stack address for ``Yy``."""

    return state.E[0] + ENV_HEADER_SIZE + y


def environment_size_at_continuation(
    compiled_program,
    compiled_query,
    cp: int,
) -> int:
    """Return the environment size encoded by the ``CALL`` before ``cp``.

    ``cp`` uses the interpreter's virtual address space: program instructions
    occupy the prefix and query instructions follow at ``program.code_size[0]``.
    ``-1`` indicates that ``cp`` does not identify a valid ``CALL``
    continuation.
    """

    call_pc = cp - 1
    if call_pc < 0:
        return -1

    program_size = compiled_program.code_size[0]
    if call_pc < program_size:
        instruction = compiled_program.code[call_pc]
    else:
        query_pc = call_pc - program_size
        if query_pc < 0 or query_pc >= compiled_query.code_size[0]:
            return -1
        instruction = compiled_query.code[query_pc]

    if instruction[0] != OP.CALL or instruction[3] < 0:
        return -1

    return int(instruction[3])


def environment_size_at_entry(
    compiled_program,
    compiled_query,
    entry_pc: int,
) -> int:
    """Return the frame size recoverable from a clause or query entry."""

    program_size = compiled_program.code_size[0]
    if entry_pc < 0:
        return -1

    if entry_pc < program_size:
        code = compiled_program.code
        code_size = program_size
        local_pc = entry_pc
    else:
        code = compiled_query.code
        code_size = compiled_query.code_size[0]
        local_pc = entry_pc - program_size
        if local_pc >= code_size:
            return -1

    saw_allocate = False
    max_y = -1
    for pc in range(local_pc, code_size):
        instruction = code[pc]
        if instruction[0] == OP.ALLOCATE:
            saw_allocate = True
        if (
            instruction[0] == OP.GET_Y_VAR
            or instruction[0] == OP.GET_Y_VAL
            or instruction[0] == OP.PUT_Y_VAR
            or instruction[0] == OP.PUT_Y_VAL
            or instruction[0] == OP.SET_Y_VAR
            or instruction[0] == OP.SET_Y_VAL
            or instruction[0] == OP.UNIFY_Y_VAR
            or instruction[0] == OP.UNIFY_Y_VAL
            or instruction[0] == OP.GET_LEVEL
            or instruction[0] == OP.CUT
        ) and instruction[1] > max_y:
            max_y = instruction[1]
        if instruction[0] == OP.PROCEED or instruction[0] == OP.EXECUTE:
            break

    if max_y >= 0:
        return int(max_y) + 1
    if saw_allocate:
        return 0
    return -1


def allocate_environment(
    machine: Machine,
    compiled_program,
    compiled_query,
    entry_pc: int,
) -> WAMStatus:
    """Push an environment sized by its continuation and clause entry."""

    state = machine.registers
    stack_memory = machine.stack
    stack = stack_memory.cells

    num_permanent = environment_size_at_continuation(
        compiled_program,
        compiled_query,
        state.CP[0],
    )
    if num_permanent < 0:
        if state.CP[0] != HALT_CONTINUATION:
            return WAMStatus.CORRUPT_ENVIRONMENT
        num_permanent = environment_size_at_entry(
            compiled_program,
            compiled_query,
            entry_pc,
        )
    else:
        # A CALL continuation can carry fewer slots than this clause needs
        # (for example, when a later clause has Y variables).  Recover the
        # entry's actual requirement as a lower bound so a small continuation
        # cannot cause a choice point to overwrite the environment.
        entry_size = environment_size_at_entry(
            compiled_program,
            compiled_query,
            entry_pc,
        )
        if entry_size > num_permanent:
            num_permanent = entry_size

    if num_permanent < 0:
        return WAMStatus.CORRUPT_ENVIRONMENT

    frame = state.local_top[0]
    new_top = frame + ENV_HEADER_SIZE + num_permanent

    if new_top > stack.shape[0]:
        return WAMStatus.ENVIRONMENT_OVERFLOW

    stack[frame + EnvSlot.CE] = state.E[0]
    stack[frame + EnvSlot.CP] = state.CP[0]
    stack[frame + EnvSlot.RETURN_DEPTH] = machine.return_depth[0]

    for y in range(num_permanent):
        stack[frame + ENV_HEADER_SIZE + y] = 0

    state.E[0] = frame
    state.local_top[0] = new_top

    return WAMStatus.SUCCESS


def deallocate_environment(
    machine: Machine,
) -> WAMStatus:
    """Pop the current environment and restore its saved continuation."""

    state = machine.registers
    stack_memory = machine.stack
    stack = stack_memory.cells

    frame = state.E[0]

    if frame < 0 or frame + EnvSlot.RETURN_DEPTH >= stack.shape[0]:
        return WAMStatus.CORRUPT_ENVIRONMENT

    ce = stack[frame + EnvSlot.CE]

    if ce >= frame or ce < -1:
        return WAMStatus.CORRUPT_ENVIRONMENT

    cp = stack[frame + EnvSlot.CP]
    return_depth = stack[frame + EnvSlot.RETURN_DEPTH]
    state.CP[0] = cp
    machine.return_depth[0] = return_depth
    state.E[0] = ce
    state.local_top[0] = frame
    if state.B[0] >= 0:
        choice_top = state.B[0] + stack_memory.choice_point_frame_size
        if choice_top > state.local_top[0]:
            state.local_top[0] = choice_top

    return WAMStatus.SUCCESS
