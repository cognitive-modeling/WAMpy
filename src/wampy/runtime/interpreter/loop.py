"""WAM instruction dispatch loop.

"How does the WAM execute instructions?"
"""

from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiled_query import CompiledQuery
from wampy.compiler.opcodes import OP
from wampy.runtime.interpreter.choice import (
    exec_JMP_RETRY,
    exec_JMP_TRUST,
    exec_RETRY_ME_ELSE,
    exec_TRUST_ME_ELSE_FAIL,
    exec_TRY_ME_ELSE,
)
from wampy.runtime.interpreter.control import (
    exec_ALLOCATE,
    exec_CALL,
    exec_CUT,
    exec_DEALLOCATE,
    exec_EXECUTE,
    exec_FAIL,
    exec_GET_LEVEL,
    exec_NECK_CUT,
    exec_PROCEED,
)
from wampy.runtime.interpreter.term import (
    exec_END_STR,
    exec_GET_CONST,
    exec_GET_STR,
    exec_GET_VAL,
    exec_GET_VAR,
    exec_GET_Y_VAL,
    exec_GET_Y_VAR,
    exec_PUT_CONST,
    exec_PUT_STR,
    exec_PUT_VAL,
    exec_PUT_VAR,
    exec_PUT_Y_VAL,
    exec_PUT_Y_VAR,
    exec_SET_CONST,
    exec_SET_VAL,
    exec_SET_VAR,
    exec_SET_Y_VAL,
    exec_SET_Y_VAR,
    exec_UNIFY_CONST,
    exec_UNIFY_VAL,
    exec_UNIFY_VAR,
    exec_UNIFY_Y_VAL,
    exec_UNIFY_Y_VAR,
)
from wampy.runtime.machine import Machine
from wampy.status import WAMStatus


def emulate(
    machine: Machine,
    compiled_program: CompiledProgram,
    compiled_query: CompiledQuery,
) -> WAMStatus:
    """Continue WAM execution until a solution or terminal status is reached."""

    state = machine.registers

    program_code_size = compiled_program.code_size[0]
    query_base = program_code_size
    code_size = query_base + compiled_query.code_size[0]

    while 0 <= state.P[0] < code_size:
        if state.fuel[0] <= 0:
            return WAMStatus.STEP_LIMIT

        state.fuel[0] -= 1

        pc = state.P[0]

        if pc < program_code_size:
            op, a, b, c = compiled_program.code[pc]
        else:
            op, a, b, c = compiled_query.code[pc - query_base]

        if op == OP.GET_VAR:
            state.P[0] = pc + 1
            exec_GET_VAR(machine, a, b)
            continue

        elif op == OP.GET_Y_VAR:
            state.P[0] = pc + 1
            exec_GET_Y_VAR(machine, a, b)
            continue

        elif op == OP.GET_VAL:
            state.P[0] = pc + 1
            status = exec_GET_VAL(machine, a, b)

        elif op == OP.GET_Y_VAL:
            state.P[0] = pc + 1
            status = exec_GET_Y_VAL(machine, a, b)

        elif op == OP.GET_CONST:
            state.P[0] = pc + 1
            status = exec_GET_CONST(machine, a, b)

        elif op == OP.PUT_VAR:
            state.P[0] = pc + 1
            status = exec_PUT_VAR(machine, a, b)

        elif op == OP.PUT_Y_VAR:
            state.P[0] = pc + 1
            status = exec_PUT_Y_VAR(machine, a, b)

        elif op == OP.PUT_VAL:
            state.P[0] = pc + 1
            status = exec_PUT_VAL(machine, a, b)

        elif op == OP.PUT_Y_VAL:
            state.P[0] = pc + 1
            status = exec_PUT_Y_VAL(machine, a, b)

        elif op == OP.PUT_CONST:
            state.P[0] = pc + 1
            status = exec_PUT_CONST(machine, a, b)

        elif op == OP.ALLOCATE:
            state.P[0] = pc + 1
            status = exec_ALLOCATE(
                machine,
                compiled_program,
                compiled_query,
                pc,
            )

        elif op == OP.DEALLOCATE:
            state.P[0] = pc + 1
            status = exec_DEALLOCATE(machine)

        elif op == OP.CALL:
            status = exec_CALL(
                machine,
                compiled_program.predicate_entry,
                a,
                b,
                c,
                pc + 1,
            )

        elif op == OP.EXECUTE:
            status = exec_EXECUTE(
                machine,
                compiled_program.predicate_entry,
                a,
                b,
            )

        elif op == OP.GET_LEVEL:
            state.P[0] = pc + 1
            status = exec_GET_LEVEL(machine, a, b)

        elif op == OP.CUT:
            state.P[0] = pc + 1
            status = exec_CUT(machine, a)

        elif op == OP.NECK_CUT:
            state.P[0] = pc + 1
            status = exec_NECK_CUT(machine)

        elif op == OP.FAIL:
            status = exec_FAIL(machine)

        elif op == OP.PROCEED:
            if exec_PROCEED(machine):
                return WAMStatus.SUCCESS
            continue

        elif op == OP.TRY_ME_ELSE:
            status = exec_TRY_ME_ELSE(machine, a, b)

            if status == WAMStatus.SUCCESS:
                state.P[0] = pc + 1
                continue

        elif op == OP.RETRY_ME_ELSE:
            state.P[0] = pc + 1
            status = exec_RETRY_ME_ELSE(machine, a)

        elif op == OP.TRUST_ME_ELSE_FAIL:
            state.P[0] = pc + 1
            status = exec_TRUST_ME_ELSE_FAIL(machine)

        elif op == OP.JMP_RETRY:
            status = exec_JMP_RETRY(machine, a, b)

        elif op == OP.JMP_TRUST:
            status = exec_JMP_TRUST(machine, a)

        elif op == OP.GET_STR:
            state.P[0] = pc + 1
            status = exec_GET_STR(machine, a, b, c)

        elif op == OP.PUT_STR:
            state.P[0] = pc + 1
            status = exec_PUT_STR(machine, a, b, c)

        elif op == OP.SET_VAR:
            state.P[0] = pc + 1
            status = exec_SET_VAR(machine, a)

        elif op == OP.SET_Y_VAR:
            state.P[0] = pc + 1
            status = exec_SET_Y_VAR(machine, a)

        elif op == OP.SET_VAL:
            state.P[0] = pc + 1
            status = exec_SET_VAL(machine, a)

        elif op == OP.SET_Y_VAL:
            state.P[0] = pc + 1
            status = exec_SET_Y_VAL(machine, a)

        elif op == OP.SET_CONST:
            state.P[0] = pc + 1
            status = exec_SET_CONST(machine, a)

        elif op == OP.UNIFY_VAR:
            state.P[0] = pc + 1
            status = exec_UNIFY_VAR(machine, a)

        elif op == OP.UNIFY_Y_VAR:
            state.P[0] = pc + 1
            status = exec_UNIFY_Y_VAR(machine, a)

        elif op == OP.UNIFY_VAL:
            state.P[0] = pc + 1
            status = exec_UNIFY_VAL(machine, a)

        elif op == OP.UNIFY_Y_VAL:
            state.P[0] = pc + 1
            status = exec_UNIFY_Y_VAL(machine, a)

        elif op == OP.UNIFY_CONST:
            state.P[0] = pc + 1
            status = exec_UNIFY_CONST(machine, a)

        elif op == OP.END_STR:
            state.P[0] = pc + 1
            status = exec_END_STR(machine)

        else:
            return WAMStatus.INVALID_OPCODE

        if status != WAMStatus.SUCCESS:
            return status

    return WAMStatus.INVALID_PC
