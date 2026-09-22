from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiled_query import CompiledQuery
from wampy.runtime.interpreter.choice import fail
from wampy.runtime.interpreter.loop import emulate
from wampy.runtime.machine.machine import Machine
from wampy.runtime.machine.machine_registers import HALT_CONTINUATION
from wampy.status import WAMStatus


def run(
    machine: Machine,
    compiled_program: CompiledProgram,
    compiled_query: CompiledQuery,
) -> WAMStatus:
    """Start a new query execution."""

    state = machine.registers
    state.B0[0] = state.B[0]
    state.P[0] = compiled_program.code_size[0]
    state.CP[0] = HALT_CONTINUATION

    return emulate(
        machine,
        compiled_program,
        compiled_query,
    )


def redo(
    machine: Machine,
    compiled_program: CompiledProgram,
    compiled_query: CompiledQuery,
) -> WAMStatus:
    """Backtrack once and continue execution."""

    status = fail(machine)

    if status != WAMStatus.SUCCESS:
        return status

    return emulate(
        machine,
        compiled_program,
        compiled_query,
    )
