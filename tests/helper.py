from collections.abc import Callable
from typing import Any

from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiled_query import CompiledQuery
from wampy.runtime.interpreter.execution import redo, run
from wampy.runtime.machine.machine import Machine
from wampy.status import WAMStatus


def collect_solutions(
    machine: Machine,
    compiled_program: CompiledProgram,
    compiled_query: CompiledQuery,
    snapshot: Callable[[Machine], Any],
    max_solutions: int = 100,
) -> tuple[list[Any], WAMStatus]:
    solutions = []

    status = run(
        machine,
        compiled_program,
        compiled_query,
    )

    while status == WAMStatus.SUCCESS and len(solutions) < max_solutions:
        solutions.append(snapshot(machine))
        status = redo(
            machine,
            compiled_program,
            compiled_query,
        )

    return solutions, status
