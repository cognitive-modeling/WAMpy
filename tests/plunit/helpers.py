"""Shared helpers for high-level and low-level query tests."""

from wampy.api import (
    compile_program,
    compile_query,
    init_compiled_query,
)
from wampy.api.prolog import Prolog
from wampy.compiler.compiler_state import undo_hypothesis
from wampy.compiler.diagnostics import display_compiler_state
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.parser import parse
from wampy.frontend.solution import decode_bindings
from wampy.runtime.interpreter import redo, run
from wampy.runtime.machine import init_machine
from wampy.status import WAMStatus


def _query_source(source: str) -> str:
    stripped = source.strip()
    if stripped.startswith("?-"):
        return stripped if stripped.endswith(".") else stripped + "."
    if stripped.endswith("."):
        stripped = stripped[:-1].rstrip()
    return f"?- {stripped}."


def solve_from_str(
    program_src: str,
    query_src: str,
    config=DEFAULT_CONFIG,
    *,
    is_debug_compiler: bool = False,
) -> list[dict[str, object]]:
    """Execute a source query through the public ``Prolog`` API."""

    prolog = Prolog(program_src, config=config)
    if is_debug_compiler:
        display_compiler_state(
            prolog.compiled_program,
            prolog.compiler_state,
            prolog.symbol_table,
        )
    return prolog.query_all(query_src)


def solve_from_str_ontop(
    dynamic_src: str,
    query_src: str,
    static_compiled_program,
    static_state,
    symbol_table,
    config=DEFAULT_CONFIG,
    *,
    is_debug_compiler: bool = False,
) -> list[dict[str, object]]:
    """Compile a dynamic extension and execute a query with direct WAM decoding."""

    dynamic_program, symbol_table = parse(dynamic_src, symbol_table, config)

    if static_state.hypothesis_predicate_entry_undo_count[0] > 0:
        undo_hypothesis(static_state, static_compiled_program)

    compile_program(
        dynamic_program,
        compiled_program=static_compiled_program,
        compiler_state=static_state,
        config=config,
    )

    if is_debug_compiler:
        display_compiler_state(static_compiled_program, static_state, symbol_table)

    query_program, symbol_table = parse(
        _query_source(query_src),
        symbol_table,
        config,
    )

    compiled_query = init_compiled_query(config)
    compile_query(
        query_program,
        compiled_query,
        static_compiled_program,
        static_state,
        config,
    )

    machine = init_machine(config.runtime)
    status = WAMStatus(int(run(machine, static_compiled_program, compiled_query)))
    solutions: list[dict[str, object]] = []
    while status == WAMStatus.SUCCESS:
        solutions.append(
            decode_bindings(
                machine,
                compiled_query,
                symbol_table,
            )
        )
        status = WAMStatus(int(redo(machine, static_compiled_program, compiled_query)))

    if status != WAMStatus.EXHAUSTED:
        raise RuntimeError(f"WAM error: {status.name}")
    return solutions


__all__ = [
    "solve_from_str",
    "solve_from_str_ontop",
]
