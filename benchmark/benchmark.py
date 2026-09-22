import json
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from numba import jit
from wampy import WAMStatus, load_config_toml, parse
from wampy.api import (
    clear_compiled_program,
    compile_program,
    compile_query,
    init_compiled_program,
    init_compiled_query,
    init_compiler_state,
    init_symbol_table,
)
from wampy.api.jitable import (
    clear_compiler_state,
    init_machine,
    reset_machine,
    run,
    undo_hypothesis,
)

try:
    janus: Any = import_module("janus_swi")
except ImportError:
    janus = None


SAMPLE_COUNT = 10
ITERATION_COUNTS = [1, 10, 100, 1_000, 10_000]
RUN_JANUS = True
RESULTS_PATH = Path(__file__).with_name("benchmark_results.json")

BENCHMARK_CONFIG = load_config_toml(
    """
    """
)

FAMILY_BASE = r"""
male(anakin).
female(padme).
male(luke).
female(leia).
male(han).
female(mara).
male(ben_solo).
male(ben_skywalker).
female(jaina).
male(jacen).
female(arya).
male(torin).
female(nira).

parent(anakin, luke).
parent(padme, luke).
parent(anakin, leia).
parent(padme, leia).
parent(han, ben_solo).
parent(leia, ben_solo).
parent(han, jaina).
parent(leia, jaina).
parent(han, jacen).
parent(leia, jacen).
parent(luke, ben_skywalker).
parent(mara, ben_skywalker).
parent(luke, arya).

mother(X, Y) :- parent(X, Y), female(X).
child(X, Y) :- parent(Y, X).
grandparent(X, Y) :- parent(X, Z), parent(Z, Y).
grandmother(X, Y) :- grandparent(X, Y), female(X).
"""

FAMILY_EXTENSION = r"""
father(X, Y) :- parent(X, Y), male(X).
grandfather(X, Y) :- grandparent(X, Y), male(X).
parents(F, M, C) :- father(F, C), mother(M, C).
"""


# Add combinations here. `extension=""` gives a static-only benchmark.
BENCHMARKS = [
    {
        "name": "family/father",
        "base": FAMILY_BASE,
        "extension": FAMILY_EXTENSION,
        "query": "father(X, luke).",
    },
    # {
    #     "name": "family/grandfather",
    #     "base": FAMILY_BASE,
    #     "extension": FAMILY_EXTENSION,
    #     "query": "grandfather(X, ben_skywalker).",
    # },
    # {
    #     "name": "family/grandparent-static",
    #     "base": FAMILY_BASE,
    #     "extension": "",
    #     "query": "grandparent(anakin, X).",
    # },
]


@jit
def compile_n_times(
    program,
    compiled_program,
    compiler_state,
    config,
    n_iterations,
    from_scratch,
):
    """Compile repeatedly with explicit artifact lifecycle preparation.

    From-scratch iterations clear both compiler state and compiled-program
    metadata. Dynamic iterations keep the base compiled outside the timed
    region, compile the first hypothesis directly, undo each later installed
    hypothesis before replacement, and leave the final hypothesis installed.
    """

    for iteration in range(n_iterations):
        if from_scratch:
            clear_compiler_state(compiler_state)
            compiler_state.pc_onto[0] = 0
            clear_compiled_program(compiled_program)
        elif iteration > 0:
            undo_hypothesis(compiler_state, compiled_program)

        compile_program(program, compiled_program, compiler_state, config)


@jit
def query_n_times(
    machine,
    compiled_program,
    compiled_query,
    n_iterations,
):
    status = -1

    for _ in range(n_iterations):
        reset_machine(machine)
        status = run(machine, compiled_program, compiled_query)

    return status


def measure_wampy(
    program,
    query_program,
    n_iterations,
    *,
    base_program=None,
):
    """Measure compile + precompiled-query execution.

    If `base_program` is supplied, it is compiled before timing and `program`
    is measured as a hypothesis. Otherwise `program` is compiled from scratch.
    """

    config = BENCHMARK_CONFIG

    compiler_state = init_compiler_state(config)
    compiled_program = init_compiled_program(config)

    dynamic = base_program is not None

    if dynamic:
        compile_program(base_program, compiled_program, compiler_state, config)

    t1 = perf_counter_ns()

    compile_n_times(
        program,
        compiled_program,
        compiler_state,
        config,
        n_iterations,
        not dynamic,
    )
    t2 = perf_counter_ns()

    compiled_query = init_compiled_query(config)
    compile_query(query_program, compiled_query, compiled_program, compiler_state, config)
    machine = init_machine(config.runtime)
    t3 = perf_counter_ns()

    status = query_n_times(machine, compiled_program, compiled_query, n_iterations)
    t4 = perf_counter_ns()

    status = WAMStatus(int(status))

    if status != WAMStatus.SUCCESS:
        raise RuntimeError(f"WAM query failed with {status.name}")

    return (
        (t2 - t1) / n_iterations / 1_000,
        (t4 - t3) / n_iterations / 1_000,
    )


def measure_janus(
    source,
    query,
    n_iterations,
):
    if janus is None:
        raise RuntimeError("janus_swi is not installed")

    compile_ns = 0
    query_ns = 0

    query = query.removesuffix(".")
    source = ":- style_check(-discontiguous).\n" + source

    for _ in range(n_iterations):
        t1 = perf_counter_ns()

        janus.consult(
            "benchmark",
            source,
        )
        t2 = perf_counter_ns()

        janus.query_once(query)
        t3 = perf_counter_ns()

        compile_ns += t2 - t1
        query_ns += t3 - t2

    return (
        compile_ns / n_iterations / 1_000,
        query_ns / n_iterations / 1_000,
    )


def main():
    if RUN_JANUS and janus is None:
        raise RuntimeError(
            "janus_swi is not installed; install the Prolog extras or set RUN_JANUS=False."
        )

    results = []

    for benchmark in BENCHMARKS:
        name = benchmark["name"]
        base_src = benchmark["base"]
        extension_src = benchmark.get("extension", "")
        query_src = benchmark["query"]

        full_src = base_src + "\n" + extension_src
        has_extension = bool(extension_src.strip())

        # Establish one symbol namespace, then parse every component against it.
        symbol_table = init_symbol_table(BENCHMARK_CONFIG.frontend.ast)
        full_program, symbol_table = parse(full_src, symbol_table, BENCHMARK_CONFIG)
        base_program, symbol_table = parse(base_src, symbol_table, BENCHMARK_CONFIG)

        extension_program = None
        if has_extension:
            extension_program, symbol_table = parse(extension_src, symbol_table, BENCHMARK_CONFIG)

        query_program, symbol_table = parse(query_src, symbol_table, BENCHMARK_CONFIG)

        print(f"\n=== {name} ===")
        print(f"query: {query_src}")

        # Warm JIT/Janus once, but do not record it.
        measure_wampy(full_program, query_program, 1)

        if has_extension:
            measure_wampy(extension_program, query_program, 1, base_program=base_program)

        if RUN_JANUS:
            measure_janus(full_src, query_src, 1)

        for sample in range(SAMPLE_COUNT):
            for n_iterations in ITERATION_COUNTS:
                static_compile_us, static_query_us = measure_wampy(
                    full_program,
                    query_program,
                    n_iterations,
                )

                dynamic_compile_us = None
                dynamic_query_us = None

                if has_extension:
                    dynamic_compile_us, dynamic_query_us = measure_wampy(
                        extension_program,
                        query_program,
                        n_iterations,
                        base_program=base_program,
                    )

                janus_compile_us = None
                janus_query_us = None

                if RUN_JANUS:
                    janus_compile_us, janus_query_us = measure_janus(
                        full_src,
                        query_src,
                        n_iterations,
                    )

                print(
                    f"sample={sample}\t"
                    f"n={n_iterations:<5}\t|\t"
                    f"WAMpy full:\t{static_compile_us:8.2f}/{static_query_us:6.2f} us\t|\t"
                    + (
                        f"extension:\t{dynamic_compile_us:8.2f}/{dynamic_query_us:6.2f} us\t|\t"
                        if has_extension
                        else ""
                    )
                    + (
                        f"Janus:\t{janus_compile_us:8.2f}/{janus_query_us:6.2f} us"
                        if RUN_JANUS
                        else ""
                    )
                )

                results.append(
                    {
                        "benchmark": name,
                        "query": query_src,
                        "sample": sample,
                        "n_iterations": n_iterations,
                        "wampy_static_compile_avg_us": static_compile_us,
                        "wampy_static_query_avg_us": static_query_us,
                        "wampy_dynamic_compile_avg_us": dynamic_compile_us,
                        "wampy_dynamic_query_avg_us": dynamic_query_us,
                        "janus_compile_avg_us": janus_compile_us,
                        "janus_query_avg_us": janus_query_us,
                    }
                )

        RESULTS_PATH.write_text(
            json.dumps(results, indent=4),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
