from dataclasses import dataclass
import json
from pathlib import Path
import timeit
from typing import Callable

from numba import jit

from wampy.compiler import (
    compile,
    compile_program_onto,
    init_compiler,
    reset_compiler,
)
from wampy.config import load_config_toml
from wampy.frontend.parser import parse

from wampy.frontend.parser import build_query
from wampy.frontend.answers import Answers, decode_answer, init_answers
from wampy.runtime.stack import init_stack, reset_stack
from wampy.engine import query
from wampy.runtime.interpreter import init_machine, reset_machine

from wampy.frontend.ast.program import init_program
from wampy.frontend.parser import _split_clauses, add_prolog_term


from time import perf_counter_ns

try:
    import janus_swi as janus
except ImportError:
    janus = None


SAMPLE_COUNT = 10
WARMUP_ITERATIONS = 1
ITERATION_COUNTS = [1, 10, 100, 1_000, 10_000]  ## , 100_000]
RESULTS_PATH = Path(__file__).with_name("benchmark_results.json")

PROGRAM_STATIC_SRC = r"""
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

PROGRAM_DYNAMIC_SRC = r"""
father(X, Y) :- parent(X, Y), male(X).
grandfather(X, Y) :- grandparent(X, Y), male(X).
parents(F, M, C) :- father(F, C), mother(M, C).
"""

JANUS_SRC_PREFIX = """:- style_check(-discontiguous).
"""

WAM_CONFIG_TOML = """
[ast]
num_terms = 60
max_terms_nodes = 16
max_terms_nodes_childs = 3

[stack]
heap_size = 2_048
trail_size = 512
cp_size = 256
unify_stack_size = 256

[entry]
max_functors = 256
max_arity = 4

[solver]
answer_max_answers = 1
answer_max_nodes = 512
"""

bm_query = "father(X, luke)."


def parse_with_name_index(source: str, name_index, config):
    program = init_program(config)

    for clause in _split_clauses(source):
        add_prolog_term(program, clause, name_index)

    return program


@dataclass(frozen=True)
class BenchmarkContext:
    config: object
    whole_program: object
    static_program: object
    dynamic_program: object
    janus_src: str
    name_index: object


@jit(nopython=True)
def compile_static_n_times(program, code_area, config, n_iterations):
    for _ in range(n_iterations):
        code_area = reset_compiler(code_area, config)
        code_area = compile(program, code_area, config)


@jit(nopython=True)
def query_once_n_times(code_area, machine, goals, stack, answers, config, n_iterations):
    nsol = None
    for _ in range(n_iterations):
        reset_machine(machine, stack)
        reset_stack(stack)
        nsol, st = query(code_area, machine, goals, stack, answers, config)
    return nsol


@jit(nopython=True)
def compile_dynamic_n_times(static_state, static_program, dynamic_program, config, n_iterations):
    static_state = compile(static_program, static_state, config)
    static_entry = static_state.entry.copy()
    static_size = static_state.pc.value

    for _ in range(n_iterations):
        static_state = compile_program_onto(static_state, static_entry, static_size, dynamic_program, config)


def run_static_benchmark(context, n_iterations):
    code_area = init_compiler(context.config)
    compiled_program = compile(context.whole_program, code_area, context.config)
    q = build_query(bm_query, context.name_index, context.config)
    stack = init_stack(context.config)
    answers = init_answers(
        context.config.solver.answer_max_answers,
        context.config.solver.answer_max_nodes,
        context.whole_program.symbol.shape[0],
    )

    machine = init_machine(
        compiled_program.code,
        compiled_program.entry,
        stack,
    )  ## withouth trace

    compile_ns = 0
    query_ns = 0

    ## We canot use timer inside jittable functions so we have to measure it like this seperately
    t1 = perf_counter_ns()
    compile_static_n_times(context.whole_program, code_area, context.config, n_iterations)
    t2 = perf_counter_ns()
    query_once_n_times(code_area, machine, q, stack, answers, context.config, n_iterations)
    t3 = perf_counter_ns()

    compile_ns = t2 - t1
    query_ns = t3 - t2

    row = {
        "wampy_static_compile_avg_us": compile_ns / n_iterations / 1_000,
        "wampy_static_query_avg_us": query_ns / n_iterations / 1_000,
    }

    print(
        "WAMpy static compile -> query:"
        f" compile: {row['wampy_static_compile_avg_us']:.2f} us ;"
        f" query: {row['wampy_static_query_avg_us']:.2f} us ;"
    )

    return row


def run_dynamic_benchmark(context, n_iterations):
    code_area = init_compiler(context.config)
    compiled_program = compile(context.whole_program, code_area, context.config)
    q = build_query(bm_query, context.name_index, context.config)
    stack = init_stack(context.config)
    answers = init_answers(
        context.config.solver.answer_max_answers,
        context.config.solver.answer_max_nodes,
        context.whole_program.symbol.shape[0],
    )

    machine = init_machine(
        compiled_program.code,
        compiled_program.entry,
        stack,
    )  ## withouth trace

    compile_ns = 0
    query_ns = 0

    ## We canot use timer inside jittable functions so we have to measure it like this seperately
    t1 = perf_counter_ns()
    compile_dynamic_n_times(code_area, context.static_program, context.dynamic_program, context.config, n_iterations)
    t2 = perf_counter_ns()
    query_once_n_times(code_area, machine, q, stack, answers, context.config, n_iterations)
    t3 = perf_counter_ns()

    compile_ns = t2 - t1
    query_ns = t3 - t2

    row = {
        "wampy_dynamic_compile_avg_us": compile_ns / n_iterations / 1_000,
        "wampy_dynamic_query_avg_us": query_ns / n_iterations / 1_000,
    }

    print(
        "WAMpy dynamic compile -> query:"
        f" compile: {row['wampy_dynamic_compile_avg_us']:.2f} us ;"
        f" query: {row['wampy_dynamic_query_avg_us']:.2f} us ;"
    )

    return row


def run_janus_benchmark(context, n_iterations):
    q = bm_query.replace(".", "")  ## remove trailing .
    compile_ns = 0
    query_ns = 0

    ## We time it "together" here since janus calls are not jittable anyways
    for _ in range(n_iterations):
        t1 = perf_counter_ns()
        janus.consult("benchmark", context.janus_src)
        t2 = perf_counter_ns()
        answer = janus.query_once(q)
        t3 = perf_counter_ns()
        compile_ns += t2 - t1
        query_ns += t3 - t2

    row = {
        "janus_compile_avg_us": compile_ns / n_iterations / 1_000,
        "janus_query_avg_us": query_ns / n_iterations / 1_000,
    }

    print(
        "janus compile -> query:"
        f" compile: {row['janus_compile_avg_us']:.2f} us ;"
        f" query: {row['janus_query_avg_us']:.2f} us ;"
    )

    return row


def make_benchmark_context(config):
    whole_src = PROGRAM_STATIC_SRC + PROGRAM_DYNAMIC_SRC
    whole_program, name_index = parse(whole_src, config)

    static_program = parse_with_name_index(
        PROGRAM_STATIC_SRC,
        name_index,
        config,
    )

    dynamic_program = parse_with_name_index(
        PROGRAM_DYNAMIC_SRC,
        name_index,
        config,
    )

    return BenchmarkContext(
        config=config,
        whole_program=whole_program,
        static_program=static_program,
        dynamic_program=dynamic_program,
        janus_src=JANUS_SRC_PREFIX + whole_src,
        name_index=name_index,
    )


def make_wampy_dynamic_callable(context, n_iterations):
    static_state = init_compiler(context.config)
    return lambda: compile_dynamic_n_times(
        static_state,
        context.static_program,
        context.dynamic_program,
        context.config,
        n_iterations,
    )


def run_benchmark_pass(context, n_iterations):
    row = {
        "n_iterations": n_iterations,
    }

    row.update(run_static_benchmark(context, n_iterations))
    row.update(run_dynamic_benchmark(context, n_iterations))
    row.update(run_janus_benchmark(context, n_iterations))

    return row


def main():
    config = load_config_toml(WAM_CONFIG_TOML)
    context = make_benchmark_context(config)

    results = []

    for sample in range(SAMPLE_COUNT):
        print("~ SAMPLE:", sample)

        if sample == 0:
            print("Warmup / JIT compilation overhead")
            run_benchmark_pass(context, WARMUP_ITERATIONS)

        for run_index, n_iterations in enumerate(ITERATION_COUNTS):
            print(f"Run:{run_index} ; n_iterations:{n_iterations}")

            row = run_benchmark_pass(context, n_iterations)
            row = {
                "sample": sample,
                **row,
            }

            results.append(row)

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
