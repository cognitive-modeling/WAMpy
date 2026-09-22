"""
WAMPY_RUN_BENCHMARKS=1 pytest \\
  packages/WAMpy/tests/frontend/ast/ops/test_benchmark.py -s
"""

import os
from statistics import median
from timeit import repeat

import pytest
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.ast.ops.analysis import (
    get_child_at,
    get_child_count,
    get_node_symbol,
    get_user_symbol_id,
)
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.program import init_ast_program

pytestmark = pytest.mark.skipif(
    os.environ.get("WAMPY_RUN_BENCHMARKS") != "1",
    reason="set WAMPY_RUN_BENCHMARKS=1 to run AST microbenchmarks",
)

_READ_ITERATIONS = 200_000
_READ_REPEATS = 7


def _program_with_children(child_count: int):
    program = init_ast_program(DEFAULT_CONFIG, num_terms=1)
    assert allocate_term(program, get_user_symbol_id(program, 0)) == 0

    child_ids = []
    for child_index in range(child_count):
        child_id, ok = add_child_node(
            program,
            0,
            0,
            get_user_symbol_id(program, child_index + 1),
        )
        assert ok
        child_ids.append(int(child_id))

    return program, child_ids


def _median_ns_per_call(statement: str, namespace: dict[str, object]) -> float:
    timings = repeat(
        statement,
        globals=namespace,
        number=_READ_ITERATIONS,
        repeat=_READ_REPEATS,
    )
    return median(timings) * 1_000_000_000 / _READ_ITERATIONS


def test_ast_read_hot_path_benchmark():
    """Measure fixed binary-link and logical argument reads."""
    arity = DEFAULT_CONFIG.compiler.max_arity
    program, child_ids = _program_with_children(arity)
    middle = arity // 2
    last = arity - 1

    assert get_child_at(program, 0, 0, 0) == child_ids[0]
    assert get_child_at(program, 0, 0, middle) == child_ids[middle]
    assert get_child_at(program, 0, 0, last) == child_ids[last]

    namespace = {
        "program": program,
        "get_node_symbol": get_node_symbol,
        "get_child_count": get_child_count,
        "get_child_at": get_child_at,
        "middle": middle,
        "last": last,
    }

    results = {
        "direct node symbol": _median_ns_per_call(
            "program.node_symbols[0, 0]",
            namespace,
        ),
        "get_node_symbol": _median_ns_per_call(
            "get_node_symbol(program, 0, 0)",
            namespace,
        ),
        "get_child_count": _median_ns_per_call(
            "get_child_count(program, 0, 0)",
            namespace,
        ),
        "get_child_at[0]": _median_ns_per_call(
            "get_child_at(program, 0, 0, 0)",
            namespace,
        ),
        f"get_child_at[{middle}]": _median_ns_per_call(
            "get_child_at(program, 0, 0, middle)",
            namespace,
        ),
        f"get_child_at[{last}] cached tail": _median_ns_per_call(
            "get_child_at(program, 0, 0, last)",
            namespace,
        ),
    }

    baseline = results["direct node symbol"]
    print("\nAST binary-link read benchmark (median ns/call)")
    for name, ns_per_call in results.items():
        print(f"  {name:<32} {ns_per_call:10.1f} ns  {ns_per_call / baseline:6.2f}x")
