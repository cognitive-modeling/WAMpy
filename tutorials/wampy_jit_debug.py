#!/usr/bin/env python3
"""Measure WAMpy import, setup, JIT compilation, and steady-state compile time.

Run from the same environment in which WAMpy is installed/developed:

    python wampy_jit_debug.py

Useful variants:

    python wampy_jit_debug.py --debug-cache
    python wampy_jit_debug.py --python-impl
    python wampy_jit_debug.py --source-file path/to/program.pl
    python wampy_jit_debug.py --cache-dir /tmp/wampy-numba-cache --debug-cache

For a true cold-cache measurement, point --cache-dir at a new/empty directory.
For a cache-hit measurement, run the same command a second time without
changing WAMpy source or the cache directory.

```
rm -rf /tmp/wampy-numba-cache

# Genuine cold cache
python wampy_jit_debug.py \
    --cache-dir /tmp/wampy-numba-cache \
    --debug-cache

# Same cache, new process: should show cache-hit behavior
python wampy_jit_debug.py \
    --cache-dir /tmp/wampy-numba-cache \
    --debug-cache
```
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

DEFAULT_SOURCE = """
parent(anakin, luke).
parent(padme, luke).
parent(luke, ben).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
"""


def elapsed(start: float) -> float:
    return time.perf_counter() - start


class CompilationResult(NamedTuple):
    compiler_state: Any
    compiled_program: Any


def compile_once(api: Any, program: Any) -> CompilationResult:
    compiler_state = api.init_compiler_state()
    compiled_program = api.init_compiled_program()
    api.compile_program(program, compiled_program, compiler_state)
    return CompilationResult(compiler_state, compiled_program)


def state_summary(result: CompilationResult) -> str:
    state = result.compiler_state
    return (
        f"pc={int(state.pc[0])}, "
        f"pc_onto={int(state.pc_onto[0])}, "
        f"predicates={int(state.predicate_count[0])}"
    )


def assert_same_compilation(left: CompilationResult, right: CompilationResult) -> None:
    """Check the important compiler outputs without timing the comparison."""
    import numpy as np

    left_state = left.compiler_state
    right_state = right.compiler_state
    left_program = left.compiled_program
    right_program = right.compiled_program
    left_pc = int(left_state.pc[0])
    right_pc = int(right_state.pc[0])
    assert left_pc == right_pc, (left_pc, right_pc)
    assert int(left_state.pc_onto[0]) == int(right_state.pc_onto[0])
    assert int(left_state.predicate_count[0]) == int(right_state.predicate_count[0])
    np.testing.assert_array_equal(left_program.code[:left_pc], right_program.code[:right_pc])
    np.testing.assert_array_equal(left_program.predicate_entry, right_program.predicate_entry)
    np.testing.assert_array_equal(
        left_state.predicate_symbols_sorted,
        right_state.predicate_symbols_sorted,
    )
    np.testing.assert_array_equal(
        left_state.predicate_slots_sorted,
        right_state.predicate_slots_sorted,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile WAMpy import and Numba compiler startup costs."
    )
    parser.add_argument(
        "--debug-cache",
        action="store_true",
        help="Set NUMBA_DEBUG_CACHE=1 before importing WAMpy.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Set NUMBA_CACHE_DIR before importing WAMpy.",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        help="Compile this Prolog source instead of the built-in sample.",
    )
    parser.add_argument(
        "--python-impl",
        action="store_true",
        help="Also time wampy.compile after the JIT measurements.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Numba cache settings must be configured before importing WAMpy/Numba.
    if args.debug_cache:
        os.environ["NUMBA_DEBUG_CACHE"] = "1"
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(args.cache_dir.resolve())

    if args.source_file is None:
        source = DEFAULT_SOURCE
        source_name = "built-in sample"
    else:
        source = args.source_file.read_text(encoding="utf-8")
        source_name = str(args.source_file)

    print("=== Environment ===")
    print(f"Python:          {platform.python_version()}")
    print(f"Executable:      {sys.executable}")
    print(f"Source:          {source_name}")
    print(f"NUMBA_CACHE_DIR: {os.environ.get('NUMBA_CACHE_DIR', '<default>')}")
    print(f"NUMBA_DEBUG_CACHE: {os.environ.get('NUMBA_DEBUG_CACHE', '0')}")
    print()

    # 1. Import timing. This includes importing Numba as a WAMpy dependency.
    start = time.perf_counter()
    import wampy
    from wampy import api as wampy_api

    import_time = elapsed(start)

    import numba

    print("=== Import ===")
    print(f"WAMpy file:      {wampy.__file__}")
    print(f"Numba version:   {numba.__version__}")
    print(f"import wampy:    {import_time:.6f} s")
    print()

    # 2. Setup timing: parse source only. Do not call compile here.
    start = time.perf_counter()
    symbols = wampy.empty_symbol_table()
    program, _symbols = wampy.parse(source, symbols)
    setup_time = elapsed(start)

    print("=== Setup ===")
    print(f"parse/setup:     {setup_time:.6f} s")
    compile_jit_signatures = getattr(wampy_api.compile_program, "signatures", None)
    print(f"JIT signatures before first call: {compile_jit_signatures}")
    print()

    # 3. First compile. On a cache miss this includes Numba compilation.
    start = time.perf_counter()
    state_first = compile_once(wampy_api, program)
    first_compile_time = elapsed(start)

    print("=== First compiled call ===")
    print(f"compile #1:      {first_compile_time:.6f} s")
    print(f"state:           {state_summary(state_first)}")
    compile_jit_signatures = getattr(wampy_api.compile_program, "signatures", None)
    print(f"JIT signatures after first call:  {compile_jit_signatures}")
    print()

    # 4. Second compile in the same process. This should contain essentially
    # no JIT compilation cost for the already-seen signature.
    start = time.perf_counter()
    state_second = compile_once(wampy_api, program)
    second_compile_time = elapsed(start)

    assert_same_compilation(state_first, state_second)

    print("=== Second compiled call ===")
    print(f"compile #2:      {second_compile_time:.6f} s")
    print(f"state:           {state_summary(state_second)}")
    print("result check:    identical")
    print()

    python_time = None
    if args.python:
        start = time.perf_counter()
        state_python = compile_once(wampy_api, program)
        python_time = elapsed(start)

        assert_same_compilation(state_first, state_python)

        print("=== Plain Python implementation ===")
        print(f"compile:    {python_time:.6f} s")
        print(f"state:           {state_summary(state_python)}")
        print("result check:    identical to compiled path")
        print()

    print("=== Summary ===")
    print(f"import:          {import_time:.6f} s")
    print(f"setup/parse:     {setup_time:.6f} s")
    print(f"first compile:   {first_compile_time:.6f} s")
    print(f"second compile:  {second_compile_time:.6f} s")
    if python_time is not None:
        print(f"compile:    {python_time:.6f} s")

    if first_compile_time > max(second_compile_time * 10.0, 1.0):
        print("\nInterpretation: most compile startup cost is in the first compiled call.")
    elif import_time > max(first_compile_time * 2.0, 1.0):
        print("\nInterpretation: import itself is unusually expensive; inspect eager JIT work.")
    else:
        print("\nInterpretation: no single phase dominates by the script's simple heuristic.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
