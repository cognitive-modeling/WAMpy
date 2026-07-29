# WAMpy

## Installation

```console
pip install wampy-prolog
```
### Usage.

WAMpy is used as a Python library rather than as a standalone Prolog interpreter.

```
import wampy as wam
ast_program, symbol_table = wam.parse("parent(anakin, luke).")
compiled = wam.compile(ast_program)
wam.query_from_str(compiled, "parent(X, luke).", symbol_table)
```

## Limitations

The current implementation provides an X-register subset of the WAM supporting atoms, variables, compound structures, predicate calls, tail calls, unification, and linear clause backtracking.
    It omits standard environment and Y-register instructions, native list operations, clause-indexing instructions, and canonical user-level cut support.
    WAMpy also does not provide the general SWI-Prolog built-in predicate library such as arithmetic, and meta-calls.
