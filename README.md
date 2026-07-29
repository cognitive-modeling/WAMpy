# WAMpy

> Paper ... cite this..

## Installation

```console
pip install wampy-prolog
```

## Benchmark

## Limitations

The current implementation provides an X-register subset of the WAM supporting atoms, variables, compound structures, predicate calls, tail calls, unification, and linear clause backtracking.
    It omits standard environment and Y-register instructions, native list operations, clause-indexing instructions, and canonical user-level cut support.
    WAMpy also does not provide the general SWI-Prolog built-in predicate library such as arithmetic, and meta-calls.
