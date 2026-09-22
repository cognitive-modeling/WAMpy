"""
WAMpy is a Numba-accelerated implementation of a restricted
[Warren Abstract Machine (WAM)](https://en.wikipedia.org/wiki/Warren_Abstract_Machine)
for executing Prolog programs from Python.

The package provides two levels of abstraction:

- **High-level API** — use `Prolog` and `Query` for stateful Prolog execution
  and backtracking.
- **Compiler/runtime API** — use the frontend, compiler, and runtime modules
  directly when access to the compilation and execution pipeline is required.

A typical high-level program looks like:

```python
from wampy import Prolog

prolog = Prolog(
    '''
    parent(anakin, luke).
    parent(padme, luke).
    '''
)

print(prolog.ast)
print(prolog.code)

query = prolog.query("parent(X, luke).")

for solution in query:
    print(solution)
```

For lower-level access:

```python
machine = init_machine()

status = run(
    machine,
    compiled_program,
    compiled_query,
)

while status == WAMStatus.SUCCESS:
    # inspect current answer

    status = redo(
        machine,
        compiled_program,
        compiled_query,
    )
```


## Architecture

WAMpy separates source processing, compilation, and execution into distinct
layers.


| Layer | Package / module | Responsibility |
| --- | --- | --- |
| Public API | `wampy.api` | Stateful `Prolog`, `Query`, and solution interfaces |
| Frontend | `wampy.frontend` | Parsing, AST construction, symbols, and rendering |
| Compiler | `wampy.compiler` | Compilation of Prolog programs into WAM instructions |
| Runtime | `wampy.runtime` | Heap, stack, trail, unification, and WAM instruction execution |
| Configuration | `wampy.config` | Frontend, compiler, runtime, and aggregate configuration |


```
     Source
        │
        │ parse()
        ▼

SymbolTable     ASTProgram        CompilerState
                    │                   │
                    └─────────┬─────────┘
                              │ compile_program()
                              ▼
                      CompiledProgram       Machine
                              │                   │
                              └────────┬──────────┘
                                       │
                                       ▼
                                     run()
                                       │
                                       ▼
                                    emulate()
```


## Compilation Pipeline

At a high level:

1. The frontend parses Prolog source into clauses, goals, and symbols.
2. Terms are represented by the AST and associated symbol metadata.
3. The compiler transforms the program into WAM instructions stored in a
   `CompilerState`.
4. Query execution initializes the required runtime state.
5. The runtime interpreter executes WAM instructions and performs unification.
6. Successful bindings are decoded into Python-visible solutions.
7. Backtracking resumes execution from the most recent choice point until no
   alternatives remain.


## Query Execution

WAMpy follows Prolog's depth-first, left-to-right resolution strategy.


Resolution proceeds conceptually as follows:

1. Select the leftmost goal.
2. Try matching clauses in program order.
3. Preserve alternatives using choice points.
4. Rename clause variables apart before use.
5. Unify the current goal with the selected clause head.
6. On success, replace the goal with the clause body.
7. Continue depth-first with the resulting goal list.
8. When no goals remain, decode the query-variable bindings as a solution.
9. On further iteration, backtrack to the most recent choice point.
10. Stop when all alternatives have been exhausted.


## Runtime Model

The runtime maintains the mutable state required by WAM execution.

```
                 WAM
                  │
        ┌─────────┴─────────┐
        │                   │
    Code Area             Machine
        │                   │
  instructions      ┌───────┴────────┐
  predicate table   │                │
                   state          memory
              P, CP, E, B      heap
              H, HB, TR        trail
              S, mode, X[]     stack
                               pdl
```

The principal runtime structures are:

- **Heap** — stores runtime terms and variable references.
- **Machine state** — stores registers and memory pointers.
- **Machine memory** — owns heap, trail, stack, and PDL storage.
- **Choice points** — preserve alternative execution paths for backtracking.
- **Trail** — records bindings that must be undone during backtracking.
- **Unification state** — coordinates term matching and variable binding.


## Public API

For most users, `Prolog` is the preferred entry point.

### Stateful API

- `Prolog` — owns a compiled Prolog program.
- `Query` — iterable stateful query execution.
- `CompiledQuery` — reusable prepared query representation.
- `Solution` — decoded query result.
- `QueryError` — query execution error.
- `QueryPhase` — query lifecycle state.
- `Prolog.ast` and `Prolog.code` — printable diagnostic views.

This API is appropriate when queries need to be iterated, interleaved, resumed,
or explicitly closed.

### Compiler and execution API

The lower-level modules expose building blocks including:

- `parse` for clauses and explicit `?-` query terms
- `wampy.api.compile_program` and `wampy.api.compile_query`
- `wampy.runtime.interpreter.run`, `redo`, and `emulate`
- `Program`, `ProgramMetadata`, and `SymbolTable`
- `CompilerState`
- WAM configuration objects and runtime initialization utilities

These interfaces are useful for experiments, compiler development, runtime
inspection, and custom execution pipelines.


## Configuration

Configuration is split by architectural layer:

```mermaid
flowchart LR
    Config["WAMConfig"]

    Config --> Frontend["FrontendConfig"]
    Config --> AST["ASTConfig"]
    Config --> Compiler["CompilerConfig"]
    Config --> Runtime["RuntimeConfig"]

    File["TOML configuration"] --> Load["load_config_toml()"]
    Load --> Config

    Overrides["Runtime overrides"] --> Apply["apply_overrides()"]
    Apply --> Config
```

`WAMConfig` provides the aggregate configuration, while the individual
configuration classes isolate options belonging to the frontend, AST,
compiler, and runtime.


## Design Principles

WAMpy is intended as a Python library rather than a standalone Prolog
interpreter.

The architecture therefore favors:

- explicit Python APIs,
- inspectable intermediate representations,
- separation between parsing, compilation, and runtime execution,
- deterministic ownership of query execution state,
- bounded runtime resources,
- and components that can be reused independently in experiments.


## Current Scope

WAMpy implements a deliberately restricted Prolog/WAM subset.

Notable current limitations include:

- X registers for arguments and temporaries, plus environment-backed Y variables;
- bounded environment frames with CE and CP;
- no native Prolog lists;
- user-level cut (`!`), including neck cuts and cuts after calls;
- no disjunction (`;`);
- no clause indexing;
- limited arithmetic and built-in predicates;
- no general meta-call or SWI-Prolog built-in library;
- bounded stack, answer, unification, and execution-step resources.

These constraints keep the runtime comparatively small and explicit while the
compiler and execution model continue to evolve.

compiler
    compile_program(...)  # fills CompiledProgram in place; returns None
    compile_query(...)    # fills CompiledQuery in place; returns None
        │
        ▼
runtime
    Machine
        │
        ├── heap
        ├── trail
        └── stack
              ├── environments
              └── choice points
        │
        ▼
    run()
    redo()
    emulate()
"""

from wampy.compiler.compiled_query import CompiledQuery
from wampy.config import (
    DEFAULT_CONFIG,
    ASTConfig,
    CompilerConfig,
    FrontendConfig,
    RuntimeConfig,
    WAMConfig,
    apply_overrides,
    debug_config,
    load_config_toml,
)
from wampy.frontend.ast.ops.analysis import count_clauses, count_nodes
from wampy.frontend.ast.program import (
    ASTProgram,
    init_ast_program,
)
from wampy.frontend.ast.program_metadata import (
    ProgramMetadata,
    SymbolKind,
    classify_program,
    init_program_metadata,
)
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.parser import load_prolog, parse
from wampy.frontend.symbol_table import (
    SymbolTable,
    build_symbol_table,
    empty_symbol_table,
    merge_symbol_tables,
    symbol_table_from_mapping,
)
from wampy.status import WAMStatus

from .api import ast
from .api.jitable import (
    init_machine,
    init_machine_registers,
    reset_machine,
    reset_machine_registers,
)
from .api.prolog import Prolog
from .api.query import (
    Query,
    QueryError,
    QueryPhase,
)
from .frontend.solution import (
    Solution,
    decode_bindings,
    decode_term,
    decode_variables,
)

__all__ = [
    "DEFAULT_CONFIG",
    "ASTConfig",
    "ASTProgram",
    "CompiledQuery",
    "CompilerConfig",
    "CoreSymID",
    "FrontendConfig",
    "ProgramMetadata",
    "Prolog",
    "Query",
    "QueryError",
    "QueryPhase",
    "RuntimeConfig",
    "Solution",
    "SymbolKind",
    "SymbolTable",
    "WAMConfig",
    "WAMStatus",
    "apply_overrides",
    "ast",
    "build_symbol_table",
    "classify_program",
    "count_clauses",
    "count_nodes",
    "debug_config",
    "decode_bindings",
    "decode_term",
    "decode_variables",
    "empty_symbol_table",
    "init_ast_program",
    "init_machine",
    "init_machine_registers",
    "init_program_metadata",
    "load_config_toml",
    "load_prolog",
    "merge_symbol_tables",
    "parse",
    "reset_machine",
    "reset_machine_registers",
    "symbol_table_from_mapping",
]
