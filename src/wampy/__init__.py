"""

1. Parse the Prolog program into clauses (head and body) and the query into an initial goal list
2. Compile all terms (clauses and query) into WAM-style heap templates (REF, CON, STR, FUN)
3. Initialize runtime state: heap, trail, choice-point stack, unification stack

[Source code]
   ↓
[Parser]
   ↓
  AST
   ↓
Compiler
   ↓
WAM code
   ↓
WAM interpreter / VM
   ↓
Runtime system
   ├─ Heap (terms)
   ├─ Stack (environments, choice points)
   ├─ Trail
   └─ Unification engine

* Start resolution with the goal list containing the query
* Select the leftmost goal (Prolog's left-to-right rule)
* For that goal, try each clause in program order
* Before trying a clause, push a choice point to allow backtracking
* Copy the clause (rename variables apart) using `copy_term` so it has fresh variables
* Unify the current goal with the copied clause head
* If unification fails, backtrack and try the next clause
* If unification succeeds, replace the goal with the clause body (prepend body goals to the remaining goals)
* Recurse on the new goal list (depth-first search)
* When the goal list becomes empty, a solution is found
* Extract the solution by dereferencing the original query variables
* Record or output the solution
* Backtrack to the most recent choice point to look for alternative solutions
* Terminate when no choice points remain and all clause alternatives are exhausted
"""

from . import ast as ast

from wampy.engine import query_from_str, query

from wampy.compiler.compiler import CodeArea, compile, init_compiler, reset_compiler, compile_program_onto
from wampy.config import WAMConfig, DEFAULT_CONFIG, load_config_toml
from wampy.frontend.answers import (
    Answers,
    any_correct_match,
    build_correct_prefix_answers,
    decode_answer,
    init_answers,
    init_prefix_answers,
)
from wampy.frontend.parser import (
    parse,
    parse_with_symbol_table,
    build_program,
    build_new_program,
    build_queries,
    build_query,
)
from wampy.frontend.ast.goals import Goals
from wampy.frontend.symbol_table import (
    SymbolTable,
    build_symbol_table,
    empty_symbol_table,
    merge_symbol_tables,
    symbol_table_from_mapping,
)
from wampy.frontend.ast.program import (
    Program,
    count_nodes,
    init_program,
)
from wampy.frontend.ast.analysis import get_count_clauses
from wampy.frontend.render import display
from wampy.runtime.stack import init_stack, reset_stack
from wampy.status import WAMStatus

__all__ = [
    "ast",
    "any_correct_match",
    "build_new_program",
    "build_program",
    "build_correct_prefix_answers",
    "build_symbol_table",
    "build_queries",
    "build_query",
    "compile",
    "compile_program_onto",
    "CodeArea",
    "count_nodes",
    "get_count_clauses",
    "display",
    "decode_answer",
    "DEFAULT_CONFIG",
    "init_compiler",
    "init_answers",
    "init_prefix_answers",
    "init_program",
    "init_stack",
    "load_config_toml",
    "merge_symbol_tables",
    "symbol_table_from_mapping",
    "SymbolTable",
    "Answers",
    "parse",
    "parse_with_symbol_table",
    "Program",
    "Goals",
    "reset_compiler",
    "reset_stack",
    "query_from_str",
    "query",
    "empty_symbol_table",
    "WAMConfig",
    "WAMStatus",
]
