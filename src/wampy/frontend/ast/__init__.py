"""
AST data structures and operations for the WAM compiler frontend.

Includes the array-backed program representation, read-only analysis helpers,
validation checks, low-level mutation primitives, and higher-level semantic
transforms.

ast/
  program.py       | Program representation and core AST storage.
  gid.py           | Global identifiers and node symbols.
  goals.py         | Goal-related helpers and abstractions.
  analysis.py      | Read-only AST queries and traversal.
  validation.py    | Structural and semantic checks / rejection predicates.
  mutation.py      | Low-level in-place tree operations.
  transforms.py    | Higher-level AST rewrites.
"""

from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.ast.goals import Goals, init_goal, init_goals

from wampy.frontend.ast.analysis import *
from wampy.frontend.ast.transforms import *

from wampy.frontend.symbol_table import (
    SymbolTable,
    build_symbol_table,
    empty_symbol_table,
    merge_symbol_tables,
    symbol_table_from_mapping,
)
from wampy.frontend.ast.program import (
    CONCEPT_ROOT,
    NONE,
    ROOT_ID,
    Program,
    Term,
    add_child_node,
    count_nodes,
    init_ast_structure,
    init_program,
    remove_all_descendant,
    update_usage,
)

__all__ = [
    "SymID",
    "Goals",
    "init_goal",
    "init_goals",
    "clause_head_body",
    "collect_head_nodes",
    "get_count_clauses",
    "get_arity_of_node",
    "is_structure_of_terms_equal",
    "is_compound",
    "is_sym_a_var",
    "copy_and_override_term",
    "SymbolTable",
    "build_symbol_table",
    "empty_symbol_table",
    "merge_symbol_tables",
    "symbol_table_from_mapping",
    "CONCEPT_ROOT",
    "NONE",
    "ROOT_ID",
    "Program",
    "Term",
    "add_child_node",
    "count_nodes",
    "init_ast_structure",
    "init_program",
    "remove_all_descendant",
    "update_usage",
]
