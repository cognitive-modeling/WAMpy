"""Prolog parsing and AST construction.

``parse`` is the sole source parser for both knowledge-base clauses and explicit
``?-`` query terms. It always receives an existing symbol table and extends it
in place.

TODO TARGET
frontend/parser/
|- __init__.py
|- tokenizer.py
|- syntax.py
`- builder.py

Tokenizer: "What characters form a token?"
Syntax: "What sequence of tokens forms valid Prolog?"
Builder: "How do I encode valid Prolog in WAMpy's array AST?"
"""

# ruff: noqa: F401 - compatibility re-exports intentionally include private helpers

from wampy.frontend.ast.program import ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.symbol_table import SymbolTable

from .ast_builder import (
    _build_term,
    _new_variable_context,
    add_prolog_term,
)
from .loader import (
    _extract_symbols,
    _validate_symbol_table_layout,
    extract_symbols,
    load_prolog,
    parse,
)
from .syntax import (
    SYMBOL_RE,
    _has_top_level_comma,
    _negation_alias_prefix,
    _parse_not_alias,
    _parse_term,
    _split_clause,
    _split_clauses,
    _split_goals,
    _split_top_level_terms,
    _strip_block_comments,
    _strip_outer_parentheses,
    is_prolog_var,
)

__all__ = [
    "SYMBOL_RE",
    "ASTProgram",
    "CoreSymID",
    "SymbolTable",
    "add_prolog_term",
    "extract_symbols",
    "is_prolog_var",
    "load_prolog",
    "parse",
]
