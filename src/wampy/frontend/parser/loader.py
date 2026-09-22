"""Load Prolog source into an initialized AST program and symbol table."""

from collections.abc import Iterable

import numpy as np

from wampy.config import (
    DEFAULT_CONFIG,
    ASTConfig,
    WAMConfig,
    normalize_config,
    resolve_symbol_id_dtype,
)
from wampy.frontend.ast.ops.transforms import copy_term_between_programs
from wampy.frontend.ast.program import ASTProgram, init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID, calculate_symbol_id_positions
from wampy.frontend.symbol_table import SymbolTable, intern_symbol

from .ast_builder import add_prolog_term
from .syntax import SYMBOL_RE, _split_clauses


def parse(
    source: str,
    symbol_table: SymbolTable,
    config: WAMConfig = DEFAULT_CONFIG,
    *,
    variable_names_by_term: list[dict[str, int]] | None = None,
) -> tuple[ASTProgram, SymbolTable]:
    """Parse Prolog source and extend an existing symbol table in place.

    Clauses and explicit ``?-`` query terms use this same parsing path. One AST
    term is created for every source term, and the AST is automatically sized
    to fit the complete input.

    ``variable_names_by_term`` optionally receives one named-variable mapping
    per parsed term.
    """
    config = normalize_config(config)

    if not isinstance(source, str):
        raise TypeError("parse expects a Prolog source string")

    _validate_symbol_table_layout(symbol_table, config.frontend.ast)

    clauses = _split_clauses(source)
    symbol_dtype = resolve_symbol_id_dtype(config.frontend.ast.symbol_id_dtype)
    max_symbol_id = int(np.iinfo(symbol_dtype).max)
    for symbol in _extract_symbols(clauses):
        intern_symbol(symbol_table, symbol, max_symbol_id=max_symbol_id)

    term_count = len(clauses)
    ast_config = config.frontend.ast
    if term_count > ast_config.max_terms:
        config = config._replace(
            frontend=config.frontend._replace(
                ast=ast_config._replace(max_terms=term_count),
            ),
        )

    program = init_ast_program(config)
    for source_term in clauses:
        variable_names = {} if variable_names_by_term is not None else None
        stripped = source_term.strip()
        is_query = stripped.startswith("?-")
        if is_query:
            stripped = stripped[2:].strip()
            if stripped.endswith("."):
                stripped = stripped[:-1].strip()
            root_gid = CoreSymID.QUERY
        else:
            stripped = source_term
            root_gid = CoreSymID.CLAUSE

        add_prolog_term(
            program,
            stripped,
            symbol_table,
            root_gid=root_gid,
            variable_names_out=variable_names,
        )
        if variable_names_by_term is not None:
            variable_names_by_term.append(variable_names or {})

    return program, symbol_table


def extract_symbols(sources: str | Iterable[str]) -> tuple[str, ...]:
    """Return user symbols found in source text using WAMpy's lexer rules."""
    if isinstance(sources, str):
        sources = (sources,)
    return tuple(_extract_symbols(list(sources)))


def load_prolog(
    source: str,
    program: ASTProgram,
    symbol_table: SymbolTable,
    config: WAMConfig = DEFAULT_CONFIG,
    *,
    start_term_id: int = 0,
) -> list[int]:
    """Parse Prolog once and load its terms into an existing AST program."""
    if start_term_id < 0:
        raise ValueError("start_term_id must be >= 0")

    parsed_program, _ = parse(source, symbol_table, config)
    parsed_term_count = int(parsed_program.term_count[0])
    end_term_id = start_term_id + parsed_term_count

    if start_term_id >= program.node_symbols.shape[0] and parsed_term_count:
        raise ValueError("start_term_id is outside the destination AST program")
    if end_term_id > program.node_symbols.shape[0]:
        raise ValueError("Parsed Prolog does not fit into the destination AST program")

    imported_term_ids: list[int] = []
    for source_term_id in range(parsed_term_count):
        destination_term_id = start_term_id + source_term_id
        copy_term_between_programs(
            parsed_program,
            source_term_id,
            program,
            destination_term_id,
        )
        imported_term_ids.append(destination_term_id)

    if end_term_id > program.term_count[0]:
        program.term_count[0] = end_term_id

    return imported_term_ids


def _validate_symbol_table_layout(symbol_table: SymbolTable, config: ASTConfig) -> None:
    _, first_user_symbol_id = calculate_symbol_id_positions(config.max_variables_per_term)
    user_symbol_id_limit = first_user_symbol_id + config.max_user_symbols
    if (
        symbol_table.first_user_symbol_id != first_user_symbol_id
        or symbol_table.user_symbol_id_limit != user_symbol_id_limit
    ):
        raise ValueError("SymbolTable layout does not match the frontend AST configuration")


def _extract_symbols(clauses: list[str]) -> list[str]:
    """Extract non-variable, non-built-in symbols from source clauses."""
    symbols = set()
    for clause in clauses:
        for quoted_atom, atom in SYMBOL_RE.findall(clause):
            symbol = quoted_atom or atom
            if not symbol:
                continue
            if symbol in {",", "true", "not", "naf", "nfa"}:
                continue
            symbols.add(symbol)

    return sorted(symbols)
