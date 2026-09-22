"""Build array-backed AST terms from parsed Prolog syntax."""

from wampy.frontend.ast.ops.analysis import (
    get_variable_capacity,
    get_variable_symbol_id,
)
from wampy.frontend.ast.ops.mutation import add_child_node, allocate_term
from wampy.frontend.ast.program import ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.symbol_table import SymbolTable, intern_symbol

from .syntax import _parse_term, _split_clause, _split_goals, is_prolog_var


def add_prolog_term(
    program: ASTProgram,
    term_str: str,
    symbol_table: SymbolTable,
    root_gid: int = CoreSymID.CLAUSE,
    *,
    variable_names_out: dict[str, int] | None = None,
) -> int:
    """Append one Prolog clause or query to an AST program.

    ``variable_names_out`` optionally receives the named variable -> symbol-ID
    mapping produced while building this term.
    """
    term_id = allocate_term(program, root_gid)
    root = 0

    variable_names, next_variable_id = _new_variable_context(program)

    if root_gid == CoreSymID.QUERY:
        query_str = term_str.strip()
        if not query_str:
            raise ValueError("Query has no callable goal")
        _build_goals(
            program,
            term_id,
            root,
            query_str,
            symbol_table,
            variable_names,
            next_variable_id,
            wrap_single=False,
        )
    else:
        head_str, body_str = _split_clause(term_str)

        _build_term(
            program,
            term_id,
            root,
            head_str,
            symbol_table,
            variable_names,
            next_variable_id,
        )

        if body_str is None:
            add_child_node(program, term_id, root, CoreSymID.TRUE)
        else:
            _build_goals(
                program,
                term_id,
                root,
                body_str,
                symbol_table,
                variable_names,
                next_variable_id,
                wrap_single=True,
            )

    if variable_names_out is not None:
        variable_names_out.update(variable_names)

    return term_id


def _build_goals(
    program: ASTProgram,
    term_id: int,
    parent_id: int,
    source: str,
    symbol_table: SymbolTable,
    variable_names: dict[str, int],
    next_variable_id: list[int],
    *,
    wrap_single: bool,
) -> None:
    """Build a goal or conjunction below ``parent_id``."""
    goals = _split_goals(source)
    while len(goals) > 1 and goals[-1] == "true":
        goals = goals[:-1]

    if not goals:
        raise ValueError("Query has no callable goal")

    if len(goals) == 1:
        if goals[0] == "true":
            add_child_node(program, term_id, parent_id, CoreSymID.TRUE)
        elif wrap_single:
            and_root, _ = add_child_node(program, term_id, parent_id, CoreSymID.CONJUNCTION)
            _build_term(
                program,
                term_id,
                and_root,
                goals[0],
                symbol_table,
                variable_names,
                next_variable_id,
            )
            add_child_node(program, term_id, and_root, CoreSymID.TRUE)
        else:
            _build_term(
                program,
                term_id,
                parent_id,
                goals[0],
                symbol_table,
                variable_names,
                next_variable_id,
            )
        return

    and_root, _ = add_child_node(program, term_id, parent_id, CoreSymID.CONJUNCTION)
    current = and_root

    for i, goal in enumerate(goals):
        _build_term(
            program,
            term_id,
            current,
            goal,
            symbol_table,
            variable_names,
            next_variable_id,
        )

        if i == len(goals) - 1:
            add_child_node(program, term_id, current, CoreSymID.TRUE)
        else:
            next_and, _ = add_child_node(
                program,
                term_id,
                current,
                CoreSymID.CONJUNCTION,
            )
            current = next_and


def _new_variable_context(program: ASTProgram) -> tuple[dict[str, int], list[int]]:
    return {}, [get_variable_symbol_id(program, 0)]


def _build_term(
    program: ASTProgram,
    term_id: int,
    parent_id: int,
    term_str: str,
    symbol_table: SymbolTable,
    variable_names: dict[str, int],
    next_variable_id: list[int],
) -> int:
    """Recursively build a program-term subtree and return its node ID."""
    functor, args = _parse_term(term_str)

    if functor.startswith("'") and functor.endswith("'"):
        functor = functor[1:-1]

    if is_prolog_var(functor):
        # Bare "_" is anonymous in Prolog: every occurrence is a fresh variable.
        if functor != "_" and functor in variable_names:
            functor_sym = variable_names[functor]
        else:
            next_var = next_variable_id[0]
            if next_var >= get_variable_symbol_id(program, get_variable_capacity(program)):
                raise RuntimeError(
                    f"Too many variables in term (maximum is {get_variable_capacity(program)})"
                )

            functor_sym = next_var
            next_variable_id[0] = next_var + 1
            if functor != "_":
                variable_names[functor] = functor_sym
                variable_slot = functor_sym - get_variable_symbol_id(program, 0)
                program.variable_name_ids[term_id, variable_slot] = intern_symbol(
                    symbol_table,
                    functor,
                )
    elif functor == "true":
        functor_sym = CoreSymID.TRUE
    elif functor == ",":
        functor_sym = CoreSymID.CONJUNCTION
    elif functor == "\\+":
        functor_sym = CoreSymID.NEGATION_AS_FAILURE
    elif functor == "!":
        functor_sym = CoreSymID.CUT
    else:
        if functor not in symbol_table.id_by_symbol:
            raise KeyError(f"Unknown functor: {functor}")
        functor_sym = symbol_table.id_by_symbol[functor]

    node_id, ok = add_child_node(program, term_id, parent_id, functor_sym)
    if not ok:
        raise RuntimeError("Failed to allocate term node")

    for arg in args:
        _build_term(
            program,
            term_id,
            node_id,
            arg,
            symbol_table,
            variable_names,
            next_variable_id,
        )

    return node_id
