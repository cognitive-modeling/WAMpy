"""
Program loader.

Responsible for turning Prolog source clauses into an initialized
Program with symbol bindings, ready for WAM conversion and execution.

Program loader.

Pipeline:
·                   +-----------+
·   Prolog source → | loader.py | → Program AST → WAM → execution
·                   +-----------+

"""

import re
import numpy as np

from wampy.config import WAMConfig, DEFAULT_CONFIG
from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.ast.goals import Goals, init_goal, init_goals
from wampy.frontend.symbol_table import SymbolTable, build_symbol_table, intern_symbol
from wampy.frontend.ast.program import Program, NONE, init_program, add_child_node
from wampy.status import WAMStatus

SYMBOL_RE = re.compile(
    r"""
    '([^'\n\r]*)'                         |  # quoted atom (no newlines inside)
    (?<![A-Za-z0-9_])([a-z][A-Za-z0-9_]*)  # unquoted atom, not part of a larger token
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)


def parse(
    source: str,
    config: WAMConfig = DEFAULT_CONFIG,
) -> tuple[Program, SymbolTable]:
    """Parse source with a newly allocated symbol table."""
    return build_new_program(source, config)


def parse_with_symbol_table(
    source: str,
    symbol_table: SymbolTable,
    config: WAMConfig = DEFAULT_CONFIG,
) -> tuple[Program, SymbolTable]:
    """Parse source and extend ``symbol_table`` while preserving existing IDs."""
    return _build_program(source, symbol_table, config)


def build_program(
    source: str,
    config: WAMConfig = DEFAULT_CONFIG,
) -> tuple[Program, SymbolTable]:
    return parse(source, config)


def build_new_program(
    source: str, config: WAMConfig = DEFAULT_CONFIG
) -> tuple[Program, SymbolTable]:
    """Initial loading of an empty program (Numpy AST)"""

    if not isinstance(source, str):
        raise TypeError("build_program expects a Prolog source string")

    clauses = _split_clauses(source)
    symbol_table = build_symbol_table(_extract_symbols(clauses))
    return _build_program_from_clauses(clauses, symbol_table, config)


def _build_program(
    source: str, symbol_table: SymbolTable, config: WAMConfig
) -> tuple[Program, SymbolTable]:
    if not isinstance(source, str):
        raise TypeError("build_program expects a Prolog source string")

    clauses = _split_clauses(source)
    for symbol in _extract_symbols(clauses):
        intern_symbol(symbol_table, symbol)

    return _build_program_from_clauses(clauses, symbol_table, config)


def _build_program_from_clauses(
    clauses: list[str], symbol_table: SymbolTable, config: WAMConfig
) -> tuple[Program, SymbolTable]:
    program = init_program(config)

    for clause in clauses:
        add_prolog_term(program, clause, symbol_table)

    return program, symbol_table


def _split_clauses(source: str):
    source = _strip_block_comments(source)
    clauses = []
    buf = ""
    depth = 0
    in_quote = False

    for c in source:
        if c == "'":
            in_quote = not in_quote

        if not in_quote:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1

        if c == "." and depth == 0 and not in_quote:
            clause = buf.strip()
            if clause:
                clauses.append(clause + ".")
            buf = ""
        else:
            buf += c

    # accept final clause without trailing dot
    tail = buf.strip()
    if tail:
        clauses.append(tail if tail.endswith(".") else tail + ".")

    return clauses


def _strip_block_comments(source: str) -> str:
    out = ""
    i = 0
    in_quote = False

    while i < len(source):
        c = source[i]
        if c == "'":
            in_quote = not in_quote
            out += c
            i += 1
            continue

        if not in_quote and i + 1 < len(source) and source[i] == "/" and source[i + 1] == "*":
            i += 2
            while i + 1 < len(source) and not (source[i] == "*" and source[i + 1] == "/"):
                if source[i] == "\n":
                    out += "\n"
                i += 1
            if i + 1 < len(source):
                i += 2
            out += " "
            continue

        out += c
        i += 1

    return out


def build_query(source: str, symbol_table: SymbolTable, config: WAMConfig = DEFAULT_CONFIG):
    """
    Build a query AST with QUERY as root.

    Args:
        source: query string, e.g. "p.", "p(X).", "p(a, X)."
        symbol_table: SymbolTable from build_program/build_new_program

    Returns:
        program: Program
        fun: functor id
        arity: arity of query
        args: list of argument node ids
    """
    if not isinstance(source, str):
        raise TypeError("build_query expects a string")

    s = _strip_block_comments(source).strip()
    if not s:
        raise ValueError("Empty query")

    if s.endswith("."):
        s = s[:-1]

    # Init query program
    program = init_goal(config)
    return _build_query_into_program(program, np.uint16(0), s, symbol_table)


def build_queries(sources, symbol_table: SymbolTable, config: WAMConfig = DEFAULT_CONFIG):
    if not isinstance(sources, (list, tuple)):
        raise TypeError("build_queries expects a list/tuple of query strings")
    if len(sources) == 0:
        return tuple()

    program = init_goals(len(sources), config)
    queries = []
    for term_id, source in enumerate(sources):
        if not isinstance(source, str):
            raise TypeError("build_queries expects only string query entries")
        s = _strip_block_comments(source).strip()
        if not s:
            raise ValueError("Empty query")
        if s.endswith("."):
            s = s[:-1]
        queries.append(_build_query_into_program(program, np.uint16(term_id), s, symbol_table))
    return tuple(queries)


def _build_query_into_program(program: Program, term_id: np.uint16, source: str, symbol_table: SymbolTable):
    symbol_by_id = symbol_table.symbol_by_id
    root = 0
    program.symbol[term_id, root] = SymID.QUERY

    functor, arg_src = _parse_term(source)

    # keep quoted predicate atoms consistent with clause parsing
    if functor.startswith("'") and functor.endswith("'"):
        functor = functor[1:-1]

    # Number of query arguments parsed from the source term
    arity = len(arg_src)

    # Keep query functor ids as plain ints so tuples of Goals stay Numba-homogeneous.
    if functor == "true":
        fun = SymID.TRUE.value
    elif functor == "\\+":
        fun = SymID.NOT_PROVABLE.value
    else:
        try:
            fun = next(k for k, v in symbol_by_id.items() if v == functor)
        except StopIteration:
            raise RuntimeError(
                f"{WAMStatus.UNDEFINED_PREDICATE.name}: " f"procedure `{functor}/{arity}` does not exist"
            )

    # Build AST
    goal, ok = add_child_node(program, term_id, root, fun)
    if not ok:
        raise RuntimeError("Failed to add query goal")

    args = []
    for arg in arg_src:
        try:
            child = _build_term(program, term_id, goal, arg, symbol_table)
        except KeyError:
            raise KeyError("Missing symbol in the program: ", arg)
        args.append(child)

    for i in range(len(args) + 1, program.children.shape[2]):
        program.children[term_id, goal, i] = NONE

    # IMPORTANT: always pass numpy array to numba code (empty list breaks numba)
    args_arr = np.asarray(args, dtype=np.uint16)
    return Goals(program, fun, arity, args_arr, np.uint16(term_id))


def _extract_symbols(clauses):
    """
    Extract predicate, functor, and constant symbols from clauses.
    Variables (capitalized) and built-ins are ignored.
    """
    symbols = set()
    for clause in clauses:
        for g1, g2 in SYMBOL_RE.findall(clause):
            sym = g1 or g2
            if not sym:
                continue

            # built-ins
            if sym == "," or sym == "true" or sym == "not" or sym == "naf" or sym == "nfa":
                continue

            symbols.add(sym)

    return sorted(symbols)


def add_prolog_term(program, term_str, symbol_table: SymbolTable, root_gid=SymID.CLAUSE):
    # --------------------------------------------------
    # Helpers
    def split_goals(s):
        goals = []
        depth = 0
        in_quote = False
        buf = ""

        for c in s:
            if c == "'":
                in_quote = not in_quote

            if c == "," and depth == 0 and not in_quote:
                goals.append(buf.strip())
                buf = ""
                continue

            if not in_quote:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1

            buf += c

        if buf.strip():
            goals.append(buf.strip())

        return goals

    # --------------------------------------------------
    # Clause root
    term_id = _allocate_term(program, root_gid)
    root = 0

    head_str, body_str = _split_clause(term_str)

    # HEAD
    _build_term(program, term_id, root, head_str, symbol_table)

    # FACT
    if body_str is None:
        add_child_node(program, term_id, root, SymID.TRUE)
        return term_id

    # BODY (AND chain)
    goals = split_goals(body_str)
    while len(goals) > 1 and goals[-1] == "true":
        goals = goals[:-1]

    if len(goals) == 1 and goals[0] == "true":
        add_child_node(program, term_id, root, SymID.TRUE)
        return term_id

    and_root, _ = add_child_node(program, term_id, root, SymID.AND)
    current = and_root

    for i, g in enumerate(goals):
        # left child
        _build_term(program, term_id, current, g, symbol_table)

        if i == len(goals) - 1:
            # terminate with TRUE
            add_child_node(program, term_id, current, SymID.TRUE)
        else:
            # nest AND on the right
            next_and, _ = add_child_node(program, term_id, current, SymID.AND)
            current = next_and

    return term_id


def _allocate_term(program: Program, root_gid: int) -> int:
    roots = program.symbol[:, 0]
    for term_id, root_sym in enumerate(roots):
        if root_sym == SymID.EMPTY:
            roots[term_id] = root_gid
            return term_id
    raise RuntimeError("No empty term available")


def _split_clause(s: str):
    """
    Splits:
      head.
      head :- body.

    Quote-aware.
    """
    s = s.strip()
    if s.endswith("."):
        s = s[:-1]

    depth = 0
    in_quote = False

    i = 0
    while i < len(s) - 1:
        c = s[i]

        if c == "'":
            in_quote = not in_quote

        if not in_quote:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif depth == 0 and s[i : i + 2] == ":-":
                return s[:i].strip(), s[i + 2 :].strip()

        i += 1

    return s, None


def _build_term(program, term_id, parent_id, term_str, symbol_table: SymbolTable):
    """
    Recursively builds a program-term subtree.
    Returns the node_id of the created term.
    """

    symbol_by_id = symbol_table.symbol_by_id
    functor, args = _parse_term(term_str)

    # Normalize quoted atoms
    if functor.startswith("'") and functor.endswith("'"):
        functor = functor[1:-1]

    # Variable
    if is_prolog_var(functor):
        # Bare "_" is anonymous in Prolog: every occurrence is a fresh variable.
        existing = None
        if functor != "_":
            # Reuse a variable only for named variables (including names like _Tmp).
            for k, v in symbol_by_id.items():
                if v == functor and SymID.VAR_FIRST <= k <= SymID.VAR_LAST:
                    existing = k
                    break

        if existing is None:
            # Allocate next Var symbol
            used = 0
            for k in symbol_by_id.keys():
                if SymID.VAR_FIRST <= k <= SymID.VAR_LAST:
                    used += 1
            next_var = SymID.VAR_FIRST + used
            if next_var > SymID.VAR_LAST:
                raise RuntimeError("Too many variables in clause")

            symbol_by_id[next_var] = functor
            functor_sym = next_var
        else:
            functor_sym = existing

    else:
        # Built-in TRUE
        if functor == "true":
            functor_sym = SymID.TRUE
        elif functor == ",":
            functor_sym = SymID.AND
        elif functor == "\\+":
            functor_sym = SymID.NOT_PROVABLE
        else:
            # Constant / functor
            try:
                functor_sym = next(k for k, v in symbol_by_id.items() if v == functor)
            except StopIteration:
                raise KeyError(f"Unknown functor: {functor}")

    node_id, ok = add_child_node(program, term_id, parent_id, functor_sym)
    if not ok:
        raise RuntimeError("Failed to allocate term node")

    # Recursively build arguments
    for arg in args:
        _build_term(program, term_id, node_id, arg, symbol_table)

    return node_id


def is_prolog_var(token: str) -> bool:
    return token and (token[0].isupper() or token[0] == "_")


def _parse_term(s: str):
    """
    Parses:
      atom
      atom(t1, t2, ...)
      (t1, t2) as shorthand for ','(t1, t2)
    Returns: (functor, [subterms])
    """

    s = s.strip()

    not_operand = _parse_not_alias(s)
    if not_operand is not None:
        return "\\+", [not_operand]

    if s.startswith("\\+"):
        operand = s[2:].strip()
        if not operand:
            raise ValueError("Negation requires an operand")

        if operand.startswith("("):
            inner = _strip_outer_parentheses(operand)
            if inner is None:
                raise ValueError("Negation in parenthesized form must be exactly '\\\\+(Goal)'")
            operand = inner.strip()

        if not operand:
            raise ValueError("Negation requires an operand")

        if _has_top_level_comma(operand):
            raise ValueError("Grouped negation is not supported yet; use a predicate goal.")

        return "\\+", [operand]

    inner = _strip_outer_parentheses(s)
    if inner is not None:
        if _has_top_level_comma(inner):
            args = _split_top_level_terms(inner)
            if len(args) != 2:
                raise ValueError("Tuple shorthand expects exactly two terms")
            return ",", args
        return _parse_term(inner)

    if "(" not in s:
        return s, []

    functor, rest = s.split("(", 1)

    return functor.strip(), _split_top_level_terms(rest[:-1])


def _split_top_level_terms(s: str):
    terms = []
    depth = 0
    in_quote = False
    buf = ""

    for c in s:
        if c == "'" and not in_quote:
            in_quote = True
        elif c == "'" and in_quote:
            in_quote = False

        if c == "," and depth == 0 and not in_quote:
            terms.append(buf.strip())
            buf = ""
            continue

        if not in_quote:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1

        buf += c

    if buf:
        terms.append(buf.strip())

    return terms


def _parse_not_alias(s: str):
    alias = _negation_alias_prefix(s)
    if alias is None:
        return None

    rest = s[len(alias) :].lstrip()
    if not rest.startswith("("):
        return None

    operand = _strip_outer_parentheses(rest)
    if operand is None:
        raise ValueError("Negation alias must be exactly 'nfa(Goal)' or 'naf(Goal)' or 'not(Goal)'")

    operand = operand.strip()
    if not operand:
        raise ValueError("Negation requires an operand")
    if _has_top_level_comma(operand):
        raise ValueError("Grouped negation is not supported yet; use a predicate goal.")

    return operand


def _negation_alias_prefix(s: str):
    for alias in ("nfa", "naf", "not"):
        if not s.startswith(alias):
            continue

        if len(s) == len(alias):
            return alias

        nxt = s[len(alias)]
        if nxt == "(" or nxt.isspace():
            return alias

    return None


def _strip_outer_parentheses(s: str):
    s = s.strip()
    if len(s) < 2 or s[0] != "(" or s[-1] != ")":
        return None

    depth = 0
    in_quote = False
    for i, c in enumerate(s):
        if c == "'":
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and i != len(s) - 1:
                return None

    if in_quote or depth != 0:
        return None

    return s[1:-1]


def _has_top_level_comma(s: str) -> bool:
    depth = 0
    in_quote = False
    for c in s:
        if c == "'":
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "," and depth == 0:
            return True

    return False


def ensure_query_symbols(program: Program, symbol_table: SymbolTable, query_str: str) -> None:
    """
    Add every atom symbol that occurs in `query_str` (except the top-level predicate
    functor and built-in `true`) to `symbol_table` if it's missing.

    This enables Prolog semantics where queries may introduce new constants, e.g.
      program: p(a).
      query:   p(b).
    where `b` is not in the program.
    """
    if not isinstance(query_str, str):
        raise TypeError("ensure_query_symbols expects a string")

    s = query_str.strip()
    if not s:
        return

    # strip trailing dot for consistent parsing
    if s.endswith("."):
        s = s[:-1].strip()

    # Identify the top-level predicate functor (must NOT be auto-added)
    top_functor, _ = _parse_term(s)

    # Collect all atom symbols in the query (quoted + unquoted lowercase atoms)
    syms = set()
    for g1, g2 in SYMBOL_RE.findall(s):
        sym = g1 or g2
        if not sym:
            continue
        if sym == "true":
            continue
        if sym == ",":
            continue
        if sym == top_functor:
            continue
        syms.add(sym)

    if not syms:
        return

    existing = set(symbol_table.symbol_by_id.values())

    # Add missing symbols
    for sym in sorted(syms):
        if sym in existing:
            continue
        intern_symbol(symbol_table, sym)
        existing.add(sym)
