import numpy as np

from wampy.frontend.ast.ops.analysis import (
    get_child_at,
    get_child_count,
    get_clause_head_body,
    get_node_symbol,
    get_term_count,
    is_variable_symbol,
    variable_index,
)
from wampy.frontend.ast.program import NO_NODE, NODE_ROOT_ID, ASTProgram
from wampy.frontend.ast.symbol_id import (
    CoreSymID,
)
from wampy.frontend.symbol_table import SymbolTable

_USAGE_EMPTY = np.empty(0, dtype=np.int64)


def render(
    program: ASTProgram,
    symbol_table: SymbolTable,
    term_id: int = -1,
    root_id: int = NODE_ROOT_ID,
    term_usage=_USAGE_EMPTY,
) -> str:
    """
    Return a JIT-safe compact rendering:
      /* 0000 */  <term>
    """
    n_terms = get_term_count(program)
    text = ""

    if term_id < 0:
        for t in range(n_terms):
            term_text, _ = render_display_term_jit(program, symbol_table, t, root_id)
            if len(text) > 0:
                text += "\n"
            text += term_label_jit(t) + term_text

    elif term_id < n_terms:
        term_text, _ = render_display_term_jit(program, symbol_table, term_id, root_id)
        text = term_label_jit(term_id) + term_text

    return text


def display(
    program: ASTProgram,
    symbol_table: SymbolTable,
    term_id: int = -1,
    root_id: int = NODE_ROOT_ID,
    term_usage=_USAGE_EMPTY,
) -> None:
    """Print :func:`render` output when the selected terms are non-empty."""
    text = render(program, symbol_table, term_id, root_id, term_usage)
    if len(text) > 0:
        print(text)


def symbol_for_id_jit(symbol_table: SymbolTable, symbol_id: np.int64) -> str:
    symbol_by_id = symbol_table.symbol_by_id
    if symbol_id in symbol_by_id:
        return symbol_by_id[symbol_id]
    return ""


def term_label_jit(term_id: int) -> str:
    label = str(term_id)
    while len(label) < 4:
        label = "0" + label
    return "/* " + label + " */  "


def right_align_str_jit(text: str, width: int) -> str:
    out = text
    while len(out) < width:
        out = " " + out
    return out


def usage_for_term_jit(term_usage, term_id: int) -> np.int64:
    if term_id < 0:
        return np.int64(0)
    if term_id < term_usage.shape[0]:
        return np.int64(term_usage[term_id])
    return np.int64(0)


def has_term_content_jit(program: ASTProgram, term_id: int, root_id: int) -> bool:
    if term_id < 0 or term_id >= get_term_count(program):
        return False
    return np.int64(get_node_symbol(program, term_id, root_id)) != np.int64(
        CoreSymID.NO_SYMBOL.value
    )


def render_display_term_jit(
    program: ASTProgram, symbol_table: SymbolTable, term_id: int, root_id: int
):
    return render_term_line_jit(program, symbol_table, term_id, root_id), ""


def symbol_to_str_jit(program: ASTProgram, sym: np.int64, symbol_table: SymbolTable) -> str:
    if is_variable_symbol(program, sym):
        return "V" + str(1 + variable_index(program, sym))

    symbol_text = symbol_for_id_jit(symbol_table, sym)
    if len(symbol_text) > 0:
        return symbol_text

    if sym == np.int64(CoreSymID.TRUE.value):
        return "true"
    if sym == np.int64(CoreSymID.CONJUNCTION.value):
        return ","
    if sym == np.int64(CoreSymID.NEGATION_AS_FAILURE.value):
        return "\\+"
    if sym == np.int64(CoreSymID.CUT.value):
        return "!"
    if sym == np.int64(CoreSymID.CLAUSE.value):
        return "clause"
    if sym == np.int64(CoreSymID.NO_SYMBOL.value):
        return ""
    if sym == np.int64(CoreSymID.QUERY.value):
        return "query"

    return str(sym)


def term_to_str_jit(
    program: ASTProgram,
    term_id: int,
    node_id: int,
    symbol_table: SymbolTable,
    depth: int = 0,
    max_depth: int = 1024,
    parenthesize_and: bool = False,
) -> str:
    OP_NODE = 0
    OP_COMMA = 1
    OP_CLOSE_PAREN = 2

    stack = [
        (
            OP_NODE,
            int(node_id),
            int(depth),
            1 if parenthesize_and else 0,
        )
    ]
    out = ""
    while stack:
        op, current_node_id, current_depth, current_parenthesize = stack.pop()

        if op == OP_COMMA:
            out += ", "
            continue

        if op == OP_CLOSE_PAREN:
            out += ")"
            continue

        if current_depth > max_depth:
            out += "<max-depth>"
            continue

        sym = np.int64(get_node_symbol(program, term_id, current_node_id))
        if sym == np.int64(CoreSymID.TRUE.value):
            out += "true"
            continue

        if sym == np.int64(CoreSymID.CONJUNCTION.value):
            found = get_child_count(program, term_id, current_node_id)
            left_id = np.int64(get_child_at(program, term_id, current_node_id, 0))
            right_id = np.int64(get_child_at(program, term_id, current_node_id, 1))

            if found == 0:
                out += "true"
                continue

            if found == 1:
                stack.append((OP_NODE, int(left_id), current_depth + 1, current_parenthesize))
                continue

            # The AST stores a trailing true/0 as the internal end-of-body
            # sentinel.  It is useful to the runtime, but it is not part of
            # the source-level rule that should be displayed.  Only suppress
            # it for body conjunctions; an explicit `(Value, true)` inside a
            # predicate argument must still be rendered verbatim.
            right_is_body_sentinel = (
                current_parenthesize == 0
                and get_node_symbol(program, term_id, int(right_id))
                == np.int64(CoreSymID.TRUE.value)
                and get_child_count(program, term_id, int(right_id)) == 0
            )
            if right_is_body_sentinel:
                stack.append((OP_NODE, int(left_id), current_depth + 1, current_parenthesize))
                continue

            if current_parenthesize:
                out += "("
                stack.append((OP_CLOSE_PAREN, 0, 0, 0))

            stack.append((OP_NODE, int(right_id), current_depth + 1, current_parenthesize))
            stack.append((OP_COMMA, 0, 0, 0))
            stack.append((OP_NODE, int(left_id), current_depth + 1, current_parenthesize))
            continue

        if sym == np.int64(CoreSymID.NEGATION_AS_FAILURE.value):
            child_id = np.int64(get_child_at(program, term_id, current_node_id, 0))

            out += "\\+ "
            if child_id == np.int64(NO_NODE):
                out += "true"
            else:
                stack.append((OP_NODE, int(child_id), current_depth + 1, 0))
            continue

        out += symbol_to_str_jit(program, sym, symbol_table)

        n_args = get_child_count(program, term_id, current_node_id)

        if n_args == 0:
            continue

        out += "("
        stack.append((OP_CLOSE_PAREN, 0, 0, 0))

        for i in range(n_args - 1, -1, -1):
            cid = np.int64(get_child_at(program, term_id, current_node_id, i))

            if i < n_args - 1:
                stack.append((OP_COMMA, 0, 0, 0))

            stack.append((OP_NODE, int(cid), current_depth + 1, 1))

    return out


def render_term_line_jit(
    program: ASTProgram, symbol_table: SymbolTable, term_id: int, root_id: int
) -> str:
    root_sym = np.int64(get_node_symbol(program, term_id, root_id))

    if root_sym == np.int64(CoreSymID.CLAUSE.value):
        head_id, body_id = get_clause_head_body(program, term_id)
        head_id = np.int64(head_id)
        body_id = np.int64(body_id)

        if head_id == np.int64(NO_NODE):
            return "true."

        head = term_to_str_jit(
            program,
            term_id,
            head_id,
            symbol_table,
        )

        if body_id == np.int64(NO_NODE):
            return head + "."

        # A clause whose body is exactly true/0 is a fact.
        if (
            np.int64(
                get_node_symbol(
                    program,
                    term_id,
                    int(body_id),
                )
            )
            == np.int64(CoreSymID.TRUE.value)
            and get_child_count(
                program,
                term_id,
                int(body_id),
            )
            == 0
        ):
            return head + "."

        body = term_to_str_jit(
            program,
            term_id,
            body_id,
            symbol_table,
        )
        return head + " :- " + body + "."

    line = ""
    n_goals = 0
    for i in range(get_child_count(program, term_id, root_id)):
        cid = np.int64(get_child_at(program, term_id, root_id, i))
        goal = term_to_str_jit(program, term_id, cid, symbol_table)
        if n_goals == 0:
            line = goal + "."
        else:
            line = line + " " + goal + "."
        n_goals += 1
    return line


def display_as_tree(
    program: ASTProgram, symbol_table: SymbolTable | None = None, term_id=None, show_ids=False
):
    """
    Debug-print program terms as ASCII trees.

    - symbol_table: maps symbol IDs (uint16) to symbols (e.g. add/zero/s)
    - term_id: if None, prints all terms
    - show_ids: if True, prefixes each label with "<node_id>: "
               if False (default), prints only labels
    """
    symbol_by_id = {} if symbol_table is None else symbol_table.symbol_by_id

    # if None, print all terms
    if term_id is None:
        for t in range(get_term_count(program)):
            print("\n=== TERM " + str(t) + " ===")
            display_as_tree(program, symbol_table=symbol_table, term_id=t, show_ids=show_ids)
        return

    def children_of_py(node_id):
        res = []
        for i in range(get_child_count(program, term_id, node_id)):
            res.append(int(get_child_at(program, term_id, node_id, i)))
        return res

    def label_only(val: int) -> str:
        # Variables FIRST (avoid symbol_by_id overriding them)
        if is_variable_symbol(program, val):
            idx = 1 + int(variable_index(program, val))  # V1, V2, ...
            return "V" + str(idx)

        # Predicate / atom binding
        if val in symbol_by_id:
            return str(symbol_by_id[val])

        # Fallback to GID or raw int
        try:
            return str(CoreSymID(val).name)
        except Exception:
            return str(val)

    def fmt_label(node_id: int) -> str:
        val = int(get_node_symbol(program, term_id, node_id))
        lab = label_only(val)
        if show_ids:
            return str(node_id) + ": " + lab
        return lab

    def _print(u, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        print(prefix + connector + fmt_label(u))
        ch = children_of_py(u)
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, c in enumerate(ch):
            _print(c, new_prefix, i == len(ch) - 1)

    root_id = 0
    print(fmt_label(root_id))
    root_children = children_of_py(root_id)
    for i, c in enumerate(root_children):
        _print(c, "", i == len(root_children) - 1)
