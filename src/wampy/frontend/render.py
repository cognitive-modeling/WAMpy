import numpy as np
from numba import jit

from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.symbol_table import SymbolTable
from wampy.frontend.ast.program import NONE, ROOT_ID, Program

_USAGE_EMPTY = np.empty(0, dtype=np.int64)

# Rendering is display/debug code and has crashed on cached overload reloads.
# Keep the executable WAM path cached, but leave rendering uncached.


@jit(cache=False)
def display(
    program: Program,
    symbol_table: SymbolTable,
    term_id: int = -1,
    root_id: int = ROOT_ID,
    term_usage=_USAGE_EMPTY,
) -> None:
    """
    JIT-safe compact dump:
      /* 0000 */  <term>
    """
    n_terms = program.children.shape[0]

    if term_id < 0:
        for t in range(n_terms):
            text, _ = render_display_term_jit(program, symbol_table, t, root_id)
            print(term_label_jit(t) + text)

    elif term_id < n_terms:
        text, _ = render_display_term_jit(program, symbol_table, term_id, root_id)
        print(term_label_jit(term_id) + text)


@jit(cache=False)
def symbol_for_id_jit(symbol_table: SymbolTable, symbol_id: np.int64) -> str:
    symbol_by_id = symbol_table.symbol_by_id
    if symbol_id in symbol_by_id:
        return symbol_by_id[symbol_id]
    return ""


@jit(cache=False)
def term_label_jit(term_id: int) -> str:
    label = str(term_id)
    while len(label) < 4:
        label = "0" + label
    return "/* " + label + " */  "


@jit(cache=False)
def right_align_str_jit(text: str, width: int) -> str:
    out = text
    while len(out) < width:
        out = " " + out
    return out


@jit(cache=False)
def usage_for_term_jit(term_usage, term_id: int) -> np.int64:
    if term_id < 0:
        return np.int64(0)
    if term_id < term_usage.shape[0]:
        return np.int64(term_usage[term_id])
    return np.int64(0)


@jit(cache=False)
def has_term_content_jit(program: Program, term_id: int, root_id: int) -> bool:
    if term_id < 0 or term_id >= program.symbol.shape[0]:
        return False
    return np.int64(program.symbol[term_id, root_id]) != np.int64(SymID.EMPTY.value)


@jit(cache=False)
def render_display_term_jit(program: Program, symbol_table: SymbolTable, term_id: int, root_id: int):
    return render_term_line_jit(program, symbol_table, term_id, root_id), ""


@jit(cache=False)
def symbol_to_str_jit(sym: np.int64, symbol_table: SymbolTable) -> str:
    if sym >= np.int64(SymID.VAR_FIRST.value) and sym <= np.int64(SymID.VAR_LAST.value):
        return "V" + str(1 + (sym - np.int64(SymID.VAR_FIRST.value)))

    symbol_text = symbol_for_id_jit(symbol_table, sym)
    if len(symbol_text) > 0:
        return symbol_text

    if sym == np.int64(SymID.TRUE.value):
        return "true"
    if sym == np.int64(SymID.AND.value):
        return "and"
    if sym == np.int64(SymID.NOT_PROVABLE.value):
        return "\\+"
    if sym == np.int64(SymID.CLAUSE.value):
        return "clause"
    if sym == np.int64(SymID.EMPTY.value):
        return ""
    if sym == np.int64(SymID.QUERY.value):
        return "query"

    return str(sym)


@jit(cache=False)
def term_to_str_jit(
    program: Program,
    term_id: int,
    node_id: int,
    symbol_table: SymbolTable,
    depth: int = 0,
    max_depth: int = 1024,
    parenthesize_and: bool = False,
) -> str:
    if depth > max_depth:
        return "<max-depth>"

    sym = np.int64(program.symbol[term_id, node_id])
    if sym == np.int64(SymID.TRUE.value):
        return "true"

    max_child_slots = program.children.shape[2]
    if sym == np.int64(SymID.AND.value):
        left = ""
        right = ""
        found = 0
        for i in range(max_child_slots):
            cid = np.int64(program.children[term_id, node_id, i])
            if cid == np.int64(NONE):
                continue
            if found == 0:
                left = term_to_str_jit(program, term_id, cid, symbol_table, depth + 1, max_depth, parenthesize_and)
            elif found == 1:
                right = term_to_str_jit(program, term_id, cid, symbol_table, depth + 1, max_depth, parenthesize_and)
            found += 1

        if found == 0:
            return "true"
        if found == 1:
            return left
        out = left + ", " + right
        if parenthesize_and:
            return "(" + out + ")"
        return out

    if sym == np.int64(SymID.NOT_PROVABLE.value):
        child = "true"
        found = 0
        for i in range(max_child_slots):
            cid = np.int64(program.children[term_id, node_id, i])
            if cid == np.int64(NONE):
                continue
            if found == 0:
                child = term_to_str_jit(program, term_id, cid, symbol_table, depth + 1, max_depth)
            found += 1
        return "\\+ " + child

    head = symbol_to_str_jit(sym, symbol_table)
    args = ""
    n_args = 0
    for i in range(max_child_slots):
        cid = np.int64(program.children[term_id, node_id, i])
        if cid == np.int64(NONE):
            continue
        child = term_to_str_jit(program, term_id, cid, symbol_table, depth + 1, max_depth, True)
        if n_args == 0:
            args = child
        else:
            args = args + ", " + child
        n_args += 1

    if n_args == 0:
        return head
    return head + "(" + args + ")"


@jit(cache=False)
def render_term_line_jit(program: Program, symbol_table: SymbolTable, term_id: int, root_id: int) -> str:
    max_child_slots = program.children.shape[2]
    root_sym = np.int64(program.symbol[term_id, root_id])

    if root_sym == np.int64(SymID.CLAUSE.value):
        head_id = np.int64(-1)
        body_id = np.int64(-1)
        found = 0
        for i in range(max_child_slots):
            cid = np.int64(program.children[term_id, root_id, i])
            if cid == np.int64(NONE):
                continue
            if found == 0:
                head_id = cid
            elif found == 1:
                body_id = cid
            found += 1

        if head_id < 0:
            return "true."
        head = term_to_str_jit(program, term_id, head_id, symbol_table)
        if body_id < 0:
            return head + "."
        body = term_to_str_jit(program, term_id, body_id, symbol_table)
        return head + " :- " + body + "."

    line = ""
    n_goals = 0
    for i in range(max_child_slots):
        cid = np.int64(program.children[term_id, root_id, i])
        if cid == np.int64(NONE):
            continue
        goal = term_to_str_jit(program, term_id, cid, symbol_table)
        if n_goals == 0:
            line = goal + "."
        else:
            line = line + " " + goal + "."
        n_goals += 1
    return line


def display_as_tree(program: Program, symbol_table: SymbolTable | None = None, term_id=None, show_ids=False):
    """
    Debug-print program terms as ASCII trees.

    - symbol_table: maps symbol IDs (uint16) to symbols (e.g. add/zero/s)
    - term_id: if None, prints all terms
    - show_ids: if True, prefixes each label with "<node_id>: "
               if False (default), prints only labels
    """
    symbol_by_id = {} if symbol_table is None else symbol_table.symbol_by_id

    children = program.children
    symbol = program.symbol

    # if None, print all terms
    if term_id is None:
        for t in range(children.shape[0]):
            print("\n=== TERM " + str(t) + " ===")
            display_as_tree(program, symbol_table=symbol_table, term_id=t, show_ids=show_ids)
        return

    def children_of_py(node_id):
        res = []
        num_child_slots = children.shape[2]
        for i in range(num_child_slots):
            x = children[term_id, node_id, i]
            if x != NONE:
                res.append(int(x))
        return res

    def label_only(val: int) -> str:
        # Variables FIRST (avoid symbol_by_id overriding them)
        vfirst = int(SymID.VAR_FIRST)
        vlast = int(SymID.VAR_LAST)
        if vfirst <= val <= vlast:
            idx = 1 + (val - vfirst)  # V1, V2, ...
            return "V" + str(idx)

        # Predicate / atom binding
        if val in symbol_by_id:
            return str(symbol_by_id[val])

        # Fallback to GID or raw int
        try:
            return str(SymID(val).name)
        except Exception:
            return str(val)

    def fmt_label(node_id: int) -> str:
        val = int(symbol[term_id, node_id])
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
