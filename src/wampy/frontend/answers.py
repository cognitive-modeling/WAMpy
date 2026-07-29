"""
## Answer

A
├─ B
│  ├─ D
│  └─ E
└─ C

(value, aritiy)

(A,2), (B,2), (D,0), (E,0), (C,0)
"""

import re
from collections import namedtuple
import numpy as np
from numba import jit

from wampy.frontend.symbol_table import SymbolTable
from wampy.runtime.stack import TAG

Answers = namedtuple(
    "Answers",
    [
        "rows",
        "term_usage",
    ],
)


def init_answers(n: int, max_nodes: int, num_terms: int = 0) -> Answers:
    return Answers(
        rows=init_prefix_answers(n, max_nodes),
        term_usage=np.zeros((n, num_terms), dtype=np.int64),
    )


def make_prefix_answers_dtype(max_nodes: int) -> np.dtype:
    return np.dtype(
        [
            ("symbols", np.int64, (max_nodes,)),
            ("tags", np.uint8, (max_nodes,)),
            ("aritys", np.uint16, (max_nodes,)),
            ("len", np.uint16),
        ]
    )


def init_prefix_answers(n: int, max_nodes: int) -> np.ndarray:
    return np.zeros((n,), dtype=make_prefix_answers_dtype(max_nodes))


def build_correct_prefix_answers(
    correct_answers_str: list[list[str]],
    symbol_table: SymbolTable,
    max_answer_heap: int = 512,
) -> np.ndarray:
    id_by_symbol = symbol_table.id_by_symbol

    class Node:
        __slots__ = ("tag", "sym", "children")

        def __init__(self, tag, sym, children=None):
            self.tag = np.uint8(tag)
            self.sym = int(sym)
            self.children = children or []

    def _skip_ws(s, i):
        n = len(s)
        while i < n and s[i].isspace():
            i += 1
        return i

    def _parse_quoted_atom(s, i):
        assert s[i] == "'"
        i += 1
        j = i
        while j < len(s) and s[j] != "'":
            j += 1
        if j >= len(s):
            raise ValueError("Unterminated quoted atom")
        symbol = s[i:j]
        i = j + 1
        if symbol not in id_by_symbol:
            raise KeyError(f"Atom '{symbol}' not in symbol table")
        return Node(TAG.CON, id_by_symbol[symbol]), i

    def _parse_ident(s, i):
        j = i
        n = len(s)
        while j < n and (s[j].isalnum() or s[j] == "_"):
            j += 1
        if j == i:
            raise ValueError(f"Expected identifier at pos {i}: {s[i:i+20]!r}")
        return s[i:j], j

    def _parse_term(s, i, var_ids):
        i = _skip_ws(s, i)
        if i >= len(s):
            raise ValueError("Unexpected end")

        if s[i] == "'":
            return _parse_quoted_atom(s, i)

        if s[i].isalpha() or s[i] == "_":
            symbol, j = _parse_ident(s, i)
            j = _skip_ws(s, j)

            if j < len(s) and s[j] == "(":
                if symbol not in id_by_symbol:
                    raise KeyError(f"Functor '{symbol}' not in symbol table")
                j += 1
                children = []
                j = _skip_ws(s, j)
                if j < len(s) and s[j] == ")":
                    j += 1
                    return Node(TAG.STR, id_by_symbol[symbol], children), j

                while True:
                    child, j = _parse_term(s, j, var_ids)
                    children.append(child)
                    j = _skip_ws(s, j)
                    if j >= len(s):
                        raise ValueError("Unterminated '('")
                    if s[j] == ",":
                        j += 1
                        continue
                    if s[j] == ")":
                        j += 1
                        break
                    raise ValueError(f"Expected ',' or ')' at pos {j}")
                return Node(TAG.STR, id_by_symbol[symbol], children), j

            if symbol[0].isupper():
                if symbol not in var_ids:
                    var_ids[symbol] = len(var_ids)
                return Node(TAG.REF, var_ids[symbol]), j

            if symbol in id_by_symbol:
                return Node(TAG.CON, id_by_symbol[symbol]), j

            raise ValueError(
                f"Unexpected identifier '{symbol}'. Use quoted atoms for constants, "
                f"capitalized for variables, or f(...) for structures."
            )

        raise ValueError(f"Unexpected char {s[i]!r} at pos {i}")

    def _encode_prefix(root: Node, out_symbols, out_tags, out_aritys):
        stack = [root]
        n = 0
        while stack:
            node = stack.pop()
            if n >= out_symbols.shape[0]:
                raise ValueError("max_nodes too small for correct answer")

            out_tags[n] = node.tag
            out_symbols[n] = np.int64(node.sym)

            if node.tag == TAG.STR:
                out_aritys[n] = np.uint16(len(node.children))
                for c in reversed(node.children):
                    stack.append(c)
            else:
                out_aritys[n] = np.uint16(0)

            n += 1
        return n

    per_query = np.empty((len(correct_answers_str),), dtype=object)

    for qi, answers_list in enumerate(correct_answers_str):
        arr = init_prefix_answers(len(answers_list), max_answer_heap)

        for ai, src in enumerate(answers_list):
            s = src.strip()
            if s.endswith("."):
                s = s[:-1]

            var_ids = {}
            term, pos = _parse_term(s, 0, var_ids)
            pos = _skip_ws(s, pos)
            if pos != len(s):
                raise ValueError(f"Trailing junk at pos {pos}: {s[pos:pos+40]!r}")

            # Solver answers are encoded with a synthetic top-level TAG.STR root
            # (functor + arity), even for 0-arity predicates like `'1'.`.
            # Match that by canonicalizing top-level constants to STR/arity=0.
            if term.tag == TAG.CON:
                out_symbols = arr[ai]["symbols"]
                out_tags = arr[ai]["tags"]
                out_aritys = arr[ai]["aritys"]

                out_tags[0] = np.uint8(TAG.STR)
                out_symbols[0] = np.int64(int(term.sym))
                out_aritys[0] = np.uint16(0)
                arr[ai]["len"] = np.uint16(1)
                continue

            ln = _encode_prefix(term, arr[ai]["symbols"], arr[ai]["tags"], arr[ai]["aritys"])
            arr[ai]["len"] = np.uint16(ln)

        per_query[qi] = arr

    return per_query


def prefix_equal(a, b):
    la = int(a["len"])
    lb = int(b["len"])
    if la != lb:
        return False
    for i in range(la):
        if a["tags"][i] != b["tags"][i]:
            return False
        if a["symbols"][i] != b["symbols"][i]:
            return False
        if a["aritys"][i] != b["aritys"][i]:
            return False
    return True


def any_correct_match(found_answer, correct_answers_arr):
    for k in range(correct_answers_arr.shape[0]):
        if prefix_equal(found_answer, correct_answers_arr[k]):
            return True
    return False


_ATOM_SAFE_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")


def _fmt_atom(symbol: str) -> str:
    """Unquoted for standard atoms, quoted+escaped otherwise."""
    if _ATOM_SAFE_RE.match(symbol):
        return symbol
    return "'" + symbol.replace("'", "''") + "'"


def decode_answer(answer, symbol_table: SymbolTable) -> str:
    """
    Decode a prefix-encoded answer into a readable Prolog-like term.

    - TAG.STR -> functor(args...)   (arity 0 -> functor)
    - TAG.CON -> atom (unquoted if safe, else 'quoted')
    - TAG.REF -> variable: X<id>
    Always returns a trailing '.'.
    """
    symbols = answer["symbols"]
    tags = answer["tags"]
    aritys = answer["aritys"]
    ln = int(answer["len"])
    symbol_by_id = symbol_table.symbol_by_id

    def sym_name(sym_id: int) -> str:
        return symbol_by_id.get(int(sym_id), f"<SYM:{int(sym_id)}>")

    out: list[str] = []
    frames: list[int] = []  # remaining children for each open "("

    def finish_node():
        """
        Called after a node is fully emitted (leaf or completed structure).
        Decrements parent arity, emits separators and closes parentheses as needed.
        """
        while frames:
            frames[-1] -= 1
            if frames[-1] > 0:
                out.append(", ")
                return
            out.append(")")
            frames.pop()
            # continue: completing this structure completes a node for its parent too

    i = 0
    while i < ln:
        tag = int(tags[i])
        sym = int(symbols[i])
        ar = int(aritys[i])
        i += 1

        if tag == int(TAG.STR):
            fname = _fmt_atom(sym_name(sym))
            if ar <= 0:
                out.append(fname)
                finish_node()
            else:
                out.append(fname)
                out.append("(")
                frames.append(ar)  # children to come
        elif tag == int(TAG.CON):
            out.append(_fmt_atom(sym_name(sym)))
            finish_node()
        elif tag == int(TAG.REF):
            out.append(f"X{sym}")
            finish_node()
        else:
            out.append(f"<TAG:{tag},SYM:{sym}>")
            finish_node()

    # Defensive close (should be empty if encoding well-formed)
    while frames:
        out.append(")")
        frames.pop()

    s = "".join(out)
    if not s.endswith("."):
        s += "."
    return s
