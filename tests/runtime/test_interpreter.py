# tests/rational_program/wam/test_interpreter2.py
import numpy as np
import pytest

from wampy.config import load_config_toml
from wampy.status import WAMStatus
from wampy.frontend.parser import build_new_program
from wampy.compiler.compiler import OP, init_compiler, compile
from wampy.runtime.stack import init_stack, make_var, make_const
from wampy.runtime.interpreter import init_machine, run, redo, exec_UNIFY_CONST
from wampy.runtime.term_decode import read_term

CONFIG_TOML = """
[ast]
num_terms = 30
max_terms_nodes = 100
max_terms_nodes_childs = 5

[entry]
max_arity = 8
""".strip()


def wam_config():
    return load_config_toml(CONFIG_TOML)


def _compile_clauses(clauses: list[str]):
    program_src = "\n".join(clauses) + "\n"
    config = wam_config()
    program, symbol_table = build_new_program(program_src, config)
    compiled = compile(program, init_compiler(config), config)
    # symbol_by_id: {id -> symbol}
    symbol_by_id = {int(k): v for k, v in symbol_table.items()}
    id_by_symbol = {v: int(k) for k, v in symbol_by_id.items()}
    return compiled.code[: compiled.pc.value], compiled.entry, symbol_by_id, id_by_symbol


def _sid(id_by_symbol: dict[str, int], s: str) -> int:
    """
    Symbol id lookup that tolerates quoted-vs-unquoted atoms.
    E.g. "1" vs "'1'", "+" vs "'+'".
    """
    if s in id_by_symbol:
        return id_by_symbol[s]
    q = f"'{s}'"
    if q in id_by_symbol:
        return id_by_symbol[q]
    raise KeyError(f"Symbol not in program: {s!r} (also tried {q!r})")


def _term_to_py(term, symbol_by_id):
    """Convert read_term output into a friendly Python structure using symbols."""
    if not isinstance(term, tuple):
        return term

    tag = term[0]
    if tag == "CONST":
        return symbol_by_id[int(term[1])]
    if tag == "VAR":
        return ("VAR", int(term[1]))
    if tag == "STRUCT":
        _, symbol_id, args = term
        symbol = symbol_by_id[int(symbol_id)]
        return (symbol, [_term_to_py(a, symbol_by_id) for a in args])

    return term


def _raise_on_hard_error(st: WAMStatus):
    if st in (WAMStatus.SUCCESS, WAMStatus.EXHAUSTED):
        return
    raise RuntimeError(f"WAM error: {WAMStatus(int(st)).name}")


def _collect_all_solutions(machine, stack, fun_id, arity, args, symbol_by_id, max_answers=10):
    """
    Enumerate solutions for a query installed into args.
    Assumes run()/redo() return WAMStatus (no exceptions).
    """
    for i, addr in enumerate(args):
        machine.X[i] = addr
    machine.pc[0] = machine.entry[fun_id, arity]
    machine.cp[0] = -1

    sols = []

    st = run(machine)
    _raise_on_hard_error(st)
    if st == WAMStatus.EXHAUSTED:
        return sols

    # First solution found; enumerate further solutions with redo()
    while True:
        sols.append([_term_to_py(read_term(stack, a), symbol_by_id) for a in args])
        if len(sols) >= max_answers:
            break

        st = redo(machine)
        _raise_on_hard_error(st)
        if st == WAMStatus.EXHAUSTED:
            break

    return sols


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.parametrize(
    "clauses, query_functor, query_args, expected_any",
    [
        # include needed atoms in the program (even for failure cases) so they exist in the symbol table
        (["p(a)."], "p", ["a"], True),
        (["p(a).", "__atom(b)."], "p", ["b"], False),
        (["edge(a,b).", "edge(b,c)."], "edge", ["a", "b"], True),
        (["edge(a,b).", "edge(b,c)."], "edge", ["a", "c"], False),
    ],
)
def test_facts_with_constants_success_and_failure(clauses, query_functor, query_args: list[str], expected_any):
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)
    stack = init_stack()
    m = init_machine(code, entry, stack)

    fun_id = id_by_symbol[query_functor]
    args = [make_const(stack, _sid(id_by_symbol, a)) for a in query_args]

    sols = _collect_all_solutions(m, stack, fun_id, len(args), args, symbol_by_id, max_answers=10)
    assert (len(sols) > 0) == expected_any
    if expected_any:
        assert sols[0] == query_args


def test_choice_point_enumerates_multiple_facts_in_order():
    clauses = ["p(a).", "p(b).", "p(c)."]
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    stack = init_stack()
    m = init_machine(code, entry, stack)

    X = make_var(stack)
    fun_id = id_by_symbol["p"]

    sols = _collect_all_solutions(m, stack, fun_id, 1, [X], symbol_by_id, max_answers=10)
    got = [s[0] for s in sols]
    assert got == ["a", "b", "c"]


def test_rule_with_multiple_goals_backtracks_across_goals():
    clauses = [
        "parent(alice, bob).",
        "parent(bob, claire).",
        "parent(bob, hans).",
        "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
    ]
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    stack = init_stack()
    m = init_machine(code, entry, stack)

    X = make_const(stack, _sid(id_by_symbol, "alice"))
    Z = make_var(stack)
    fun_id = id_by_symbol["grandparent"]

    sols = _collect_all_solutions(m, stack, fun_id, 2, [X, Z], symbol_by_id, max_answers=10)
    assert sols == [["alice", "claire"], ["alice", "hans"]]


def test_backtracking_restores_bindings_across_answers():
    clauses = [
        "p(X, Y) :- q(X), r(Y).",
        "p(X, Y) :- q(Y), r(X).",
        "q(a).",
        "r(b).",
    ]
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    stack = init_stack()
    m = init_machine(code, entry, stack)

    X = make_var(stack)
    Y = make_var(stack)
    fun_id = id_by_symbol["p"]

    sols = _collect_all_solutions(m, stack, fun_id, 2, [X, Y], symbol_by_id, max_answers=10)
    assert sols == [["a", "b"], ["b", "a"]]


def test_argument_aliasing_success_and_failure():
    # ensure atoms exist in the program symbol table
    clauses = ["same(X, X).", "__atom(a).", "__atom(b)."]
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    fun_id = id_by_symbol["same"]

    # success: same(a,a)
    stack = init_stack()
    m = init_machine(code, entry, stack)
    a1 = make_const(stack, _sid(id_by_symbol, "a"))
    a2 = make_const(stack, _sid(id_by_symbol, "a"))
    sols_ok = _collect_all_solutions(m, stack, fun_id, 2, [a1, a2], symbol_by_id, max_answers=10)
    assert sols_ok == [["a", "a"]]

    # failure: same(a,b)
    stack2 = init_stack()
    m2 = init_machine(code, entry, stack2)
    a = make_const(stack2, _sid(id_by_symbol, "a"))
    b = make_const(stack2, _sid(id_by_symbol, "b"))
    sols_bad = _collect_all_solutions(m2, stack2, fun_id, 2, [a, b], symbol_by_id, max_answers=10)
    assert sols_bad == []


def test_math_ground_and_two_var_queries():
    clauses = [
        "sym(zero, '0').",
        "sym(s(zero), '1').",
        "sym(s(s(zero)), '2').",
        "sym(s(s(s(zero))), '3').",
        "sym(s(s(s(s(zero)))), '4').",
        "sym(s(s(s(s(s(zero))))), '5').",
        "add(zero, Y, Y).",
        "add(s(X), Y, s(Z)) :- add(X, Y, Z).",
        "math(N1, '+', N2, '=', A) :- sym(P1, N1), sym(P2, N2), add(P1, P2, R), sym(R, A).",
    ]
    code, entry, symbol_by_id, id_by_symbol = _compile_clauses(clauses)
    fun_id = id_by_symbol["math"]

    # ground query: math('1','+','2','=',X) -> X='3'
    stack = init_stack()
    m = init_machine(code, entry, stack)

    n1 = make_const(stack, _sid(id_by_symbol, "1"))
    plus = make_const(stack, _sid(id_by_symbol, "+"))
    n2 = make_const(stack, _sid(id_by_symbol, "2"))
    eq = make_const(stack, _sid(id_by_symbol, "="))
    X = make_var(stack)

    sols = _collect_all_solutions(m, stack, fun_id, 5, [n1, plus, n2, eq, X], symbol_by_id, max_answers=10)
    assert sols == [["1", "+", "2", "=", "3"]]

    # two-var query: math('1','+',X,'=',Y) -> 5 solutions: (0,1)..(4,5)
    stack2 = init_stack()
    m2 = init_machine(code, entry, stack2)

    n1b = make_const(stack2, _sid(id_by_symbol, "1"))
    plusb = make_const(stack2, _sid(id_by_symbol, "+"))
    Xb = make_var(stack2)
    eqb = make_const(stack2, _sid(id_by_symbol, "="))
    Yb = make_var(stack2)

    sols2 = _collect_all_solutions(m2, stack2, fun_id, 5, [n1b, plusb, Xb, eqb, Yb], symbol_by_id, max_answers=10)
    got_pairs = [(row[2], row[4]) for row in sols2]
    assert got_pairs == [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")]


def test_unify_const_read_mode_does_not_allocate_heap():
    code, entry, _, id_by_symbol = _compile_clauses(["p(a)."])
    stack = init_stack()
    m = init_machine(code, entry, stack)

    a = make_const(stack, _sid(id_by_symbol, "a"))
    m.mode[0] = 0  # READ
    m.S[0] = a

    heap_before = int(stack.heap_top[0])
    st = exec_UNIFY_CONST(m, _sid(id_by_symbol, "a"))

    assert st == WAMStatus.SUCCESS
    assert int(stack.heap_top[0]) == heap_before


def test_fail_backtracks_to_latest_choicepoint():
    code = np.array(
        [
            [OP.TRY, 2, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    stack = init_stack()
    m = init_machine(code, init_compiler().entry, stack)

    st = run(m)

    assert st == WAMStatus.SUCCESS
    assert stack.cp_top[0] == 0


def test_cut_removes_newer_choicepoints_before_fail():
    code = np.array(
        [
            [OP.MARK_CUT, 0, 0, 0],
            [OP.TRY, 4, 0, 0],
            [OP.CUT, 0, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    stack = init_stack()
    m = init_machine(code, init_compiler().entry, stack)

    st = run(m)

    assert st == WAMStatus.EXHAUSTED
    assert stack.cp_top[0] == 0


def test_choicepoint_restore_drops_cut_markers_from_failed_branch():
    code = np.array(
        [
            [OP.TRY, 3, 0, 0],
            [OP.MARK_CUT, 0, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    stack = init_stack()
    m = init_machine(code, init_compiler().entry, stack)

    st = run(m)

    assert st == WAMStatus.SUCCESS
    assert m.cut_top[0] == 0
    assert stack.cp_top[0] == 0
