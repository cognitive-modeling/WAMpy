import numpy as np
import pytest
from wampy.compiler.compiled_program import CompiledProgram, init_compiled_program
from wampy.compiler.compiled_query import init_compiled_query
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.compiler.query import compile_query
from wampy.config import load_config_toml
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table
from wampy.runtime.interpreter import (
    emulate,
    exec_CALL,
    exec_EXECUTE,
    exec_PROCEED,
    exec_PUT_Y_VAL,
    exec_PUT_Y_VAR,
    exec_TRUST_ME_ELSE_FAIL,
    exec_TRY_ME_ELSE,
    exec_UNIFY_CONST,
    redo,
    run,
)
from wampy.runtime.interpreter.choice import mark_enclosing_naf_cutoff
from wampy.runtime.machine import init_machine, reset_machine
from wampy.runtime.machine.environment import (
    ENV_HEADER_SIZE,
    EnvSlot,
    allocate_environment,
    deallocate_environment,
    environment_size_at_continuation,
    environment_size_at_entry,
    environment_y_address,
)
from wampy.runtime.machine.heap import make_const, make_var
from wampy.runtime.machine.machine_registers import (
    HALT_CONTINUATION,
    UnificationMode,
)
from wampy.runtime.machine.stack import ChoicePointKind, ChoiceSlot
from wampy.runtime.term_decode import read_term
from wampy.status import WAMStatus


def _init_test_machine(code, entry, memory):
    compiled_program = CompiledProgram(
        code,
        np.asarray([code.shape[0]], dtype=np.int32),
        entry,
    )
    compiled_query = init_compiled_query()
    return memory, compiled_program, compiled_query


def _compile_clauses(clauses: list[str]):
    program_src = "\n".join(clauses) + "\n"
    symbol_table = empty_symbol_table()
    program, symbol_table = parse(program_src, symbol_table)
    compiler_state = init_compiler_state()
    compiled = init_compiled_program()
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
    )
    # symbol_by_id: {id -> symbol}
    symbol_by_id = {int(k): v for k, v in symbol_table.items()}
    id_by_symbol = {v: int(k) for k, v in symbol_by_id.items()}
    return (
        compiled.code[: compiled.code_size[0]],
        compiled.predicate_entry,
        compiler_state,
        symbol_by_id,
        id_by_symbol,
    )


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


def _collect_predicate_solutions(
    machine,
    compiled_program,
    compiled_query,
    fun_id,
    arity,
    args,
    compiler_state,
    symbol_by_id,
    max_answers=10,
):
    state = machine.registers
    memory = machine

    for i, addr in enumerate(args):
        memory.X[i] = addr

    state.P[0] = compiled_program.predicate_entry[
        find_predicate_slot(compiler_state, fun_id),
        arity,
    ]
    state.CP[0] = HALT_CONTINUATION

    solutions = []

    status = emulate(
        machine,
        compiled_program,
        compiled_query,
    )
    _raise_on_hard_error(status)

    while status == WAMStatus.SUCCESS:
        solutions.append(
            [
                _term_to_py(
                    read_term(state, memory, address),
                    symbol_by_id,
                )
                for address in args
            ]
        )

        if len(solutions) >= max_answers:
            break

        status = redo(
            machine,
            compiled_program,
            compiled_query,
        )
        _raise_on_hard_error(status)

    return solutions


# ----------------------------
# Tests
# ----------------------------


def test_logical_depth_and_limit_start_disabled_and_reset():
    machine = init_machine()
    assert machine.depth[0] == 0
    assert machine.depth_limit[0] == 0
    assert machine.return_depth[0] == 0

    machine.depth[0] = 9
    machine.depth_limit[0] = 7
    reset_machine(machine)
    assert machine.depth[0] == 0
    assert machine.depth_limit[0] == 0
    assert machine.return_depth[0] == 0


def test_call_and_proceed_restore_logical_parent_depth():
    machine = init_machine()
    entry = init_compiled_program().predicate_entry
    entry[0, 0] = 10

    assert exec_CALL(machine, entry, 0, 0, 0, 37) == WAMStatus.SUCCESS
    assert machine.depth[0] == 1
    assert machine.return_depth[0] == 0
    assert machine.registers.CP[0] == 37

    assert exec_PROCEED(machine) is False
    assert machine.depth[0] == 0
    assert machine.return_depth[0] == 0
    assert machine.registers.P[0] == 37


def test_failed_call_does_not_change_logical_depth():
    machine = init_machine()
    entry = init_compiled_program().predicate_entry
    machine.depth[0] = 6

    assert exec_CALL(machine, entry, 0, 0, 0, 37) == WAMStatus.EXHAUSTED
    assert machine.depth[0] == 6


def test_emulate_returns_generic_depth_limit_before_tail_execute():
    code = np.array([[OP.EXECUTE, 0, 0, 0]], dtype=np.int32)
    entry = init_compiled_program().predicate_entry
    entry[0, 0] = 0
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        entry,
        init_machine(),
    )
    machine.depth_limit[0] = 1
    machine.registers.P[0] = 0

    status = emulate(machine, compiled_program, compiled_query)

    assert status == WAMStatus.DEPTH_LIMIT
    assert machine.registers.P[0] == 0
    assert machine.depth[0] == 1


def test_tail_execute_contributes_to_logical_depth_without_a_call_frame():
    machine = init_machine()
    entry = init_compiled_program().predicate_entry
    entry[0, 0] = 10

    assert exec_CALL(machine, entry, 0, 0, 0, 37) == WAMStatus.SUCCESS
    assert exec_EXECUTE(machine, entry, 0, 0) == WAMStatus.SUCCESS
    assert machine.depth[0] == 2
    assert machine.return_depth[0] == 0

    assert exec_PROCEED(machine) is False
    assert machine.depth[0] == 0


def test_choicepoint_restore_restores_logical_depth_metadata():
    machine = init_machine()
    entry = init_compiled_program().predicate_entry
    entry[0, 0] = 10

    assert exec_CALL(machine, entry, 0, 0, 0, 37) == WAMStatus.SUCCESS
    assert exec_TRY_ME_ELSE(machine, 42) == WAMStatus.SUCCESS
    saved_depth = int(machine.depth[0])
    saved_return_depth = int(machine.return_depth[0])
    address = machine.registers.B[0]
    assert machine.stack.cells[address + ChoiceSlot.DEPTH] == saved_depth
    assert machine.stack.cells[address + ChoiceSlot.RETURN_DEPTH] == saved_return_depth
    assert machine.stack.cells[address + ChoiceSlot.KIND] == ChoicePointKind.NORMAL

    assert exec_EXECUTE(machine, entry, 0, 0) == WAMStatus.SUCCESS
    machine.depth[0] = 99
    machine.return_depth[0] = 88

    from wampy.runtime.interpreter import restore_choice

    restore_choice(machine, machine.registers.B[0])
    assert machine.depth[0] == saved_depth
    assert machine.return_depth[0] == saved_return_depth


def test_repeated_cutoff_stays_on_nearest_truncated_naf():
    machine = init_machine()
    assert exec_TRY_ME_ELSE(machine, 10, ChoicePointKind.NAF) == WAMStatus.SUCCESS
    outer = machine.registers.B[0]
    assert exec_TRY_ME_ELSE(machine, 20, ChoicePointKind.NAF) == WAMStatus.SUCCESS
    inner = machine.registers.B[0]

    assert mark_enclosing_naf_cutoff(machine) == WAMStatus.SUCCESS
    assert machine.stack.cells[inner + ChoiceSlot.KIND] == ChoicePointKind.NAF_TRUNCATED
    assert machine.stack.cells[outer + ChoiceSlot.KIND] == ChoicePointKind.NAF

    assert mark_enclosing_naf_cutoff(machine) == WAMStatus.SUCCESS
    assert machine.stack.cells[inner + ChoiceSlot.KIND] == ChoicePointKind.NAF_TRUNCATED
    assert machine.stack.cells[outer + ChoiceSlot.KIND] == ChoicePointKind.NAF


@pytest.mark.parametrize(
    "clauses, query_functor, query_args, expected_any",
    [
        # Include needed atoms in the program, even for failure cases, so
        # they exist in the symbol table.
        (["p(a)."], "p", ["a"], True),
        (["p(a).", "__atom(b)."], "p", ["b"], False),
        (["edge(a,b).", "edge(b,c)."], "edge", ["a", "b"], True),
        (["edge(a,b).", "edge(b,c)."], "edge", ["a", "c"], False),
    ],
)
def test_facts_with_constants_success_and_failure(
    clauses, query_functor, query_args: list[str], expected_any
):
    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)
    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(code, entry, memory)
    state = machine.registers

    fun_id = id_by_symbol[query_functor]
    args = [make_const(state, memory, _sid(id_by_symbol, a)) for a in query_args]

    sols = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        len(args),
        args,
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
    assert (len(sols) > 0) == expected_any
    if expected_any:
        assert sols[0] == query_args


def test_choice_point_enumerates_multiple_facts_in_order():
    clauses = ["p(a).", "p(b).", "p(c)."]
    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(code, entry, memory)
    state = machine.registers

    X = make_var(state, memory)
    fun_id = id_by_symbol["p"]

    sols = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        1,
        [X],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
    got = [s[0] for s in sols]
    assert got == ["a", "b", "c"]


def test_rule_with_multiple_goals_backtracks_across_goals():
    clauses = [
        "parent(alice, bob).",
        "parent(bob, claire).",
        "parent(bob, hans).",
        "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
    ]
    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(code, entry, memory)
    state = machine.registers

    X = make_const(state, memory, _sid(id_by_symbol, "alice"))
    Z = make_var(state, memory)
    fun_id = id_by_symbol["grandparent"]

    sols = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        2,
        [X, Z],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
    assert sols == [["alice", "claire"], ["alice", "hans"]]


def test_backtracking_restores_bindings_across_answers():
    clauses = [
        "p(X, Y) :- q(X), r(Y).",
        "p(X, Y) :- q(Y), r(X).",
        "q(a).",
        "r(b).",
    ]
    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(code, entry, memory)
    state = machine.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)
    fun_id = id_by_symbol["p"]

    sols = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        2,
        [X, Y],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
    assert sols == [["a", "b"], ["b", "a"]]


def test_argument_aliasing_success_and_failure():
    # ensure atoms exist in the program symbol table
    clauses = ["same(X, X).", "__atom(a).", "__atom(b)."]
    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)

    fun_id = id_by_symbol["same"]

    # success: same(a,a)
    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(code, entry, memory)
    state = machine.registers
    a1 = make_const(state, memory, _sid(id_by_symbol, "a"))
    a2 = make_const(state, memory, _sid(id_by_symbol, "a"))
    sols_ok = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        2,
        [a1, a2],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
    assert sols_ok == [["a", "a"]]

    # failure: same(a,b)
    memory2 = init_machine()
    machine2, compiled_program2, compiled_query2 = _init_test_machine(code, entry, memory2)
    state2 = machine2.registers
    a = make_const(state2, memory2, _sid(id_by_symbol, "a"))
    b = make_const(state2, memory2, _sid(id_by_symbol, "b"))
    sols_bad = _collect_predicate_solutions(
        machine2,
        compiled_program2,
        compiled_query2,
        fun_id,
        2,
        [a, b],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )
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
        "math(N1, N2, '+', A) :- sym(P1, N1), sym(P2, N2), add(P1, P2, R), sym(R, A).",
    ]

    code, entry, compiler_state, symbol_by_id, id_by_symbol = _compile_clauses(clauses)
    fun_id = id_by_symbol["math"]

    # ground query: math('1','2','+',X) -> X='3'
    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        entry,
        memory,
    )
    state = machine.registers

    n1 = make_const(state, memory, _sid(id_by_symbol, "1"))
    n2 = make_const(state, memory, _sid(id_by_symbol, "2"))
    plus = make_const(state, memory, _sid(id_by_symbol, "+"))
    X = make_var(state, memory)

    sols = _collect_predicate_solutions(
        machine,
        compiled_program,
        compiled_query,
        fun_id,
        4,
        [n1, n2, plus, X],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )

    assert sols == [["1", "2", "+", "3"]]

    # two-var query: math('1',X,'+',Y) -> 5 solutions: (0,1)..(4,5)
    memory2 = init_machine()
    machine2, compiled_program2, compiled_query2 = _init_test_machine(
        code,
        entry,
        memory2,
    )
    state2 = machine2.registers

    n1b = make_const(state2, memory2, _sid(id_by_symbol, "1"))
    Xb = make_var(state2, memory2)
    plusb = make_const(state2, memory2, _sid(id_by_symbol, "+"))
    Yb = make_var(state2, memory2)

    sols2 = _collect_predicate_solutions(
        machine2,
        compiled_program2,
        compiled_query2,
        fun_id,
        4,
        [n1b, Xb, plusb, Yb],
        compiler_state,
        symbol_by_id,
        max_answers=10,
    )

    got_pairs = [(row[1], row[3]) for row in sols2]

    assert got_pairs == [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")]


def test_unify_const_read_mode_does_not_allocate_heap():
    code, entry, _, _, id_by_symbol = _compile_clauses(["p(a)."])
    memory = init_machine()
    machine, _, _ = _init_test_machine(code, entry, memory)
    state = machine.registers

    a = make_const(state, memory, _sid(id_by_symbol, "a"))
    state.mode[0] = UnificationMode.READ
    state.S[0] = a

    heap_before = int(state.H[0])
    st = exec_UNIFY_CONST(machine, _sid(id_by_symbol, "a"))

    assert st == WAMStatus.SUCCESS
    assert int(state.H[0]) == heap_before


def test_fail_backtracks_to_latest_choicepoint():
    code = np.array(
        [
            [OP.TRY_ME_ELSE, 2, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST_ME_ELSE_FAIL, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        init_compiled_program().predicate_entry,
        memory,
    )
    state = machine.registers

    state.P[0] = 0
    state.CP[0] = HALT_CONTINUATION

    st = emulate(
        machine,
        compiled_program,
        compiled_query,
    )

    assert st == WAMStatus.SUCCESS
    assert state.B[0] == -1


def test_choice_point_register_stores_latest_stack_address():
    memory = init_machine()
    machine = memory

    assert machine.registers.B[0] == -1
    assert exec_TRY_ME_ELSE(machine, 10) == WAMStatus.SUCCESS
    assert machine.registers.B[0] == 0
    assert exec_TRY_ME_ELSE(machine, 20) == WAMStatus.SUCCESS
    assert machine.registers.B[0] == (memory.stack.choice_point_frame_size)

    assert exec_TRUST_ME_ELSE_FAIL(machine) == WAMStatus.SUCCESS
    assert machine.registers.B[0] == 0
    assert exec_TRUST_ME_ELSE_FAIL(machine) == WAMStatus.SUCCESS
    assert machine.registers.B[0] == -1


def test_allocate_and_deallocate_manage_real_environment_frame():
    memory = init_machine()
    state = memory.registers
    machine = memory
    compiled_program = init_compiled_program()
    compiled_program.code[:2] = [
        [OP.CALL, 0, 0, 0],
        [OP.CALL, 0, 0, 2],
    ]
    compiled_program.code_size[0] = 2
    compiled_query = init_compiled_query()
    assert ENV_HEADER_SIZE == 3
    assert tuple(slot.name for slot in EnvSlot) == (
        "CE",
        "CP",
        "RETURN_DEPTH",
    )

    state.CP[0] = 1
    assert allocate_environment(machine, compiled_program, compiled_query, 0) == WAMStatus.SUCCESS
    assert machine.stack.cells[EnvSlot.RETURN_DEPTH] == 0
    state.CP[0] = 2
    machine.return_depth[0] = 9

    assert allocate_environment(machine, compiled_program, compiled_query, 1) == WAMStatus.SUCCESS
    assert state.E[0] == 3
    assert state.local_top[0] == 8
    assert memory.stack.cells[3:6].tolist() == [0, 2, 9]

    first = make_var(state, memory)
    assert exec_PUT_Y_VAR(machine, 0, 0) == WAMStatus.SUCCESS
    memory.stack.cells[environment_y_address(state, 0)] = first
    assert exec_PUT_Y_VAL(machine, 0, 1) == WAMStatus.SUCCESS
    assert memory.X[1] == first

    assert deallocate_environment(machine) == WAMStatus.SUCCESS
    assert state.E[0] == 0
    assert state.CP[0] == 2
    assert machine.return_depth[0] == 9
    assert state.local_top[0] == 3


def test_allocate_uses_combined_stack_capacity():
    config = load_config_toml("""
        [runtime]
        environment_size = 2
        """)
    memory = init_machine(config.runtime)
    state = memory.registers
    machine = memory
    compiled_program = init_compiled_program()
    compiled_program.code[0] = [OP.CALL, 0, 0, 1]
    compiled_program.code_size[0] = 1
    compiled_query = init_compiled_query()
    state.CP[0] = 1

    assert allocate_environment(machine, compiled_program, compiled_query, 0) == WAMStatus.SUCCESS
    assert state.local_top[0] == ENV_HEADER_SIZE + 1


def test_environment_size_at_continuation_reads_program_and_query_calls():
    compiled_program = init_compiled_program()
    compiled_program.code[4] = [OP.CALL, 0, 0, 3]
    compiled_program.code_size[0] = 5

    compiled_query = init_compiled_query()
    compiled_query.code[1] = [OP.CALL, 0, 0, 7]
    compiled_query.code[0] = [OP.PROCEED, 0, 0, 0]
    compiled_query.code_size[0] = 2

    assert environment_size_at_continuation(compiled_program, compiled_query, 5) == 3
    assert environment_size_at_continuation(compiled_program, compiled_query, 7) == 7


def test_environment_size_at_continuation_rejects_invalid_continuations():
    compiled_program = init_compiled_program()
    compiled_program.code_size[0] = 1
    compiled_query = init_compiled_query()
    compiled_query.code_size[0] = 1
    compiled_query.code[0] = [OP.PROCEED, 0, 0, 0]

    assert environment_size_at_continuation(compiled_program, compiled_query, -1) == -1
    assert environment_size_at_continuation(compiled_program, compiled_query, 2) == -1
    assert environment_size_at_continuation(compiled_program, compiled_query, 4) == -1


def test_environment_size_at_entry_stops_at_tail_execute():
    compiled_program = init_compiled_program()
    compiled_program.code[:8] = [
        [OP.ALLOCATE, 0, 0, 0],
        [OP.PUT_Y_VAR, 0, 0, 0],
        [OP.DEALLOCATE, 0, 0, 0],
        [OP.EXECUTE, 0, 0, 0],
        [OP.ALLOCATE, 0, 0, 0],
        [OP.PUT_Y_VAR, 1, 0, 0],
        [OP.DEALLOCATE, 0, 0, 0],
        [OP.PROCEED, 0, 0, 0],
    ]
    compiled_program.code_size[0] = 8

    assert (
        environment_size_at_entry(
            compiled_program,
            init_compiled_query(),
            0,
        )
        == 1
    )


def test_call_does_not_snapshot_or_restore_x_registers():
    code = np.array(
        [
            [OP.ALLOCATE, 0, 0, 0],
            [OP.PUT_Y_VAR, 0, 0, 0],
            [OP.CALL, 0, 0, 1],
            [OP.PUT_Y_VAL, 0, 1, 0],
            [OP.DEALLOCATE, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
            [OP.PUT_CONST, 123, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    entry = init_compiled_program().predicate_entry
    entry[0, 0] = 6

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        entry,
        memory,
    )

    assert (
        emulate(
            machine,
            compiled_program,
            compiled_query,
        )
        == WAMStatus.SUCCESS
    )

    assert memory.heap.cells[memory.X[0]] == 123
    assert memory.X[1] != memory.X[0]


def test_cut_removes_newer_choicepoints_before_fail():
    code = np.array(
        [
            [OP.ALLOCATE, 0, 0, 0],
            [OP.GET_LEVEL, 0, 0, 0],
            [OP.TRY_ME_ELSE, 5, 0, 0],
            [OP.CUT, 0, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST_ME_ELSE_FAIL, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        init_compiled_program().predicate_entry,
        memory,
    )
    state = machine.registers

    status = emulate(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.EXHAUSTED
    assert state.B[0] == -1


def test_choicepoint_restore_restores_b0_from_failed_branch():
    code = np.array(
        [
            [OP.ALLOCATE, 0, 0, 0],
            [OP.TRY_ME_ELSE, 4, 0, 0],
            [OP.GET_LEVEL, 0, 0, 0],
            [OP.FAIL, 0, 0, 0],
            [OP.TRUST_ME_ELSE_FAIL, 0, 0, 0],
            [OP.DEALLOCATE, 0, 0, 0],
            [OP.PROCEED, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    memory = init_machine()
    machine, compiled_program, compiled_query = _init_test_machine(
        code,
        init_compiled_program().predicate_entry,
        memory,
    )
    state = machine.registers

    status = emulate(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.SUCCESS
    assert state.B[0] == -1


def test_depth_cutoff_inside_naf_does_not_make_negation_succeed():
    program_source = """
    deep :- leaf.
    leaf.
    candidate(a) :- deep.
    candidate(b).
    bad :- \\+ candidate(a).
    """

    symbol_table = empty_symbol_table()
    program, symbol_table = parse(program_source, symbol_table)

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()

    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
    )

    try_rows = compiled_program.code[: compiled_program.code_size[0]][
        compiled_program.code[: compiled_program.code_size[0], 0] == OP.TRY_ME_ELSE
    ]
    assert ChoicePointKind.NORMAL in try_rows[:, 2]
    assert ChoicePointKind.NAF in try_rows[:, 2]

    query, symbol_table = parse("?- bad.", symbol_table)
    compiled_query = init_compiled_query()

    compile_query(
        query,
        compiled_query,
        compiled_program,
        compiler_state,
    )

    machine = init_machine()
    machine.depth_limit[0] = 2

    status = run(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.DEPTH_LIMIT
    candidate_choice = machine.registers.B[0]
    assert machine.stack.cells[candidate_choice + ChoiceSlot.KIND] == ChoicePointKind.NORMAL
    naf_choice = machine.stack.cells[candidate_choice + ChoiceSlot.PREVIOUS_B]
    assert machine.stack.cells[naf_choice + ChoiceSlot.KIND] == ChoicePointKind.NAF_TRUNCATED

    # Exhaust everything still searchable at this depth.
    # A cutoff must never turn `bad` into a successful answer.
    while status == WAMStatus.DEPTH_LIMIT:
        status = redo(
            machine,
            compiled_program,
            compiled_query,
        )
        assert status != WAMStatus.SUCCESS

    assert status == WAMStatus.EXHAUSTED


def test_deeper_search_proves_negated_goal_and_bad_still_fails():
    program_source = """
    deep :- leaf.
    leaf.
    candidate(a) :- deep.
    candidate(b).
    bad :- \\+ candidate(a).
    """

    symbol_table = empty_symbol_table()
    program, symbol_table = parse(program_source, symbol_table)

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()

    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
    )

    positive_query_program, symbol_table = parse("?- candidate(a).", symbol_table)
    positive_query = init_compiled_query()
    compile_query(
        positive_query_program,
        positive_query,
        compiled_program,
        compiler_state,
    )
    positive_machine = init_machine()
    positive_machine.depth_limit[0] = 4
    assert run(positive_machine, compiled_program, positive_query) == WAMStatus.SUCCESS

    query, symbol_table = parse("?- bad.", symbol_table)
    compiled_query = init_compiled_query()

    compile_query(
        query,
        compiled_query,
        compiled_program,
        compiler_state,
    )

    machine = init_machine()
    machine.depth_limit[0] = 4

    status = run(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.EXHAUSTED


def test_depth_cutoff_propagates_through_nested_naf():
    program_source = r"""
    deep :- leaf.
    leaf.
    candidate(a) :- deep.
    candidate(b).
    bad :- \+ \+ candidate(a).
    """

    symbol_table = empty_symbol_table()
    program, symbol_table = parse(program_source, symbol_table)
    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program()
    compile_program(program, compiled_program, compiler_state)

    query, symbol_table = parse("?- bad.", symbol_table)
    compiled_query = init_compiled_query()
    compile_query(query, compiled_query, compiled_program, compiler_state)

    machine = init_machine()
    machine.depth_limit[0] = 2
    status = run(machine, compiled_program, compiled_query)

    cutoff_count = 0
    while status == WAMStatus.DEPTH_LIMIT:
        cutoff_count += 1
        status = redo(
            machine,
            compiled_program,
            compiled_query,
        )
        assert status != WAMStatus.SUCCESS

    assert cutoff_count == 3
    assert status == WAMStatus.EXHAUSTED
