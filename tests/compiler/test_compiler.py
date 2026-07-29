import pytest

import numpy as np

from wampy.config import DEFAULT_CONFIG, load_config_toml
from wampy.frontend.parser import build_new_program, build_query
from wampy.compiler.compiler import (
    OP,
    ENTRY_INVALID_PC,
    init_compiler,
    compile,
    reset_compiler,
)
from wampy.utils.debug import compiled_global_opcode_trace, compiled_predicate_opcode_trace
from wampy.frontend.ast.program import init_program, add_child_node
from wampy.frontend.ast.symbol_id import SymID

CONFIG_TOML = """
[ast]
num_terms = 30
max_terms_nodes = 100
max_terms_nodes_childs = 5
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _assert_no_predicate_entries(entry: np.ndarray):
    assert np.all(entry == ENTRY_INVALID_PC)


def test_init_compiler():
    code_area = init_compiler()

    assert code_area.pc == 0
    assert code_area.code.ndim == 2
    assert code_area.code.shape[0] == DEFAULT_CONFIG.limits.max_instr
    assert code_area.code.shape[1] == 4
    assert code_area.entry.shape == (
        DEFAULT_CONFIG.entry.max_functors,
        DEFAULT_CONFIG.entry.max_arity + 1,
    )


def test_init_compiler_uses_configured_max_instr():
    config = load_config_toml("""
        [limits]
        max_instr = 128
        """)
    code_area = init_compiler(config)

    assert code_area.code.shape == (config.limits.max_instr, 4)
    assert code_area.clause_entry_term.shape[0] == config.limits.max_instr


def test_fact_no_args(wam_config):
    program, binding = build_new_program(
        """
        p.
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    assert compiled_global_opcode_trace(compiled_program) == [OP.PROCEED]


def test_fact_constant_argument(wam_config):
    program, binding = build_new_program(
        """
        p(a).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    assert compiled_global_opcode_trace(compiled_program) == [
        OP.GET_CONST,
        OP.PROCEED,
    ]


def test_fact_variable_argument(wam_config):
    program, binding = build_new_program(
        """
        p(X).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    assert compiled_global_opcode_trace(compiled_program) == [
        OP.GET_VAR,
        OP.PROCEED,
    ]


def test_simple_rule_call(wam_config):
    program, binding = build_new_program(
        """
        q(X).
        p(X) :- q(X).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    p_code = compiled_program.code[pc : compiled_program.pc.value]
    assert [OP(instr[0]) for instr in p_code[:3]] == [
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
    ]


def test_rule_with_constant_call(wam_config):
    program, binding = build_new_program(
        """
        q(a).
        p :- q(a).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 0]

    assert [OP(instr[0]) for instr in compiled_program.code[pc : pc + 3]] == [
        OP.PUT_CONST,
        OP.EXECUTE,
        OP.PROCEED,
    ]


def test_multiple_goals_allocate(wam_config):
    program, binding = build_new_program(
        """
        q(X).
        r(X).
        p(X) :- q(X), r(X).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert OP.GET_VAR in seq
    assert OP.CALL in seq
    assert OP.EXECUTE in seq


def test_tail_call_optimization(wam_config):
    program, binding = build_new_program(
        """
        q(X).
        p(X) :- q(X).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    assert [OP(instr[0]) for instr in compiled_program.code[pc : pc + 3]] == [
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
    ]


def test_multiple_clauses_choicepoints(wam_config):
    program, binding = build_new_program(
        """
        p(a).
        p(b).
        p(c).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert seq[0] == OP.TRY
    assert OP.RETRY in seq
    assert OP.TRUST in seq


def test_variable_then_constant_clause(wam_config):
    program, binding = build_new_program(
        """
        p(X).
        p(a).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert seq[0] == OP.TRY
    assert OP.GET_VAR in seq
    assert OP.TRUST in seq
    assert OP.GET_CONST in seq


def test_entry_points_created(wam_config):

    program, binding = build_new_program(
        """
        p(a).
        q(b).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled_program = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")

    assert compiled_program.entry[p_fun, 1] != ENTRY_INVALID_PC
    assert compiled_program.entry[q_fun, 1] != ENTRY_INVALID_PC


def test_fact_repeated_variable_head(wam_config):
    """Repeated variable in head"""
    program, binding = build_new_program(
        """
        p(X, X).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    assert compiled_global_opcode_trace(compiled_program) == [
        OP.GET_VAR,
        OP.GET_VAL,
        OP.PROCEED,
    ]


def test_fact_variable_and_constant(wam_config):
    """Mixed variable / constant in head"""
    program, binding = build_new_program(
        """
        p(X, a).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    assert compiled_global_opcode_trace(compiled_program) == [
        OP.GET_VAR,
        OP.GET_CONST,
        OP.PROCEED,
    ]


def test_fact_two_constants(wam_config):
    """Two constant arguments"""
    program, binding = build_new_program(
        """
        p(a, b).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    assert compiled_global_opcode_trace(compiled_program) == [
        OP.GET_CONST,
        OP.GET_CONST,
        OP.PROCEED,
    ]


def test_rule_two_variables(wam_config):
    """Two-variable rule, single goal"""
    program, binding = build_new_program(
        """
        q(X, Y).
        p(X, Y) :- q(X, Y).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 2]

    p_code = compiled_program.code[pc : compiled_program.pc.value]
    trace = [OP(instr[0]) for instr in p_code]

    assert trace.count(OP.GET_VAR) == 2
    assert OP.EXECUTE in trace


def test_variable_introduced_in_body(wam_config):
    """Variable introduced in body"""
    program, binding = build_new_program(
        """
        q(Y).
        r(X, Y).
        p(X) :- q(Y), r(X, Y).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert OP.PUT_VAR in seq
    assert OP.CALL in seq
    assert OP.EXECUTE in seq


def test_three_goal_conjunction(wam_config):
    """Three-goal conjunction"""
    program, binding = build_new_program(
        """
        q(X).
        r(X).
        s(X).
        p(X) :- q(X), r(X), s(X).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert seq.count(OP.CALL) == 2
    assert OP.EXECUTE in seq


def test_fact_and_rule_same_functor(wam_config):
    """Fact and rule with same functor"""
    program, binding = build_new_program(
        """
        p(a).
        p(X) :- q(X).
        q(b).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert OP.TRY in seq
    assert OP.TRUST in seq


def test_rule_then_fact_ordering(wam_config):
    """Rule before fact ordering"""
    program, binding = build_new_program(
        """
        p(X) :- q(X).
        p(a).
        q(a).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = compiled_program.entry[p_fun, 1]

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]

    assert seq[0] == OP.TRY
    assert OP.GET_VAR in seq
    assert OP.GET_CONST in seq


def test_same_functor_different_arity(wam_config):
    """Same functor, different arity"""
    program, binding = build_new_program(
        """
        p.
        p(a).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert compiled_program.entry[p_fun, 0] != ENTRY_INVALID_PC
    assert compiled_program.entry[p_fun, 1] != ENTRY_INVALID_PC


def test_unused_predicate_compiled(wam_config):
    """Unused predicate still compiled"""
    program, binding = build_new_program(
        """
        unused(a).
        p :- q.
        q.
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    unused_fun = next(k for k, v in binding.items() if v == "unused")
    assert compiled_program.entry[unused_fun, 1] != ENTRY_INVALID_PC


def test_grandparent_tail_call_and_no_allocate(wam_config):
    program, binding = build_new_program(
        """
        parent(alice, bob).
        parent(bob, claire).
        parent(bob, hans).
        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)
    code = compiled.code
    entry = compiled.entry

    grandparent_id = next(k for k, v in binding.items() if v == "grandparent")
    pc = entry[grandparent_id, 2]

    ops = []
    i = pc
    while i < code.shape[0]:
        op = OP(code[i, 0])
        ops.append(op)
        if op == OP.EXECUTE:
            break
        i += 1

    # tail-call optimization
    assert ops[-1] == OP.EXECUTE

    # optimized mode: no environment
    assert OP.ALLOCATE not in ops
    assert OP.DEALLOCATE not in ops


def test_reset_compiler_reuses_buffers_and_clears_entry(wam_config):
    program, binding = build_new_program(
        """
        parent(alice, bob).
        parent(bob, claire).
        parent(bob, hans).
        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """,
        wam_config,
    )

    code_area = init_compiler()
    code_buf = code_area.code
    pc_buf = code_area.pc.storage
    entry_buf = code_area.entry

    compiled1 = compile(program, code_area)
    assert compiled1.pc.storage is pc_buf

    # reset and recompile
    code_area = reset_compiler(code_area)

    assert code_area.code is code_buf
    assert code_area.pc.storage is pc_buf
    assert code_area.entry is entry_buf
    assert code_area.pc.value == 0
    _assert_no_predicate_entries(code_area.entry)

    compiled2 = compile(program, code_area)
    entry2 = compiled2.entry
    assert np.any(entry2 != ENTRY_INVALID_PC)


def test_variable_reused_across_multiple_goals(wam_config):
    """
    Same variable appears in multiple goals.
    Ensures PUT_VAL is used consistently.
    """
    program, binding = build_new_program(
        """
        q(X).
        r(X).
        p(X) :- q(X), r(X).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    trace = compiled_predicate_opcode_trace(compiled, p_fun, 1)

    assert trace.count(OP.GET_VAR) == 1
    assert trace.count(OP.PUT_VAL) >= 1
    assert OP.EXECUTE in trace


def test_non_consecutive_variable_reuse(wam_config):
    """
    Variable reused after another variable is introduced.
    Tests variable table stability.
    """
    program, binding = build_new_program(
        """
        q(X).
        r(Y).
        s(X, Y).
        p(X) :- q(X), r(Y), s(X, Y).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    trace = compiled_predicate_opcode_trace(compiled, p_fun, 1)

    assert OP.PUT_VAL in trace
    assert OP.PUT_VAR in trace
    assert OP.EXECUTE in trace


def test_constant_then_variable_in_body(wam_config):
    """
    Constant call followed by variable use.
    Ensures constants do not disturb var table.
    """
    program, binding = build_new_program(
        """
        q(a).
        r(X).
        p(X) :- q(a), r(X).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    trace = compiled_predicate_opcode_trace(compiled, p_fun, 1)

    assert OP.PUT_CONST in trace
    assert OP.PUT_VAL in trace or OP.PUT_VAR in trace
    assert OP.EXECUTE in trace


def test_variable_table_resets_per_clause(wam_config):
    """
    Same functor, different clauses.
    Variable tables must not leak across clauses.
    """
    program, binding = build_new_program(
        """
        p(X, Y).
        p(Y, X).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    trace = compiled_predicate_opcode_trace(compiled, p_fun, 2)

    # Each clause should start with GET_VAR
    assert trace.count(OP.GET_VAR) >= 4
    assert OP.TRY in trace
    assert OP.TRUST in trace


def test_interleaved_predicates_choicepoint_patching(wam_config):
    """
    Interleaved predicates should not break TRY/RETRY/TRUST patching.
    """
    program, binding = build_new_program(
        """
        p(a).
        q(a).
        p(b).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    trace = compiled_predicate_opcode_trace(compiled, p_fun, 1)

    assert OP.TRY in trace
    assert OP.TRUST in trace


def test_interleaved_p_q_p_try_alt_stays_on_p_chain(wam_config):
    program, binding = build_new_program(
        """
        p(a).
        q(a).
        p(b).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")

    pc_p = int(compiled.entry[p_fun, 1])
    pc_q = int(compiled.entry[q_fun, 1])

    assert OP(compiled.code[pc_p, 0]) == OP.TRY

    alt_pc = int(compiled.code[pc_p, 1])

    # Must not jump into q/1 entry
    assert alt_pc != pc_q

    # For exactly 2 p-clauses, next block should be TRUST (not foreign clause code)
    assert OP(compiled.code[alt_pc, 0]) == OP.TRUST

    # TRY/RETRY/TRUST wrapper PCs are not clause-entry PCs
    assert int(code_area.clause_entry_term[alt_pc]) == -1


def test_interleaved_p_q_p_r_p_choice_chain_stays_on_p_only(wam_config):
    program, binding = build_new_program(
        """
        p(a).
        q(a).
        p(b).
        r(a).
        p(c).
        """,
        wam_config,
    )

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")
    r_fun = next(k for k, v in binding.items() if v == "r")

    pc_p = int(compiled.entry[p_fun, 1])
    foreign_entries = {
        int(compiled.entry[q_fun, 1]),
        int(compiled.entry[r_fun, 1]),
    }

    assert OP(compiled.code[pc_p, 0]) == OP.TRY

    alt1 = int(compiled.code[pc_p, 1])
    assert alt1 not in foreign_entries
    assert OP(compiled.code[alt1, 0]) == OP.RETRY

    alt2 = int(compiled.code[alt1, 1])
    assert alt2 not in foreign_entries
    assert OP(compiled.code[alt2, 0]) == OP.TRUST


def test_empty_program_compiles_safely(wam_config):
    """
    Empty program should compile without crashing.
    """
    program, binding = build_new_program("", wam_config)

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)

    assert compiled.pc.value == 0
    _assert_no_predicate_entries(compiled.entry)


def test_p_s_a_should_compile_as_structure_unification(wam_config):
    from wampy.frontend import display_as_tree
    from wampy.utils.debug import debug_compiled_program

    program, binding = build_new_program("p(s(a)).", wam_config)
    display_as_tree(program, binding, 0)
    display_as_tree(program, binding, 1)
    display_as_tree(program, binding, 2)
    display_as_tree(program, binding, 3)

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)
    debug_compiled_program(compiled, binding)

    assert compiled_global_opcode_trace(compiled) == [
        OP.GET_STR,
        OP.UNIFY_CONST,
        OP.END_STR,
        OP.PROCEED,
    ]
    assert compiled.pc.value == 4


def test_positional_body_true_compiles_as_fact(wam_config):
    # Ensures: "p :- true." compiles p/0 (body TRUE is not mis-read as head)
    program, binding = build_new_program(
        """
        p :- true.
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled = compile(program, code_area)

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert compiled.entry[p_fun, 0] != ENTRY_INVALID_PC
    assert compiled_global_opcode_trace(compiled) == [OP.PROCEED]


def test_true_head_is_ignored_as_directive(wam_config):
    # Ensures: "true :- p(a)." does NOT define p/1 and emits no code.
    program, binding = build_new_program(
        """
        true :- p(a).
        """,
        wam_config,
    )

    code_area = init_compiler()
    compiled = compile(program, code_area)

    # p and a should exist as bindings, but p/1 must not be compiled as a predicate
    p_fun = next(k for k, v in binding.items() if v == "p")
    assert compiled.entry[p_fun, 1] == ENTRY_INVALID_PC
    assert compiled_global_opcode_trace(compiled) == []


def _mk_clause_root(program, tid: int):
    program.symbol[tid, 0] = np.uint16(SymID.CLAUSE)


def _mk_fact(program, tid: int, functor_sym: np.uint16, args_syms=()):
    """
    Build: functor(args...).
    Clause root has only Head child (no Body child) -> fact.
    """
    _mk_clause_root(program, tid)
    head_id, ok = add_child_node(program, tid, 0, functor_sym)
    assert ok
    for s in args_syms:
        _, ok2 = add_child_node(program, tid, head_id, np.uint16(s))
        assert ok2


def _mk_rule_with_head_and(program, tid: int, head_terms, body_term=None):
    """
    Build: (H1 , H2 , ...) :- Body.
    Root children order MUST be: [HEAD, BODY]
    """
    _mk_clause_root(program, tid)

    # 1) head FIRST
    assert len(head_terms) >= 2
    head_and_id, ok = add_child_node(program, tid, 0, np.uint16(SymID.AND))
    assert ok

    def attach_term(parent_and_id, term):
        f, args = term
        t_id, ok3 = add_child_node(program, tid, parent_and_id, np.uint16(f))
        assert ok3
        for a in args:
            _, ok4 = add_child_node(program, tid, t_id, np.uint16(a))
            assert ok4
        return t_id

    # AND(H1, AND(H2, AND(...)))
    attach_term(head_and_id, head_terms[0])

    cur_and = head_and_id
    for term in head_terms[1:-1]:
        next_and, ok5 = add_child_node(program, tid, cur_and, np.uint16(SymID.AND))
        assert ok5
        attach_term(next_and, term)
        cur_and = next_and

    attach_term(cur_and, head_terms[-1])

    # 2) body SECOND (optional)
    if body_term is not None:
        body_fun, body_args = body_term
        # If body is TRUE you can omit it entirely; but it's also ok to store it.
        body_id, okb = add_child_node(program, tid, 0, np.uint16(body_fun))
        assert okb
        for a in body_args:
            _, ok2 = add_child_node(program, tid, body_id, np.uint16(a))
            assert ok2


def test_head_conjunction_splits_into_multiple_facts(wam_config):
    # Pick small non-GID functor symbols. In your system, bindings are small ints;
    # GID values are distinct, so these are safe.
    p_fun = np.uint16(1)
    q_fun = np.uint16(2)

    program = init_program(wam_config)

    # One term encodes: (p, q) :- true.
    # Body is TRUE => represent body as a TRUE goal node.
    # We still need a body node positionally, so use body functor TRUE with no args.
    _mk_rule_with_head_and(
        program,
        tid=0,
        head_terms=[(p_fun, ()), (q_fun, ())],
        body_term=(np.uint16(SymID.TRUE), ()),
    )

    code_area = init_compiler()
    compiled = compile(program, code_area)

    # Expected new behavior: compiler splits AND-head into separate clauses:
    #   p.
    #   q.
    assert compiled.entry[p_fun, 0] != ENTRY_INVALID_PC
    assert compiled.entry[q_fun, 0] != ENTRY_INVALID_PC

    # each is a single fact => PROCEED, PROCEED (order not important here)
    assert sorted(compiled_global_opcode_trace(compiled), key=int) == sorted([OP.PROCEED, OP.PROCEED], key=int)


def test_head_conjunction_splits_and_shares_body(wam_config):
    p_fun = np.uint16(1)
    q_fun = np.uint16(2)
    r_fun = np.uint16(3)

    X = np.uint16(SymID.VAR_FIRST)  # shared variable symbol

    program = init_program(wam_config)

    # r(X).
    _mk_fact(program, tid=0, functor_sym=r_fun, args_syms=(X,))

    # (p(X), q(X)) :- r(X).
    _mk_rule_with_head_and(
        program,
        tid=1,
        head_terms=[(p_fun, (X,)), (q_fun, (X,))],
        body_term=(r_fun, (X,)),
    )

    code_area = init_compiler()
    compiled = compile(program, code_area)

    pc_p = int(compiled.entry[p_fun, 1])
    pc_q = int(compiled.entry[q_fun, 1])

    assert pc_p != ENTRY_INVALID_PC
    assert pc_q != ENTRY_INVALID_PC

    # p(X) :- r(X).
    p_ops = [OP(instr[0]) for instr in compiled.code[pc_p : pc_p + 4]]
    assert p_ops == [OP.GET_VAR, OP.PUT_VAL, OP.EXECUTE, OP.PROCEED]

    # q(X) :- r(X).
    q_ops = [OP(instr[0]) for instr in compiled.code[pc_q : pc_q + 4]]
    assert q_ops == [OP.GET_VAR, OP.PUT_VAL, OP.EXECUTE, OP.PROCEED]


def test_query_true_has_no_predicate_entry_point(wam_config):
    program, symbol_table = build_new_program("", wam_config)
    query = build_query("true.", symbol_table, wam_config)

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    assert compiled_program.entry[query.fun, query.arity] == ENTRY_INVALID_PC


def test_query_not_true_has_no_predicate_entry_point(wam_config):
    program, symbol_table = build_new_program("", wam_config)
    query = build_query("\\+ true.", symbol_table, wam_config)

    code_area = init_compiler(wam_config)
    compiled_program = compile(program, code_area, wam_config)

    assert compiled_program.entry[query.fun, query.arity] == ENTRY_INVALID_PC


def test_negation_lowers_to_call_cut_fail(wam_config):
    program, _ = build_new_program(r"p(a). ok :- \+ p(a).", wam_config)

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)
    ops = compiled_global_opcode_trace(compiled)

    assert OP.MARK_CUT in ops
    assert OP.CALL in ops
    assert OP.CUT in ops
    assert OP.FAIL in ops
    assert OP.DROP_CUT in ops
    assert "NOT_CALL" not in {op.name for op in ops}


def test_nested_negation_lowers_to_nested_cut_fail_blocks(wam_config):
    program, _ = build_new_program(r"p(a). ok :- \+ (\+ p(a)).", wam_config)

    code_area = init_compiler(wam_config)
    compiled = compile(program, code_area, wam_config)
    ops = compiled_global_opcode_trace(compiled)

    assert ops.count(OP.MARK_CUT) == 2
    assert ops.count(OP.CUT) == 2
    assert ops.count(OP.FAIL) == 2
    assert ops.count(OP.DROP_CUT) == 2
