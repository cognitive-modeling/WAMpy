import numpy as np
import pytest
from wampy import Prolog
from wampy.compiler import CompilerState
from wampy.compiler.codegen.clause import (
    analyze_clause,
    init_clause_analysis_scratch,
)
from wampy.compiler.compiled_program import (
    CompiledProgram,
    clear_compiled_program,
    init_compiled_program,
)
from wampy.compiler.compiled_query import init_compiled_query
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import (
    ENTRY_INVALID_PC,
    clear_compiler_state,
    init_compiler_state,
    undo_hypothesis,
)
from wampy.compiler.diagnostics import (
    display_compiler_state,
    opcode_trace,
    predicate_opcode_trace,
)
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.compiler.query import compile_query
from wampy.config import DEFAULT_CONFIG, WAMConfig, load_config_toml
from wampy.frontend.ast.ops.analysis import (
    get_clause_head_body,
    get_node_capacity,
    get_query_arity,
    get_query_functor,
    get_user_symbol_id,
    get_variable_symbol_id,
)
from wampy.frontend.ast.ops.mutation import (
    add_child_node,
    allocate_term,
)
from wampy.frontend.ast.program import init_ast_program
from wampy.frontend.ast.symbol_id import (
    CoreSymID,
)
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table
from wampy.runtime.interpreter import exec_PUT_STR, exec_SET_CONST
from wampy.runtime.machine import init_machine
from wampy.status import WAMStatus


@pytest.fixture(scope="session")
def wam_config():
    return DEFAULT_CONFIG


def _assert_no_predicate_entries(entry: np.ndarray):
    assert np.all(entry == ENTRY_INVALID_PC)


def _symbol_id(binding, symbol):
    return next(int(k) for k, v in binding.items() if v == symbol)


def _entry_for_symbol(compiled, compiler_state, symbol_id, arity):
    predicate_slot = find_predicate_slot(compiler_state, int(symbol_id))
    if predicate_slot < 0:
        return ENTRY_INVALID_PC
    return compiled.predicate_entry[predicate_slot, arity]


def _predicate_slot_for_symbol(compiled, compiler_state, symbol_id):
    predicate_slot = find_predicate_slot(compiler_state, int(symbol_id))
    assert predicate_slot >= 0
    return predicate_slot


def _predicate_instructions(compiled, compiler_state, binding, symbol, arity):
    pc = int(_entry_for_symbol(compiled, compiler_state, _symbol_id(binding, symbol), arity))
    instructions = []
    while True:
        instruction = compiled.code[pc]
        op = OP(int(instruction[0]))
        instructions.append((op, int(instruction[1]), int(instruction[2]), int(instruction[3])))
        pc += 1
        if op == OP.PROCEED or op == OP.EXECUTE:
            return instructions


def _x_capacity_config(capacity) -> WAMConfig:
    return DEFAULT_CONFIG._replace(
        compiler=DEFAULT_CONFIG.compiler._replace(
            max_arity=capacity,
        ),
        runtime=DEFAULT_CONFIG.runtime._replace(
            max_x_registers=capacity,
        ),
    )


def test_x_register_allocation_succeeds_at_exact_capacity() -> None:
    config = _x_capacity_config(4)
    program, binding = parse("p(X, Y).", empty_symbol_table(config.frontend.ast), config)

    compiler_state = init_compiler_state(config)

    compiled = init_compiled_program(config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=config,
    )
    instructions = _predicate_instructions(compiled, compiler_state, binding, "p", 2)

    assert instructions[:2] == [
        (OP.GET_VAR, 2, 0, 0),
        (OP.GET_VAR, 3, 1, 0),
    ]


def test_x_register_allocation_rejects_first_register_beyond_capacity():
    config = _x_capacity_config(4)
    program, _ = parse("p(A, B, C, D).", empty_symbol_table(config.frontend.ast), config)

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="X register capacity exceeded"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_compile_rejects_predicate_arity_above_x_capacity() -> None:
    config = _x_capacity_config(4)
    program, _ = parse("p(a, b, c, d, e).", empty_symbol_table(config.frontend.ast), config)

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="Predicate arity exceeds X register capacity"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_nested_output_structure_overflow_is_compile_time_error() -> None:
    config = _x_capacity_config(2)
    program, _ = parse(
        "q(X). p :- q((((a, b), c), d)).",
        empty_symbol_table(config.frontend.ast),
        config,
    )

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="X register capacity exceeded"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_nested_input_structure_overflow_is_compile_time_error() -> None:
    config = _x_capacity_config(2)
    program, _ = parse("q((((a, b), c), d)).", empty_symbol_table(config.frontend.ast), config)

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="X register capacity exceeded"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_logical_variable_overflow_is_compile_time_error():
    config = _x_capacity_config(3)
    program, _ = parse("p(A, B).", empty_symbol_table(config.frontend.ast), config)

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="X register capacity exceeded"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_largest_body_call_reservation_is_checked_against_capacity():
    config = _x_capacity_config(3)
    program, _ = parse(
        "q(A, B, C). p(X) :- q(a, b, X).",
        empty_symbol_table(config.frontend.ast),
        config,
    )

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="X register capacity exceeded"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_init_compiler_state():
    compiler_state = init_compiler_state()

    assert isinstance(compiler_state, CompilerState)
    assert compiler_state.pc[0] == 0
    assert compiler_state.hypothesis_predicate_entry_undo.shape == (
        DEFAULT_CONFIG.compiler.max_hypothesis_predicate_changes,
        3,
    )
    assert compiler_state.hypothesis_predicate_entry_undo.dtype == np.int32
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 0
    assert not hasattr(compiler_state, "code")
    assert not hasattr(compiler_state, "code_size")
    assert not hasattr(compiler_state, "predicate_entry")


def test_clear_compiler_state_resets_undo_count_without_clearing_log():
    compiler_state = init_compiler_state()
    compiler_state.hypothesis_predicate_entry_undo[0] = (3, 2, 17)
    compiler_state.hypothesis_predicate_entry_undo_count[0] = 1

    clear_compiler_state(compiler_state)

    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 0
    assert tuple(compiler_state.hypothesis_predicate_entry_undo[0]) == (3, 2, 17)


def test_init_compiled_program():
    compiled_program = init_compiled_program()

    assert isinstance(compiled_program, CompiledProgram)
    assert compiled_program.code_size.shape == (1,)
    assert compiled_program.code_size[0] == 0
    assert compiled_program.code.shape == (
        DEFAULT_CONFIG.compiler.max_instructions,
        4,
    )
    assert compiled_program.predicate_entry.shape == (
        DEFAULT_CONFIG.compiler.max_predicates,
        DEFAULT_CONFIG.compiler.max_arity + 1,
    )


def test_clear_compiled_program_resets_metadata_without_clearing_code():
    compiled_program = init_compiled_program()
    compiled_program.code[0] = (OP.GET_CONST, 7, 0, 0)
    compiled_program.code_size[0] = 1
    compiled_program.predicate_entry[0, 1] = 0

    code = compiled_program.code
    predicate_entry = compiled_program.predicate_entry
    result = clear_compiled_program(compiled_program)

    assert result is compiled_program
    assert compiled_program.code is code
    assert compiled_program.predicate_entry is predicate_entry
    assert compiled_program.code_size[0] == 0
    assert tuple(compiled_program.code[0]) == (OP.GET_CONST, 7, 0, 0)
    assert np.all(compiled_program.predicate_entry == ENTRY_INVALID_PC)


def test_explicit_clear_reuses_buffers_for_an_unrelated_fresh_base(wam_config):
    symbol_table = empty_symbol_table(wam_config.frontend.ast)
    program_a, symbol_table = parse("p(a).", symbol_table, wam_config)
    program_b, symbol_table = parse("q.", symbol_table, wam_config)

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(program_a, compiled_program, compiler_state, wam_config)

    p_id = int(symbol_table.id_by_symbol["p"])
    p_slot = find_predicate_slot(compiler_state, p_id)
    assert compiled_program.predicate_entry[p_slot, 1] != ENTRY_INVALID_PC

    clear_compiler_state(compiler_state)
    compiler_state.pc_onto[0] = 0
    clear_compiled_program(compiled_program)
    compile_program(program_b, compiled_program, compiler_state, wam_config)

    q_id = int(symbol_table.id_by_symbol["q"])
    q_slot = find_predicate_slot(compiler_state, q_id)
    assert compiled_program.predicate_entry[q_slot, 0] != ENTRY_INVALID_PC
    assert compiled_program.predicate_entry[q_slot, 1] == ENTRY_INVALID_PC


def test_fact_no_args(wam_config):
    program, binding = parse(
        """
        p.
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiled_program = init_compiled_program(wam_config)
    compiler_state = init_compiler_state()
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (OP.PROCEED,)


def test_fact_constant_argument(wam_config):
    program, binding = parse(
        """
        p(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (
        OP.GET_CONST,
        OP.PROCEED,
    )


def test_fact_variable_argument(wam_config):
    program, binding = parse(
        """
        p(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (
        OP.GET_VAR,
        OP.PROCEED,
    )


def test_simple_rule_call(wam_config):
    program, binding = parse(
        """
        q(X).
        p(X) :- q(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    p_code = compiled_program.code[pc : compiled_program.code_size[0]]
    assert [OP(instr[0]) for instr in p_code[:3]] == [
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
    ]


def test_rule_with_constant_call(wam_config):
    program, binding = parse(
        """
        q(a).
        p :- q(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 0)

    assert [OP(instr[0]) for instr in compiled_program.code[pc : pc + 2]] == [
        OP.PUT_CONST,
        OP.EXECUTE,
    ]


def test_multiple_goals_allocate(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        q(X).
        r(X).
        p(X) :- q(X), r(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.code_size[0]]]

    assert OP.ALLOCATE in seq
    assert int(compiled_program.code[pc, 1]) == 0
    assert OP.GET_Y_VAR in seq
    assert OP.DEALLOCATE in seq
    assert OP.CALL in seq
    assert OP.EXECUTE in seq


def test_clause_liveness_distinguishes_temporary_x_from_permanent_y(wam_config):
    program, binding = parse(
        "q(a). r(a). p(X, Y) :- q(X), r(Y).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    q_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "q"),
    )
    r_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "r"),
    )

    assert _predicate_instructions(compiled, compiler_state, binding, "p", 2) == [
        (OP.ALLOCATE, 0, 0, 0),
        (OP.GET_VAR, 2, 0, 0),
        (OP.GET_Y_VAR, 0, 1, 0),
        (OP.PUT_VAL, 2, 0, 0),
        (OP.CALL, q_slot, 1, 1),
        (OP.PUT_Y_VAL, 0, 0, 0),
        (OP.DEALLOCATE, 0, 0, 0),
        (OP.EXECUTE, r_slot, 1, 0),
    ]


def test_clause_liveness_assigns_distinct_y_slots_without_reuse(wam_config):
    program, binding = parse(
        "q(a). r(a). s(a). t(a). p :- q(X), r(X), s(Y), t(Y).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    q_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "q"),
    )
    r_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "r"),
    )
    s_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "s"),
    )
    t_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "t"),
    )

    assert _predicate_instructions(compiled, compiler_state, binding, "p", 0) == [
        (OP.ALLOCATE, 0, 0, 0),
        (OP.PUT_Y_VAR, 0, 0, 0),
        (OP.CALL, q_slot, 1, 1),
        (OP.PUT_Y_VAL, 0, 0, 0),
        (OP.CALL, r_slot, 1, 0),
        (OP.PUT_Y_VAR, 1, 0, 0),
        (OP.CALL, s_slot, 1, 1),
        (OP.PUT_Y_VAL, 1, 0, 0),
        (OP.DEALLOCATE, 0, 0, 0),
        (OP.EXECUTE, t_slot, 1, 0),
    ]


def test_returning_call_without_permanent_variables_allocates_zero_y_slots(wam_config):
    program, binding = parse(
        "q(a). r(b). p :- q(a), r(b).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled, compiler_state, p_fun, 0)

    assert tuple(compiled.code[pc]) == (OP.ALLOCATE, 0, 0, 0)
    assert OP.DEALLOCATE in predicate_opcode_trace(compiled, compiler_state, p_fun, 0)


def test_tail_call_optimization(wam_config):
    program, binding = parse(
        """
        q(X).
        p(X) :- q(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    assert [OP(instr[0]) for instr in compiled_program.code[pc : pc + 3]] == [
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
    ]


def test_multiple_clauses_choicepoints(wam_config):
    program, binding = parse(
        """
        p(a).
        p(b).
        p(c).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.code_size[0]]]

    assert seq[0] == OP.TRY_ME_ELSE
    assert OP.RETRY_ME_ELSE in seq
    assert OP.TRUST_ME_ELSE_FAIL in seq


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "p(a).",
            (OP.GET_CONST, OP.PROCEED),
        ),
        (
            "p(a). p(b).",
            (
                OP.TRY_ME_ELSE,
                OP.GET_CONST,
                OP.PROCEED,
                OP.TRUST_ME_ELSE_FAIL,
                OP.GET_CONST,
                OP.PROCEED,
            ),
        ),
        (
            "p(a). p(b). p(c).",
            (
                OP.TRY_ME_ELSE,
                OP.GET_CONST,
                OP.PROCEED,
                OP.RETRY_ME_ELSE,
                OP.GET_CONST,
                OP.PROCEED,
                OP.TRUST_ME_ELSE_FAIL,
                OP.GET_CONST,
                OP.PROCEED,
            ),
        ),
    ],
)
def test_clause_prefixes_follow_remaining_clause_counts(wam_config, source, expected):
    program, binding = parse(
        source,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled_program, compiler_state, p_fun, 1) == expected


def test_variable_then_constant_clause(wam_config):
    program, binding = parse(
        """
        p(X).
        p(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.code_size[0]]]

    assert seq[0] == OP.TRY_ME_ELSE
    assert OP.GET_VAR in seq
    assert OP.TRUST_ME_ELSE_FAIL in seq
    assert OP.GET_CONST in seq


def test_entry_points_created(wam_config):
    program, binding = parse(
        """
        p(a).
        q(b).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")

    assert _entry_for_symbol(compiled_program, compiler_state, p_fun, 1) != ENTRY_INVALID_PC
    assert _entry_for_symbol(compiled_program, compiler_state, q_fun, 1) != ENTRY_INVALID_PC


def test_fact_repeated_variable_head(wam_config):
    """Repeated variable in head"""
    program, binding = parse(
        """
        p(X, X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (
        OP.GET_VAR,
        OP.GET_VAL,
        OP.PROCEED,
    )


def test_fact_variable_and_constant(wam_config):
    """Mixed variable / constant in head"""
    program, binding = parse(
        """
        p(X, a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (
        OP.GET_VAR,
        OP.GET_CONST,
        OP.PROCEED,
    )


def test_fact_two_constants(wam_config):
    """Two constant arguments"""
    program, binding = parse(
        """
        p(a, b).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert opcode_trace(compiled_program) == (
        OP.GET_CONST,
        OP.GET_CONST,
        OP.PROCEED,
    )


def test_rule_two_variables(wam_config: WAMConfig) -> None:
    """Two-variable rule, single goal"""
    program, binding = parse(
        """
        q(X, Y).
        p(X, Y) :- q(X, Y).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 2)

    p_code = compiled_program.code[pc : compiled_program.code_size[0]]
    trace = [OP(instr[0]) for instr in p_code]

    assert trace.count(OP.GET_VAR) == 2
    assert OP.EXECUTE in trace


def test_variable_introduced_in_body(wam_config: WAMConfig) -> None:
    """Variable introduced in body"""
    program, binding = parse(
        """
        q(Y).
        r(X, Y).
        p(X) :- q(Y), r(X, Y).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.code_size[0]]]

    assert OP.PUT_Y_VAR in seq
    assert OP.CALL in seq
    assert OP.EXECUTE in seq


def test_three_goal_conjunction(wam_config):
    """Three-goal conjunction"""
    program, binding = parse(
        """
        q(X).
        r(X).
        s(X).
        p(X) :- q(X), r(X), s(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    pc = _entry_for_symbol(compiled_program, compiler_state, p_fun, 1)

    seq = [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.code_size[0]]]

    assert seq.count(OP.CALL) == 2
    assert OP.EXECUTE in seq


def test_fact_and_rule_same_functor(wam_config):
    program, binding = parse(
        """
        p(a).
        p(X) :- q(X).
        q(b).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.TRY_ME_ELSE,
        OP.GET_CONST,
        OP.PROCEED,
        OP.TRUST_ME_ELSE_FAIL,
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
    )


def test_rule_then_fact_ordering(wam_config):
    program, binding = parse(
        """
        p(X) :- q(X).
        p(a).
        q(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.TRY_ME_ELSE,
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.EXECUTE,
        OP.TRUST_ME_ELSE_FAIL,
        OP.GET_CONST,
        OP.PROCEED,
    )


def test_same_functor_different_arity(wam_config):
    """Same functor, different arity"""
    program, binding = parse(
        """
        p.
        p(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert _entry_for_symbol(compiled_program, compiler_state, p_fun, 0) != ENTRY_INVALID_PC
    assert _entry_for_symbol(compiled_program, compiler_state, p_fun, 1) != ENTRY_INVALID_PC


def test_unused_predicate_compiled(wam_config: WAMConfig) -> None:
    """Unused predicate still compiled"""
    program, binding = parse(
        """
        unused(a).
        p :- q.
        q.
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    unused_fun = next(k for k, v in binding.items() if v == "unused")
    assert _entry_for_symbol(compiled_program, compiler_state, unused_fun, 1) != ENTRY_INVALID_PC


def test_grandparent_tail_call_deallocates_environment(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        parent(alice, bob).
        parent(bob, claire).
        parent(bob, hans).
        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    code = compiled.code

    grandparent_id = next(k for k, v in binding.items() if v == "grandparent")
    pc = _entry_for_symbol(compiled, compiler_state, grandparent_id, 2)

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

    assert ops[0] == OP.ALLOCATE
    assert OP.DEALLOCATE in ops
    assert ops.index(OP.DEALLOCATE) == len(ops) - 2


def test_explicit_state_and_program_clear_reuses_buffers(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        parent(alice, bob).
        parent(bob, claire).
        parent(bob, hans).
        grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled_program = init_compiled_program(wam_config)
    code_buf = compiled_program.code
    pc_buf = compiler_state.pc
    entry_buf = compiled_program.predicate_entry

    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )
    assert compiled_program.code is code_buf
    first_pc = compiled_program.code_size[0]
    assert compiler_state.pc_onto[0] == first_pc

    undo_hypothesis(compiler_state, compiled_program)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )
    assert compiler_state.pc_onto[0] == first_pc
    assert compiled_program.code_size[0] > first_pc

    # reset and recompile
    compiler_state.pc_onto[0] = 20
    compiler_state = clear_compiler_state(compiler_state, start_pc=0)

    assert compiler_state.pc is pc_buf
    assert compiled_program.code is code_buf
    assert compiled_program.predicate_entry is entry_buf
    assert compiler_state.pc[0] == 0
    assert compiler_state.pc_onto[0] == 20
    assert np.all(compiler_state.predicate_symbols_sorted == -1)
    assert np.all(compiler_state.predicate_slots_sorted == -1)
    assert np.all(compiler_state.symbol_by_predicate_slot == -1)

    compiler_state.pc_onto[0] = 0
    clear_compiled_program(compiled_program)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )
    entry2 = compiled_program.predicate_entry
    assert np.any(entry2 != ENTRY_INVALID_PC)


def test_clear_compiler_state_restarts_at_onto_boundary(wam_config):
    compiler_state = init_compiler_state(wam_config)
    compiler_state.pc_onto[0] = 100
    compiler_state.pc[0] = 116

    clear_compiler_state(compiler_state, compiler_state.pc_onto[0])

    assert compiler_state.pc[0] == 100
    assert compiler_state.pc_onto[0] == 100


def test_variable_reused_across_multiple_goals(wam_config):
    """
    Same variable appears in multiple goals.
    Ensures PUT_VAL is used consistently.
    """
    program, binding = parse(
        """
        q(X).
        r(X).
        p(X) :- q(X), r(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.ALLOCATE,
        OP.GET_Y_VAR,
        OP.PUT_Y_VAL,
        OP.CALL,
        OP.PUT_Y_VAL,
        OP.DEALLOCATE,
        OP.EXECUTE,
    )


def test_non_consecutive_variable_reuse(wam_config: WAMConfig) -> None:
    """
    Variable reused after another variable is introduced.
    Tests variable-table stability.
    """
    program, binding = parse(
        """
        q(X).
        r(Y).
        s(X, Y).
        p(X) :- q(X), r(Y), s(X, Y).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.ALLOCATE,
        OP.GET_Y_VAR,
        OP.PUT_Y_VAL,
        OP.CALL,
        OP.PUT_Y_VAR,
        OP.CALL,
        OP.PUT_Y_VAL,
        OP.PUT_Y_VAL,
        OP.DEALLOCATE,
        OP.EXECUTE,
    )


def test_constant_then_variable_in_body(wam_config: WAMConfig) -> None:
    """
    Constant call followed by variable use.
    Ensures constants do not disturb the variable table.
    """
    program, binding = parse(
        """
        q(a).
        r(X).
        p(X) :- q(a), r(X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.ALLOCATE,
        OP.GET_Y_VAR,
        OP.PUT_CONST,
        OP.CALL,
        OP.PUT_Y_VAL,
        OP.DEALLOCATE,
        OP.EXECUTE,
    )


def test_variable_table_resets_per_clause(wam_config):
    """
    Variable tables are reset before compiling each clause.
    """
    program, binding = parse(
        """
        p(X, Y).
        p(Y, X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 2) == (
        OP.TRY_ME_ELSE,
        OP.GET_VAR,
        OP.GET_VAR,
        OP.PROCEED,
        OP.TRUST_ME_ELSE_FAIL,
        OP.GET_VAR,
        OP.GET_VAR,
        OP.PROCEED,
    )


def test_interleaved_predicates_choicepoint_patching(wam_config: WAMConfig) -> None:
    """
    Predicate tracing follows the alternative chain and excludes
    instructions belonging to interleaved predicates.
    """
    program, binding = parse(
        """
        p(a).
        q(a).
        p(b).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert predicate_opcode_trace(compiled, compiler_state, p_fun, 1) == (
        OP.TRY_ME_ELSE,
        OP.GET_CONST,
        OP.PROCEED,
        OP.TRUST_ME_ELSE_FAIL,
        OP.GET_CONST,
        OP.PROCEED,
    )


def test_predicate_grouping_preserves_source_order_within_each_signature() -> None:
    prolog = Prolog(
        """
        p(a).
        q(x).
        p(b).
        q(y).
        p(c).
        p(a, b).
        """
    )

    assert prolog.query_all("p(X).") == [{"X": "a"}, {"X": "b"}, {"X": "c"}]
    assert prolog.query_all("q(X).") == [{"X": "x"}, {"X": "y"}]
    assert prolog.query_all("p(X, Y).") == [{"X": "a", "Y": "b"}]


def test_interleaved_p_q_p_try_alt_stays_on_p_chain(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        p(a).
        q(a).
        p(b).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")

    pc_p = int(_entry_for_symbol(compiled, compiler_state, p_fun, 1))
    pc_q = int(_entry_for_symbol(compiled, compiler_state, q_fun, 1))

    assert OP(compiled.code[pc_p, 0]) == OP.TRY_ME_ELSE

    alt_pc = int(compiled.code[pc_p, 1])

    # Must not jump into q/1 entry
    assert alt_pc != pc_q

    # For exactly 2 p-clauses, next block should be the trust instruction
    assert OP(compiled.code[alt_pc, 0]) == OP.TRUST_ME_ELSE_FAIL


def test_interleaved_p_q_p_r_p_choice_chain_stays_on_p_only(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        p(a).
        q(a).
        p(b).
        r(a).
        p(c).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")
    q_fun = next(k for k, v in binding.items() if v == "q")
    r_fun = next(k for k, v in binding.items() if v == "r")

    pc_p = int(_entry_for_symbol(compiled, compiler_state, p_fun, 1))
    foreign_entries = {
        int(_entry_for_symbol(compiled, compiler_state, q_fun, 1)),
        int(_entry_for_symbol(compiled, compiler_state, r_fun, 1)),
    }

    assert OP(compiled.code[pc_p, 0]) == OP.TRY_ME_ELSE

    alt1 = int(compiled.code[pc_p, 1])
    assert alt1 not in foreign_entries
    assert OP(compiled.code[alt1, 0]) == OP.RETRY_ME_ELSE

    alt2 = int(compiled.code[alt1, 1])
    assert alt2 not in foreign_entries
    assert OP(compiled.code[alt2, 0]) == OP.TRUST_ME_ELSE_FAIL


def test_empty_program_compiles_safely(wam_config):
    """
    Empty program should compile without crashing.
    """
    program, binding = parse("", empty_symbol_table(wam_config.frontend.ast), wam_config)

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert compiled.code_size[0] == 0
    _assert_no_predicate_entries(compiled.predicate_entry)


def test_p_s_a_should_compile_as_structure_unification(wam_config: WAMConfig) -> None:
    from wampy.frontend import display_as_tree

    program, binding = parse("p(s(a)).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    display_as_tree(program, binding, 0)
    display_as_tree(program, binding, 1)
    display_as_tree(program, binding, 2)
    display_as_tree(program, binding, 3)

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    display_compiler_state(compiled, compiler_state, binding)

    assert opcode_trace(compiled) == (
        OP.GET_STR,
        OP.UNIFY_CONST,
        OP.END_STR,
        OP.PROCEED,
    )
    assert compiled.code_size[0] == 4


def test_nested_output_structures_are_emitted_child_before_parent(wam_config: WAMConfig) -> None:
    program, binding = parse(
        """
        q(X).
        p :- q((((a, b), c), b)).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)

    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    and_id = int(CoreSymID.CONJUNCTION)
    a_id = _symbol_id(binding, "a")
    b_id = _symbol_id(binding, "b")
    c_id = _symbol_id(binding, "c")
    q_id = _symbol_id(binding, "q")
    q_slot = _predicate_slot_for_symbol(compiled, compiler_state, q_id)

    assert _predicate_instructions(
        compiled,
        compiler_state,
        binding,
        "p",
        0,
    ) == [
        # Build (a, b) into X1.
        (OP.PUT_STR, and_id, 2, 1),
        (OP.SET_CONST, a_id, 0, 0),
        (OP.SET_CONST, b_id, 0, 0),
        # Build ((a, b), c) into X2.
        (OP.PUT_STR, and_id, 2, 2),
        (OP.SET_VAL, 1, 0, 0),
        (OP.SET_CONST, c_id, 0, 0),
        # Build (((a, b), c), b) into X0.
        (OP.PUT_STR, and_id, 2, 0),
        (OP.SET_VAL, 2, 0, 0),
        (OP.SET_CONST, b_id, 0, 0),
        (OP.EXECUTE, q_slot, 1, 0),
    ]


def test_nested_output_structure_reuses_nested_variable_register(wam_config):
    program, binding = parse(
        """
        q(((X, Y), X)).
        p(X, Y) :- q(((X, Y), X)).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)

    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    q_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(binding, "q"),
    )

    assert _predicate_instructions(compiled, compiler_state, binding, "p", 2) == [
        (OP.GET_VAR, 2, 0, 0),
        (OP.GET_VAR, 3, 1, 0),
        # Build (X, Y) into X4.
        (OP.PUT_STR, CoreSymID.CONJUNCTION, 2, 4),
        (OP.SET_VAL, 2, 0, 0),
        (OP.SET_VAL, 3, 0, 0),
        # Build ((X, Y), X) into X0.
        (OP.PUT_STR, CoreSymID.CONJUNCTION, 2, 0),
        (OP.SET_VAL, 4, 0, 0),
        (OP.SET_VAL, 2, 0, 0),
        (OP.EXECUTE, q_slot, 1, 0),
    ]


def test_clause_reserves_registers_for_largest_body_call(wam_config):
    program, binding = parse(
        """
        q(A, B, C).
        p(X) :- q(a, b, X).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    q_slot = _predicate_slot_for_symbol(compiled, compiler_state, _symbol_id(binding, "q"))
    p_code = _predicate_instructions(compiled, compiler_state, binding, "p", 1)

    assert p_code[0] == (OP.GET_VAR, 3, 0, 0)
    assert p_code[1:4] == [
        (OP.PUT_CONST, _symbol_id(binding, "a"), 0, 0),
        (OP.PUT_CONST, _symbol_id(binding, "b"), 1, 0),
        (OP.PUT_VAL, 3, 2, 0),
    ]
    assert p_code[4] == (OP.EXECUTE, q_slot, 3, 0)


def test_clause_head_finishes_parent_before_nested_structure(wam_config):
    program, binding = parse(
        "q(((X, Y), X)).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert _predicate_instructions(compiled, compiler_state, binding, "q", 1) == [
        (OP.GET_STR, int(CoreSymID.CONJUNCTION), 2, 0),
        (OP.UNIFY_VAR, 1, 0, 0),
        (OP.UNIFY_VAR, 2, 0, 0),
        (OP.END_STR, 0, 0, 0),
        (OP.GET_STR, CoreSymID.CONJUNCTION, 2, 1),
        (OP.UNIFY_VAL, 2, 0, 0),
        (OP.UNIFY_VAR, 3, 0, 0),
        (OP.END_STR, 0, 0, 0),
        (OP.PROCEED, 0, 0, 0),
    ]


def test_nested_output_structure_depth_overflow_is_explicit():
    config = DEFAULT_CONFIG._replace(
        compiler=DEFAULT_CONFIG.compiler._replace(
            max_structure_depth=2,
        ),
    )
    program, _ = parse(
        "q(X). p :- q((((a, b), c), d)).",
        empty_symbol_table(config.frontend.ast),
        config,
    )

    compiled_program = init_compiled_program(config)
    with pytest.raises(ValueError, match="Structure nesting is too deep"):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(config),
            config=config,
        )


def test_nested_sibling_output_structures_use_separate_compound_temporaries(wam_config: WAMConfig):
    program, binding = parse(
        """
        q(X).
        p :- q(f((a, b), (c, d))).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    q_slot = _predicate_slot_for_symbol(compiled, compiler_state, _symbol_id(binding, "q"))
    p_code = _predicate_instructions(compiled, compiler_state, binding, "p", 0)
    a_id = _symbol_id(binding, "a")
    b_id = _symbol_id(binding, "b")
    c_id = _symbol_id(binding, "c")
    d_id = _symbol_id(binding, "d")

    assert p_code == [
        # Build (a, b) into X1.
        (OP.PUT_STR, int(CoreSymID.CONJUNCTION), 2, 1),
        (OP.SET_CONST, a_id, 0, 0),
        (OP.SET_CONST, b_id, 0, 0),
        # Build (c, d) into X2.
        (OP.PUT_STR, int(CoreSymID.CONJUNCTION), 2, 2),
        (OP.SET_CONST, c_id, 0, 0),
        (OP.SET_CONST, d_id, 0, 0),
        # Build f(X1, X2) into X0.
        (OP.PUT_STR, _symbol_id(binding, "f"), 2, 0),
        (OP.SET_VAL, 1, 0, 0),
        (OP.SET_VAL, 2, 0, 0),
        (OP.EXECUTE, q_slot, 1, 0),
    ]


def test_positional_body_true_compiles_as_fact(wam_config):
    # Ensures: "p :- true." compiles p/0 (body TRUE is not mis-read as head)
    program, binding = parse(
        """
        p :- true.
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_fun = next(k for k, v in binding.items() if v == "p")

    assert _entry_for_symbol(compiled, compiler_state, p_fun, 0) != ENTRY_INVALID_PC
    assert opcode_trace(compiled) == (OP.PROCEED,)


def test_true_head_is_ignored_as_directive(wam_config):
    # Ensures: "true :- p(a)." does NOT define p/1 and emits no code.
    program, binding = parse(
        """
        true :- p(a).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state()
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    # p and a should exist as bindings, but p/1 must not be compiled as a predicate
    p_fun = next(k for k, v in binding.items() if v == "p")
    assert _entry_for_symbol(compiled, compiler_state, p_fun, 1) == ENTRY_INVALID_PC
    assert opcode_trace(compiled) == ()


def _mk_clause_root(program, tid: int):
    assert allocate_term(program, CoreSymID.CLAUSE) == tid


def _mk_fact(program, tid: int, functor_sym: int, args_syms=()):
    """Build: functor(args...)."""
    _mk_clause_root(program, tid)

    head_id, ok = add_child_node(program, tid, 0, functor_sym)
    assert ok

    for symbol_id in args_syms:
        _, ok = add_child_node(program, tid, head_id, symbol_id)
        assert ok


def _mk_rule_with_head_and(program, tid: int, head_terms, body_term=None):
    """Build: (H1, H2, ...) :- Body."""
    _mk_clause_root(program, tid)

    assert len(head_terms) >= 2

    head_and_id, ok = add_child_node(program, tid, 0, CoreSymID.CONJUNCTION)
    assert ok

    def attach_term(parent_id, term):
        functor, args = term
        term_id, ok = add_child_node(program, tid, parent_id, functor)
        assert ok
        for symbol_id in args:
            _, ok = add_child_node(program, tid, term_id, symbol_id)
            assert ok
        return term_id

    # AND(H1, AND(H2, AND(...)))
    attach_term(head_and_id, head_terms[0])
    current_and = head_and_id
    for term in head_terms[1:-1]:
        next_and, ok = add_child_node(program, tid, current_and, CoreSymID.CONJUNCTION)
        assert ok
        attach_term(next_and, term)
        current_and = next_and
    attach_term(current_and, head_terms[-1])

    if body_term is not None:
        body_functor, body_args = body_term
        body_id, ok = add_child_node(program, tid, 0, body_functor)
        assert ok
        for symbol_id in body_args:
            _, ok = add_child_node(program, tid, body_id, symbol_id)
            assert ok


def test_head_conjunction_splits_into_multiple_facts(wam_config):
    program = init_ast_program(wam_config)

    p_fun = get_user_symbol_id(program, 0)
    q_fun = get_user_symbol_id(program, 1)

    _mk_rule_with_head_and(
        program,
        tid=0,
        head_terms=[
            (p_fun, ()),
            (q_fun, ()),
        ],
        body_term=(CoreSymID.TRUE, ()),
    )

    compiler_state = init_compiler_state()
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert _entry_for_symbol(compiled, compiler_state, p_fun, 0) != ENTRY_INVALID_PC
    assert _entry_for_symbol(compiled, compiler_state, q_fun, 0) != ENTRY_INVALID_PC

    assert sorted(opcode_trace(compiled), key=int) == sorted([OP.PROCEED, OP.PROCEED], key=int)


def test_head_conjunction_splits_and_shares_body(wam_config):
    program = init_ast_program(wam_config)

    X = get_variable_symbol_id(program, 0)

    p_fun = get_user_symbol_id(program, 0)
    q_fun = get_user_symbol_id(program, 1)
    r_fun = get_user_symbol_id(program, 2)

    _mk_fact(program, tid=0, functor_sym=r_fun, args_syms=(X,))

    _mk_rule_with_head_and(
        program,
        tid=1,
        head_terms=[
            (p_fun, (X,)),
            (q_fun, (X,)),
        ],
        body_term=(r_fun, (X,)),
    )

    compiler_state = init_compiler_state()
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    pc_p = int(_entry_for_symbol(compiled, compiler_state, p_fun, 1))
    pc_q = int(_entry_for_symbol(compiled, compiler_state, q_fun, 1))

    assert pc_p != ENTRY_INVALID_PC
    assert pc_q != ENTRY_INVALID_PC

    p_ops = [OP(instr[0]) for instr in compiled.code[pc_p : pc_p + 3]]
    assert p_ops == [OP.GET_VAR, OP.PUT_VAL, OP.EXECUTE]

    q_ops = [OP(instr[0]) for instr in compiled.code[pc_q : pc_q + 3]]
    assert q_ops == [OP.GET_VAR, OP.PUT_VAL, OP.EXECUTE]


def test_query_true_has_no_predicate_entry_point(wam_config):
    program, symbol_table = parse("", empty_symbol_table(wam_config.frontend.ast), wam_config)
    query, _ = parse("?- true.", symbol_table, wam_config)

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert (
        _entry_for_symbol(
            compiled_program,
            compiler_state,
            get_query_functor(query, 0),
            get_query_arity(query, 0),
        )
        == ENTRY_INVALID_PC
    )


def test_query_not_true_has_no_predicate_entry_point(wam_config):
    program, symbol_table = parse("", empty_symbol_table(wam_config.frontend.ast), wam_config)
    query, _ = parse(r"?- \+ true.", symbol_table, wam_config)

    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert (
        _entry_for_symbol(
            compiled_program,
            compiler_state,
            get_query_functor(query, 0),
            get_query_arity(query, 0),
        )
        == ENTRY_INVALID_PC
    )


def test_compile_negated_goal_reserves_target_registers_for_every_caller(wam_config):
    program, binding = parse(
        "p(X). a. b. c.", empty_symbol_table(wam_config.frontend.ast), wam_config
    )
    query, _ = parse(r"?- \+ p(((a, b), c)).", binding, wam_config)
    compiler_state = init_compiler_state(wam_config)
    compiled_program = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program,
        compiler_state,
        wam_config,
    )
    compiled_query = init_compiled_query(wam_config)
    compile_query(
        query,
        compiled_query,
        compiled_program,
        compiler_state,
        wam_config,
    )

    # p/1 owns X0 and the displayed negation operand owns X1. Nested
    # structure temporaries must therefore start above both registers.
    assert OP(compiled_query.code[0, 0]) == OP.ALLOCATE
    assert OP(compiled_query.code[1, 0]) == OP.PUT_STR
    assert int(compiled_query.code[1, 3]) == 1
    assert OP(compiled_query.code[compiled_query.code_size[0] - 2, 0]) == OP.DEALLOCATE


def test_negation_lowers_to_call_cut_fail(wam_config):
    program, _ = parse(
        r"p(a). ok :- \+ p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    ops = opcode_trace(compiled)

    assert OP.GET_LEVEL in ops
    assert OP.CALL in ops
    assert OP.CUT in ops
    assert OP.FAIL in ops
    assert OP.TRUST_ME_ELSE_FAIL in ops
    assert "NOT_CALL" not in {op.name for op in ops}


def test_grouped_negation_lowers_each_inner_goal(wam_config):
    program, _ = parse(
        r"q(a). r(a). ok :- \+ (q(a), r(a)).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)

    compile_program(program, compiled, compiler_state, wam_config)

    ops = opcode_trace(compiled)
    assert ops.count(OP.GET_LEVEL) == 1
    assert ops.count(OP.CALL) == 2
    assert ops.count(OP.CUT) == 1
    assert ops.count(OP.FAIL) == 1


def test_grouped_negation_rejects_control_cut(wam_config):
    program, _ = parse(
        r"q. ok :- \+ (q, !).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    with pytest.raises(ValueError, match="Cut inside grouped negation is not supported yet"):
        compile_program(
            program,
            init_compiled_program(wam_config),
            init_compiler_state(wam_config),
            wam_config,
        )


def test_clause_control_cuts_preserve_neck_and_non_neck_lowering(wam_config):
    program, binding = parse(
        "q(X). r(X). neck(X) :- !, q(X). non_neck(X) :- q(X), !, r(X).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)

    compile_program(program, compiled, compiler_state, wam_config)

    neck_ops = predicate_opcode_trace(
        compiled,
        compiler_state,
        _symbol_id(binding, "neck"),
        1,
    )
    non_neck_ops = predicate_opcode_trace(
        compiled,
        compiler_state,
        _symbol_id(binding, "non_neck"),
        1,
    )

    assert OP.NECK_CUT in neck_ops
    assert OP.GET_LEVEL not in neck_ops
    assert OP.GET_LEVEL in non_neck_ops
    assert OP.CUT in non_neck_ops


def test_clause_analysis_reuses_compile_local_scratch_buffers(wam_config):
    program, _ = parse(
        "p(a). q(X) :- r(X).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )
    scratch = init_clause_analysis_scratch(
        get_node_capacity(program),
        wam_config.compiler,
        wam_config.frontend.ast.max_variables_per_term,
    )
    compiler_state = init_compiler_state(wam_config)

    fact_head, fact_body = get_clause_head_body(program, 0)
    fact_analysis = analyze_clause(
        program,
        0,
        fact_head,
        fact_body,
        compiler_state,
        scratch,
        wam_config.compiler,
    )
    rule_head, rule_body = get_clause_head_body(program, 1)
    rule_analysis = analyze_clause(
        program,
        1,
        rule_head,
        rule_body,
        compiler_state,
        scratch,
        wam_config.compiler,
    )

    assert fact_analysis.goal_count == 0
    assert rule_analysis.goal_count == 1
    assert fact_analysis.var_table is scratch.var_table
    assert rule_analysis.var_table is scratch.var_table
    assert fact_analysis.goal_nodes is scratch.goal_nodes
    assert rule_analysis.call_environment_sizes is scratch.call_environment_sizes


def test_nested_negation_lowers_to_nested_cut_fail_blocks(wam_config):
    program, _ = parse(
        r"p(a). ok :- \+ (\+ p(a)).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )
    ops = opcode_trace(compiled)

    assert ops.count(OP.GET_LEVEL) == 2
    assert ops.count(OP.CUT) == 2
    assert ops.count(OP.FAIL) == 2
    assert ops.count(OP.TRUST_ME_ELSE_FAIL) == 2


def test_unknown_quoted_predicate_fails_during_query_compilation():
    prolog = Prolog("p(a).")

    with pytest.raises(
        RuntimeError,
        match="WAM error: UNDEFINED_PREDICATE",
    ):
        prolog.prepare("'2'.")


def test_output_structure_uses_set_var_then_set_val_for_repeated_fresh_variable(
    wam_config: WAMConfig,
) -> None:
    """Ensure a fresh structure variable is created once and then reused.

    This covers the classic WAM output-construction distinction between
    SET_VAR for the first occurrence of a temporary variable and SET_VAL for
    later occurrences of that same variable. Without this invariant, f(Y, Y)
    could incorrectly be compiled as two independent fresh variables.
    """
    program, symbol_table = parse(
        """
        q(f(X, X)).
        p :- q(f(Y, Y)).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    q_slot = _predicate_slot_for_symbol(
        compiled,
        compiler_state,
        _symbol_id(symbol_table, "q"),
    )
    f_id = _symbol_id(symbol_table, "f")

    p_code = _predicate_instructions(
        compiled,
        compiler_state,
        symbol_table,
        "p",
        0,
    )

    assert p_code[0] == (OP.PUT_STR, f_id, 2, 0)
    assert p_code[1][0] == OP.SET_VAR

    variable_register = p_code[1][1]

    assert p_code[2] == (OP.SET_VAL, variable_register, 0, 0)
    assert p_code[3] == (OP.EXECUTE, q_slot, 1, 0)


def test_output_structures_use_set_y_var_then_set_y_val_across_calls(
    wam_config: WAMConfig,
) -> None:
    """Ensure structure construction preserves variables that survive a call.

    X occurs in both q(f(X)) and r(g(X)), so it must live in a permanent Y
    slot. The first structure must introduce it with SET_Y_VAR and the second
    structure must reuse the same Y slot with SET_Y_VAL. This covers the
    interaction between SET_* construction and clause liveness/environment
    allocation.
    """
    program, symbol_table = parse(
        """
        q(f(a)).
        r(g(a)).
        p :- q(f(X)), r(g(X)).
        """,
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    compiler_state = init_compiler_state(wam_config)
    compiled = init_compiled_program(wam_config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    p_code = _predicate_instructions(
        compiled,
        compiler_state,
        symbol_table,
        "p",
        0,
    )

    set_y_var = next(instruction for instruction in p_code if instruction[0] == OP.SET_Y_VAR)
    set_y_val = next(instruction for instruction in p_code if instruction[0] == OP.SET_Y_VAL)

    assert set_y_val[1] == set_y_var[1]


def test_set_const_heap_overflow_occurs_after_successful_put_str() -> None:
    """Ensure heap exhaustion is reported at the individual WAM instruction.

    PUT_STR is allowed to consume its own three-cell structure representation
    even when no room remains for the following argument. SET_CONST must then
    return HEAP_OVERFLOW without mutating H. This protects the Ait-Kaci-style
    instruction boundary where PUT_STR and SET_* allocate independently.
    """
    config = load_config_toml(
        """
        [runtime]
        heap_size = 3
        """
    )

    machine = init_machine(config.runtime)
    state = machine.registers

    assert exec_PUT_STR(machine, 123, 1, 0) == WAMStatus.SUCCESS
    assert int(state.H[0]) == 3

    heap_before = int(state.H[0])

    assert exec_SET_CONST(machine, 456) == WAMStatus.HEAP_OVERFLOW
    assert int(state.H[0]) == heap_before
