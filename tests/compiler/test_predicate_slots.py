import numpy as np
import pytest
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiled_query import init_compiled_query
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import (
    NO_PREDICATE_SLOT,
    init_compiler_state,
)
from wampy.compiler.diagnostics import predicate_opcode_trace
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.compiler.query import compile_query
from wampy.config import DEFAULT_CONFIG, WAMConfig
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import (
    intern_symbol,
    symbol_table_from_mapping,
)
from wampy.runtime.interpreter import run
from wampy.runtime.machine import init_machine
from wampy.status import WAMStatus


def _config(
    max_predicates: int,
    symbol_id_dtype: np.dtype | None = None,
) -> WAMConfig:
    config = DEFAULT_CONFIG._replace(
        compiler=DEFAULT_CONFIG.compiler._replace(
            max_predicates=max_predicates,
        ),
    )

    if symbol_id_dtype is not None:
        config = config._replace(
            frontend=config.frontend._replace(
                ast=config.frontend.ast._replace(
                    symbol_id_dtype=symbol_id_dtype,
                ),
            ),
        )

    return config


def test_sparse_raw_symbol_ids_use_compact_predicate_slots() -> None:
    config = _config(max_predicates=2)

    symbol_table = symbol_table_from_mapping(
        {
            10_000: "p",
            10_001: "a",
            10_002: "q",
            10_003: "b",
        },
        config=config.frontend.ast,
    )

    program, symbol_table = parse(
        "q(b). p(a) :- q(a).",
        symbol_table,
        config,
    )

    compiler_state = init_compiler_state(config)
    compiled = init_compiled_program(config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=config,
    )

    p_id = int(symbol_table.id_by_symbol["p"])
    q_id = int(symbol_table.id_by_symbol["q"])
    a_id = int(symbol_table.id_by_symbol["a"])
    b_id = int(symbol_table.id_by_symbol["b"])

    p_slot = find_predicate_slot(
        compiler_state,
        p_id,
    )
    q_slot = find_predicate_slot(
        compiler_state,
        q_id,
    )

    assert compiled.predicate_entry.shape == (
        2,
        config.compiler.max_arity + 1,
    )
    assert compiler_state.predicate_count[0] == 2
    assert {p_slot, q_slot} == {0, 1}

    assert compiler_state.symbol_by_predicate_slot[p_slot] == p_id
    assert compiler_state.symbol_by_predicate_slot[q_slot] == q_id

    assert (
        find_predicate_slot(
            compiler_state,
            a_id,
        )
        == NO_PREDICATE_SLOT
    )
    assert (
        find_predicate_slot(
            compiler_state,
            b_id,
        )
        == NO_PREDICATE_SLOT
    )

    p_pc = int(
        compiled.predicate_entry[p_slot, 1],
    )
    p_code = compiled.code[p_pc : compiled.code_size[0]]

    execute = next(row for row in p_code if OP(int(row[0])) == OP.EXECUTE)

    assert int(execute[1]) == q_slot

    assert any(OP(int(row[0])) == OP.PUT_CONST and int(row[1]) == a_id for row in p_code)


def test_body_only_predicate_is_registered_during_clause_analysis() -> None:
    config = _config(max_predicates=2)
    program, symbol_table = parse(
        "p(a) :- q(a).",
        symbol_table_from_mapping({}),
        config,
    )

    compiler_state = init_compiler_state(config)
    compiled = init_compiled_program(config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=config,
    )

    p_id = int(symbol_table.id_by_symbol["p"])
    q_id = int(symbol_table.id_by_symbol["q"])
    p_slot = find_predicate_slot(compiler_state, p_id)
    q_slot = find_predicate_slot(compiler_state, q_id)

    assert compiler_state.predicate_count[0] == 2
    assert p_slot == 0
    assert q_slot == 1

    p_pc = int(compiled.predicate_entry[p_slot, 1])
    p_code = compiled.code[p_pc : compiled.code_size[0]]
    execute = next(row for row in p_code if OP(int(row[0])) == OP.EXECUTE)

    assert int(execute[1]) == q_slot


def test_dynamic_compilation_keeps_existing_slots_and_appends_new_predicates() -> None:
    config = _config(
        max_predicates=3,
        symbol_id_dtype=np.dtype(np.uint32),
    )

    symbol_table = symbol_table_from_mapping(
        {
            2_000_000_000: "p",
            200: "a",
            100: "q",
            201: "b",
        },
        config=config.frontend.ast,
    )

    static_program, symbol_table = parse(
        "p(a).",
        symbol_table,
        config,
    )

    compiler_state = init_compiler_state(config)
    compiled_program = init_compiled_program(config)
    compile_program(
        static_program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=config,
    )

    p_id = int(
        symbol_table.id_by_symbol["p"],
    )
    q_id = int(
        symbol_table.id_by_symbol["q"],
    )

    p_slot_before = find_predicate_slot(
        compiler_state,
        p_id,
    )

    assert p_slot_before == 0

    dynamic_program, symbol_table = parse(
        "q(b). p(b).",
        symbol_table,
        config,
    )

    compile_program(
        dynamic_program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=config,
    )

    p_slot_after = find_predicate_slot(
        compiler_state,
        p_id,
    )
    q_slot_after = find_predicate_slot(
        compiler_state,
        q_id,
    )

    assert p_slot_after == p_slot_before
    assert q_slot_after == 1

    assert compiler_state.predicate_count[0] == 2

    assert compiler_state.symbol_by_predicate_slot[0] == p_id
    assert compiler_state.symbol_by_predicate_slot[1] == q_id

    assert compiled_program.predicate_entry.shape == (
        3,
        config.compiler.max_arity + 1,
    )

    assert compiler_state.predicate_symbols_sorted[:2].tolist() == [q_id, p_id]

    assert compiler_state.predicate_slots_sorted[:2].tolist() == [1, 0]


def test_large_raw_symbol_id_does_not_expand_lookup_storage() -> None:
    config = _config(
        max_predicates=1,
        symbol_id_dtype=np.dtype(np.uint32),
    )

    symbol_table = symbol_table_from_mapping(
        {
            48: "a",
            2_000_000_000: "p",
        },
        config=config.frontend.ast,
    )

    program, symbol_table = parse(
        "p(a).",
        symbol_table,
        config,
    )

    compiler_state = init_compiler_state(config)
    compiled = init_compiled_program(config)
    compile_program(
        program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=config,
    )

    p_id = int(
        symbol_table.id_by_symbol["p"],
    )

    assert (
        find_predicate_slot(
            compiler_state,
            p_id,
        )
        == 0
    )

    assert compiler_state.predicate_symbols_sorted.shape == (1,)
    assert compiler_state.predicate_slots_sorted.shape == (1,)
    assert compiler_state.symbol_by_predicate_slot.shape == (1,)

    assert predicate_opcode_trace(
        compiled,
        compiler_state,
        p_id,
        1,
    )

    assert (
        predicate_opcode_trace(
            compiled,
            compiler_state,
            2_000_000_001,
            1,
        )
        == ()
    )


def test_resolving_more_unique_predicates_than_capacity_fails() -> None:
    config = _config(max_predicates=1)

    symbol_table = symbol_table_from_mapping(
        {
            10_000: "p",
            10_001: "q",
            10_002: "a",
        },
        config=config.frontend.ast,
    )

    program, symbol_table = parse(
        "p(a). q(a).",
        symbol_table,
        config,
    )

    compiled_program = init_compiled_program(config)
    with pytest.raises(
        ValueError,
        match="max_predicates is too small for the program",
    ):
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(
                config,
            ),
            config=config,
        )


def test_sparse_raw_symbol_program_executes_through_top_level_slot_lookup() -> None:
    config = _config(max_predicates=1)

    symbol_table = symbol_table_from_mapping(
        {
            30_000: "p",
            30_001: "a",
        },
        config=config.frontend.ast,
    )

    program, symbol_table = parse(
        "p(a).",
        symbol_table,
        config,
    )

    compiler_state = init_compiler_state(config)
    compiled_program = init_compiled_program(config)
    compile_program(
        program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=config,
    )

    query_program, _ = parse(
        "?- p(a).",
        symbol_table,
        config,
    )

    compiled_query = init_compiled_query(config)
    compile_query(
        query_program,
        compiled_query,
        compiled_program,
        compiler_state,
        config,
    )

    machine = init_machine(
        config.runtime,
    )

    status = run(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.SUCCESS


def test_default_user_symbol_allocation_starts_at_user_symbol_first() -> None:
    symbol_table = symbol_table_from_mapping({})

    assert (
        intern_symbol(
            symbol_table,
            "p",
        )
        == symbol_table.first_user_symbol_id
    )
