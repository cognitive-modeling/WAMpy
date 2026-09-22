import numpy as np
import pytest
from wampy import Prolog
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import (
    ENTRY_INVALID_PC,
    init_compiler_state,
    undo_hypothesis,
)
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.config import load_config_toml
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml()


def _sid(symbol_table, symbol: str) -> int:
    return int(symbol_table.id_by_symbol[symbol])


def _op(compiled, pc: int) -> OP:
    return OP(int(compiled.code[pc, 0]))


def _arg(compiled, pc: int, arg_index: int) -> int:
    return int(compiled.code[pc, arg_index])


def _entry_at(compiler_state, entry, symbol_id: int, arity: int) -> int:
    predicate_slot = find_predicate_slot(compiler_state, symbol_id)
    assert predicate_slot >= 0
    return int(entry[predicate_slot, arity])


def _compile_static(static_src: str, config):
    static_program, symbol_table = parse(
        static_src,
        empty_symbol_table(config.frontend.ast),
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
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 0
    base_entry = compiled_program.predicate_entry.copy()
    base_end = compiled_program.code_size[0]
    base_code = compiled_program.code[:base_end].copy()

    return (
        compiled_program,
        compiler_state,
        base_entry,
        base_end,
        base_code,
        symbol_table,
    )


def _build_dynamic_program(dynamic_src: str, symbol_table, config):
    dynamic_program, _ = parse(
        dynamic_src,
        symbol_table,
        config,
    )
    return dynamic_program


def _compile_extension(
    compiled_program,
    compiler_state,
    dynamic_src,
    symbol_table,
    config,
):
    dynamic_program = _build_dynamic_program(
        dynamic_src,
        symbol_table,
        config,
    )

    if compiler_state.hypothesis_predicate_entry_undo_count[0] > 0:
        undo_hypothesis(compiler_state, compiled_program)

    compile_program(
        dynamic_program,
        compiled_program=compiled_program,
        compiler_state=compiler_state,
        config=config,
    )


def test_compile_extension_enforces_x_register_capacity():
    config = load_config_toml(
        """
        [compiler]
        max_arity = 2

        [runtime]
        max_x_registers = 2
        """
    )

    (
        static_compiled_program,
        static_state,
        _,
        _,
        _,
        symbol_table,
    ) = _compile_static(
        "base.",
        config,
    )

    with pytest.raises(
        ValueError,
        match="X register capacity exceeded",
    ):
        _compile_extension(
            static_compiled_program,
            static_state,
            "q((((a, b), c), d)).",
            symbol_table,
            config,
        )


def test_multi_clause_dynamic_over_multi_clause_static_emits_jmp_retry_bridge(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        base_code,
        symbol_table,
    ) = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        likes(john, pasta).
        marker(a).
        likes(john, sushi).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    likes_id = _sid(symbol_table, "likes")

    assert np.array_equal(
        compiled.code[:base_end],
        base_code,
    )
    assert (
        _entry_at(
            static_state,
            compiled.predicate_entry,
            likes_id,
            2,
        )
        == base_end
    )

    first_dynamic_prefix = base_end
    second_dynamic_prefix = int(compiled.code[first_dynamic_prefix, 1])
    bridge_pc = int(compiled.code[second_dynamic_prefix, 1])
    static_prefix_pc = _entry_at(
        static_state,
        base_entry,
        likes_id,
        2,
    )

    assert _op(compiled, first_dynamic_prefix) == OP.TRY_ME_ELSE
    assert _op(compiled, second_dynamic_prefix) == OP.RETRY_ME_ELSE
    assert _arg(compiled, first_dynamic_prefix, 2) == 0
    assert _arg(compiled, second_dynamic_prefix, 2) == 0
    assert _op(compiled, bridge_pc) == OP.JMP_RETRY
    assert _arg(compiled, bridge_pc, 1) == static_prefix_pc + 1
    assert _arg(compiled, bridge_pc, 2) == int(compiled.code[static_prefix_pc, 1])

    likes_slot = find_predicate_slot(static_state, likes_id)
    marker_id = _sid(symbol_table, "marker")
    marker_slot = find_predicate_slot(static_state, marker_id)
    assert static_state.hypothesis_predicate_entry_undo_count[0] == 2
    assert tuple(static_state.hypothesis_predicate_entry_undo[0]) == (
        likes_slot,
        2,
        static_prefix_pc,
    )
    assert tuple(static_state.hypothesis_predicate_entry_undo[1]) == (
        marker_slot,
        1,
        ENTRY_INVALID_PC,
    )


def test_single_dynamic_clause_over_single_static_clause_emits_jmp_trust_bridge(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        base_code,
        symbol_table,
    ) = _compile_static(
        """
        likes(john, pizza).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        likes(john, sushi).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    likes_id = _sid(symbol_table, "likes")
    static_clause_pc = _entry_at(
        static_state,
        base_entry,
        likes_id,
        2,
    )
    first_dynamic_prefix = base_end
    bridge_pc = int(compiled.code[first_dynamic_prefix, 1])

    assert np.array_equal(
        compiled.code[:base_end],
        base_code,
    )
    assert (
        _entry_at(
            static_state,
            compiled.predicate_entry,
            likes_id,
            2,
        )
        == base_end
    )
    assert _op(compiled, first_dynamic_prefix) == OP.TRY_ME_ELSE
    assert _op(compiled, bridge_pc) == OP.JMP_TRUST
    assert _arg(compiled, bridge_pc, 1) == static_clause_pc


def test_single_dynamic_clause_over_multi_static_clause_emits_jmp_retry_bridge(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        _,
        symbol_table,
    ) = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        likes(peter, pizza).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    likes_id = _sid(symbol_table, "likes")
    first_dynamic_prefix = base_end
    bridge_pc = int(compiled.code[first_dynamic_prefix, 1])
    static_prefix_pc = _entry_at(
        static_state,
        base_entry,
        likes_id,
        2,
    )

    assert (
        _entry_at(
            static_state,
            compiled.predicate_entry,
            likes_id,
            2,
        )
        == base_end
    )
    assert _op(compiled, first_dynamic_prefix) == OP.TRY_ME_ELSE
    assert _op(compiled, bridge_pc) == OP.JMP_RETRY
    assert _arg(compiled, bridge_pc, 1) == static_prefix_pc + 1
    assert _arg(compiled, bridge_pc, 2) == int(compiled.code[static_prefix_pc, 1])


def test_new_dynamic_predicate_has_no_static_bridge_and_keeps_static_entries(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        _,
        symbol_table,
    ) = _compile_static(
        """
        likes(john, pizza).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        dislikes(peter, pizza).
        dislikes(peter, pasta).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    likes_id = _sid(symbol_table, "likes")
    dislikes_id = _sid(symbol_table, "dislikes")

    dislikes_entry = _entry_at(
        static_state,
        compiled.predicate_entry,
        dislikes_id,
        2,
    )

    dynamic_ops = [OP(int(row[0])) for row in compiled.code[base_end : compiled.code_size[0]]]

    assert _entry_at(
        static_state,
        compiled.predicate_entry,
        likes_id,
        2,
    ) == _entry_at(
        static_state,
        base_entry,
        likes_id,
        2,
    )
    assert dislikes_entry == base_end
    assert dynamic_ops == [
        OP.TRY_ME_ELSE,
        OP.GET_CONST,
        OP.GET_CONST,
        OP.PROCEED,
        OP.TRUST_ME_ELSE_FAIL,
        OP.GET_CONST,
        OP.GET_CONST,
        OP.PROCEED,
    ]
    assert OP.JMP_RETRY not in dynamic_ops
    assert OP.JMP_TRUST not in dynamic_ops

    dislikes_slot = find_predicate_slot(static_state, dislikes_id)
    assert static_state.hypothesis_predicate_entry_undo_count[0] == 1
    assert tuple(static_state.hypothesis_predicate_entry_undo[0]) == (
        dislikes_slot,
        2,
        ENTRY_INVALID_PC,
    )


def test_interleaved_hypothesis_groups_preserve_solution_order_and_undo(wam_config):
    prolog = Prolog(
        """
        p(base1).
        p(base2).
        q(baseq).
        """,
        wam_config,
    )
    extension, prolog.symbol_table = parse(
        """
        p(ext1).
        q(extq).
        p(ext2).
        """,
        prolog.symbol_table,
        wam_config,
    )

    compile_program(
        extension,
        prolog.compiled_program,
        prolog.compiler_state,
        wam_config,
    )

    assert prolog.query_all("p(X).") == [
        {"X": "ext1"},
        {"X": "ext2"},
        {"X": "base1"},
        {"X": "base2"},
    ]
    assert prolog.query_all("q(X).") == [{"X": "extq"}, {"X": "baseq"}]

    undo_hypothesis(prolog.compiler_state, prolog.compiled_program)

    assert prolog.query_all("p(X).") == [{"X": "base1"}, {"X": "base2"}]
    assert prolog.query_all("q(X).") == [{"X": "baseq"}]


def test_undo_hypothesis_restores_an_extension_only_predicate_entry(wam_config):
    prolog = Prolog("p(base).", wam_config)
    extension, prolog.symbol_table = parse(
        """
        h(a).
        h(b).
        """,
        prolog.symbol_table,
        wam_config,
    )

    compile_program(
        extension,
        prolog.compiled_program,
        prolog.compiler_state,
        wam_config,
    )

    h_slot = find_predicate_slot(prolog.compiler_state, _sid(prolog.symbol_table, "h"))
    assert prolog.query_all("h(X).") == [{"X": "a"}, {"X": "b"}]

    undo_hypothesis(prolog.compiler_state, prolog.compiled_program)

    assert h_slot >= 0
    assert prolog.compiled_program.predicate_entry[h_slot, 1] == ENTRY_INVALID_PC


def test_undo_hypothesis_replays_sparse_entry_log_and_resets_only_count(wam_config):
    (
        compiled,
        compiler_state,
        base_entry,
        base_end,
        _,
        symbol_table,
    ) = _compile_static("likes(john, pizza).", wam_config)

    _compile_extension(
        compiled,
        compiler_state,
        "likes(john, sushi).",
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    likes_slot = find_predicate_slot(compiler_state, likes_id)
    undo_row = tuple(compiler_state.hypothesis_predicate_entry_undo[0])
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 1

    undo_hypothesis(compiler_state, compiled)

    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 0
    assert tuple(compiler_state.hypothesis_predicate_entry_undo[0]) == undo_row
    assert compiled.code_size[0] == base_end
    assert int(compiled.predicate_entry[likes_slot, 2]) == int(base_entry[likes_slot, 2])


def test_compile_program_requires_undo_before_replacing_an_installed_hypothesis(wam_config):
    compiled, compiler_state, _, _, _, symbol_table = _compile_static(
        "p(a).",
        wam_config,
    )

    _compile_extension(
        compiled,
        compiler_state,
        "p(b).",
        symbol_table,
        wam_config,
    )
    first_hypothesis_end = int(compiled.code_size[0])

    second_program = _build_dynamic_program("p(c).", symbol_table, wam_config)
    with pytest.raises(
        ValueError,
        match="undo_hypothesis\\(\\) must be called",
    ):
        compile_program(
            second_program,
            compiled_program=compiled,
            compiler_state=compiler_state,
            config=wam_config,
        )

    assert compiled.code_size[0] == first_hypothesis_end
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 1

    undo_hypothesis(compiler_state, compiled)
    compile_program(
        second_program,
        compiled_program=compiled,
        compiler_state=compiler_state,
        config=wam_config,
    )

    assert compiled.code_size[0] == first_hypothesis_end
    assert compiler_state.hypothesis_predicate_entry_undo_count[0] == 1


def test_hypothesis_entry_undo_log_enforces_configured_capacity():
    config = load_config_toml(
        """
        [compiler]
        max_hypothesis_predicate_changes = 1
        """
    )
    compiled, compiler_state, _, _, _, symbol_table = _compile_static(
        "p(a). q(a).",
        config,
    )

    with pytest.raises(
        ValueError,
        match="max_hypothesis_predicate_changes is too small",
    ):
        _compile_extension(
            compiled,
            compiler_state,
            "p(b). q(b).",
            symbol_table,
            config,
        )


def test_dynamic_rule_over_static_rule_uses_bridge_after_dynamic_rule_body(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        _,
        symbol_table,
    ) = _compile_static(
        """
        father(ted, bob).
        mother(jane, bob).
        parent(X, Y) :- father(X, Y).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        parent(X, Y) :- mother(X, Y).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    parent_id = _sid(symbol_table, "parent")

    parent_entry = _entry_at(
        static_state,
        compiled.predicate_entry,
        parent_id,
        2,
    )

    bridge_pc = int(compiled.code[parent_entry, 1])

    static_parent_entry = _entry_at(
        static_state,
        base_entry,
        parent_id,
        2,
    )

    dynamic_ops = [OP(int(row[0])) for row in compiled.code[parent_entry:bridge_pc]]

    assert parent_entry == base_end
    assert dynamic_ops == [
        OP.TRY_ME_ELSE,
        OP.GET_VAR,
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.PUT_VAL,
        OP.EXECUTE,
    ]
    assert _op(compiled, bridge_pc) == OP.JMP_TRUST
    assert _arg(compiled, bridge_pc, 1) == static_parent_entry


def test_recompiling_extension_restores_base_entries_and_replaces_previous_extension(
    wam_config,
):
    (
        base_compiled_program,
        base_state,
        base_entry,
        base_end,
        base_code,
        symbol_table,
    ) = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    _compile_extension(
        base_compiled_program,
        base_state,
        """
        likes(peter, pizza).
        """,
        symbol_table,
        wam_config,
    )

    first_dynamic_pc = int(base_state.pc[0])

    _compile_extension(
        base_compiled_program,
        base_state,
        """
        dislikes(peter, pasta).
        """,
        symbol_table,
        wam_config,
    )
    compiled = base_compiled_program

    likes_id = _sid(symbol_table, "likes")
    dislikes_id = _sid(symbol_table, "dislikes")

    assert first_dynamic_pc > base_end
    assert base_state.pc_onto[0] == base_end

    assert np.array_equal(
        compiled.code[:base_end],
        base_code,
    )

    assert _entry_at(
        base_state,
        compiled.predicate_entry,
        likes_id,
        2,
    ) == _entry_at(
        base_state,
        base_entry,
        likes_id,
        2,
    )

    assert (
        _entry_at(
            base_state,
            compiled.predicate_entry,
            dislikes_id,
            2,
        )
        == base_end
    )

    assert compiled.code_size[0] < first_dynamic_pc


def test_dynamic_predicate_arity_does_not_shadow_static_same_name_other_arity(
    wam_config,
):
    (
        static_compiled_program,
        static_state,
        base_entry,
        base_end,
        _,
        symbol_table,
    ) = _compile_static(
        """
        p(a).
        """,
        wam_config,
    )

    _compile_extension(
        static_compiled_program,
        static_state,
        """
        p(a, b).
        """,
        symbol_table,
        wam_config,
    )
    compiled = static_compiled_program

    p_id = _sid(symbol_table, "p")

    assert _entry_at(
        static_state,
        compiled.predicate_entry,
        p_id,
        1,
    ) == _entry_at(
        static_state,
        base_entry,
        p_id,
        1,
    )

    assert (
        _entry_at(
            static_state,
            compiled.predicate_entry,
            p_id,
            2,
        )
        == base_end
    )

    assert (
        _entry_at(
            static_state,
            compiled.predicate_entry,
            p_id,
            0,
        )
        == ENTRY_INVALID_PC
    )
