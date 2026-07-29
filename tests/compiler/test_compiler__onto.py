import numpy as np
import pytest

from wampy.compiler.compiler import (
    ENTRY_INVALID_PC,
    OP,
    compile,
    compile_program_onto,
    init_compiler,
)
from wampy.config import load_config_toml
from wampy.frontend.ast.program import init_program
from wampy.frontend.parser import _split_clauses, add_prolog_term, build_new_program
from ..test_engine import extend_symbol_table_from_source

CONFIG_TOML = """
[ast]
num_terms = 40
max_terms_nodes = 120
max_terms_nodes_childs = 5

[entry]
max_functors = 256
max_arity = 8
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _sid(symbol_table, symbol: str) -> int:
    return int(symbol_table.id_by_symbol[symbol])


def _op(compiled, pc: int) -> OP:
    return OP(int(compiled.code[pc, 0]))


def _arg(compiled, pc: int, arg_index: int) -> int:
    return int(compiled.code[pc, arg_index])


def _compile_static(static_src: str, config):
    static_program, symbol_table = build_new_program(static_src, config)
    static_state = compile(static_program, init_compiler(config), config)
    static_entry = static_state.entry.copy()
    static_size = int(static_state.pc.value)
    static_code = static_state.code[:static_size].copy()
    return static_state, static_entry, static_size, static_code, symbol_table


def _build_dynamic_program(dynamic_src: str, symbol_table, config):
    extend_symbol_table_from_source(symbol_table, dynamic_src)
    dynamic_program = init_program(config)
    for clause in _split_clauses(dynamic_src):
        add_prolog_term(dynamic_program, clause, symbol_table)
    return dynamic_program


def _compile_ontop(static_state, static_entry, static_size, dynamic_src, symbol_table, config):
    dynamic_program = _build_dynamic_program(dynamic_src, symbol_table, config)
    return compile_program_onto(static_state, static_entry, static_size, dynamic_program, config)


def test_multi_clause_dynamic_over_multi_clause_static_emits_jmp_retry_bridge(wam_config):
    static_state, static_entry, static_size, static_code, symbol_table = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        likes(john, pasta).
        likes(john, sushi).
        """,
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    assert np.array_equal(compiled.code[:static_size], static_code)
    assert compiled.entry[likes_id, 2] == static_size

    first_dynamic_prefix = static_size
    second_dynamic_prefix = int(compiled.code[first_dynamic_prefix, 1])
    bridge_pc = int(compiled.code[second_dynamic_prefix, 1])
    static_prefix_pc = int(static_entry[likes_id, 2])

    assert _op(compiled, first_dynamic_prefix) == OP.TRY
    assert _op(compiled, second_dynamic_prefix) == OP.RETRY
    assert _op(compiled, bridge_pc) == OP.JMP_RETRY
    assert _arg(compiled, bridge_pc, 1) == static_prefix_pc + 1
    assert _arg(compiled, bridge_pc, 2) == int(compiled.code[static_prefix_pc, 1])


def test_single_dynamic_clause_over_single_static_clause_emits_jmp_trust_bridge(wam_config):
    static_state, static_entry, static_size, static_code, symbol_table = _compile_static(
        """
        likes(john, pizza).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        likes(john, sushi).
        """,
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    static_clause_pc = int(static_entry[likes_id, 2])
    first_dynamic_prefix = static_size
    bridge_pc = int(compiled.code[first_dynamic_prefix, 1])

    assert np.array_equal(compiled.code[:static_size], static_code)
    assert compiled.entry[likes_id, 2] == static_size
    assert _op(compiled, first_dynamic_prefix) == OP.TRY
    assert _op(compiled, bridge_pc) == OP.JMP_TRUST
    assert _arg(compiled, bridge_pc, 1) == static_clause_pc


def test_single_dynamic_clause_over_multi_static_clause_emits_jmp_retry_bridge(wam_config):
    static_state, static_entry, static_size, _, symbol_table = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        likes(peter, pizza).
        """,
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    first_dynamic_prefix = static_size
    bridge_pc = int(compiled.code[first_dynamic_prefix, 1])
    static_prefix_pc = int(static_entry[likes_id, 2])

    assert compiled.entry[likes_id, 2] == static_size
    assert _op(compiled, first_dynamic_prefix) == OP.TRY
    assert _op(compiled, bridge_pc) == OP.JMP_RETRY
    assert _arg(compiled, bridge_pc, 1) == static_prefix_pc + 1
    assert _arg(compiled, bridge_pc, 2) == int(compiled.code[static_prefix_pc, 1])


def test_new_dynamic_predicate_has_no_static_bridge_and_keeps_static_entries(wam_config):
    static_state, static_entry, static_size, _, symbol_table = _compile_static(
        """
        likes(john, pizza).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        dislikes(peter, pizza).
        dislikes(peter, pasta).
        """,
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    dislikes_id = _sid(symbol_table, "dislikes")
    dislikes_entry = int(compiled.entry[dislikes_id, 2])
    dynamic_ops = [OP(int(row[0])) for row in compiled.code[static_size : compiled.pc.value]]

    assert compiled.entry[likes_id, 2] == static_entry[likes_id, 2]
    assert dislikes_entry == static_size
    assert dynamic_ops[0] == OP.TRY
    assert OP.TRUST in dynamic_ops
    assert OP.JMP_RETRY not in dynamic_ops
    assert OP.JMP_TRUST not in dynamic_ops


def test_dynamic_rule_over_static_rule_uses_bridge_after_dynamic_rule_body(wam_config):
    static_state, static_entry, static_size, _, symbol_table = _compile_static(
        """
        father(ted, bob).
        mother(jane, bob).
        parent(X, Y) :- father(X, Y).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        parent(X, Y) :- mother(X, Y).
        """,
        symbol_table,
        wam_config,
    )

    parent_id = _sid(symbol_table, "parent")
    parent_entry = int(compiled.entry[parent_id, 2])
    bridge_pc = int(compiled.code[parent_entry, 1])
    static_parent_entry = int(static_entry[parent_id, 2])
    dynamic_ops = [OP(int(row[0])) for row in compiled.code[parent_entry:bridge_pc]]

    assert parent_entry == static_size
    assert dynamic_ops == [
        OP.TRY,
        OP.GET_VAR,
        OP.GET_VAR,
        OP.PUT_VAL,
        OP.PUT_VAL,
        OP.EXECUTE,
        OP.PROCEED,
    ]
    assert _op(compiled, bridge_pc) == OP.JMP_TRUST
    assert _arg(compiled, bridge_pc, 1) == static_parent_entry


def test_recompiling_ontop_restores_static_entries_and_overwrites_old_dynamic_program(wam_config):
    static_state, static_entry, static_size, _, symbol_table = _compile_static(
        """
        likes(john, pizza).
        likes(mary, sushi).
        """,
        wam_config,
    )

    _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        likes(peter, pizza).
        """,
        symbol_table,
        wam_config,
    )
    first_dynamic_pc = int(static_state.pc.value)

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        dislikes(peter, pasta).
        """,
        symbol_table,
        wam_config,
    )

    likes_id = _sid(symbol_table, "likes")
    dislikes_id = _sid(symbol_table, "dislikes")

    assert first_dynamic_pc > static_size
    assert (
        compiled.entry[likes_id, 2] == static_entry[likes_id, 2]
    )  ## Checks that the static predicate likes/2 still points to static entry point
    assert (
        compiled.entry[dislikes_id, 2] == static_size
    )  ## Checks that the dynamic predicate dislikes/2 starts at static_size
    assert int(compiled.pc.value) < first_dynamic_pc
    assert np.all(compiled.clause_entry_term[compiled.pc.value :] == -1)


def test_dynamic_predicate_arity_does_not_shadow_static_same_name_other_arity(wam_config):
    static_state, static_entry, static_size, _, symbol_table = _compile_static(
        """
        p(a).
        """,
        wam_config,
    )

    compiled = _compile_ontop(
        static_state,
        static_entry,
        static_size,
        """
        p(a, b).
        """,
        symbol_table,
        wam_config,
    )

    p_id = _sid(symbol_table, "p")

    assert compiled.entry[p_id, 1] == static_entry[p_id, 1]
    assert compiled.entry[p_id, 2] == static_size
    assert compiled.entry[p_id, 0] == ENTRY_INVALID_PC
