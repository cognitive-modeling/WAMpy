# core/test_loader.py

import pytest

import wampy as wam
import wampy.frontend.parser as parser
from wampy.config import load_config_toml
from wampy.frontend.symbol_table import SymbolTable


class DummyProgram:
    def __init__(self):
        pass


@pytest.fixture
def dummy_program():
    return DummyProgram()


CONFIG_TOML = """
[ast]
num_terms = 20
max_terms_nodes = 100
max_terms_nodes_childs = 5
""".strip()


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


# ----------------------------
# _extract_symbols
# ----------------------------


def test_extract_symbols_basic():
    clauses = [
        "parent(john, mary).",
        "parent(mary, alice).",
        "ancestor(X, Y) :- parent(X, Y).",
    ]

    symbols = parser._extract_symbols(clauses)

    assert symbols == sorted(["ancestor", "alice", "john", "mary", "parent"])


def test_extract_symbols_no_duplicates():
    clauses = [
        "p(a).",
        "p(b).",
        "p(c).",
    ]

    symbols = parser._extract_symbols(clauses)

    assert symbols == ["a", "b", "c", "p"]


def test_extract_symbols_quoted_atoms_are_unquoted():
    clauses = [
        "add('1', '+', '2', '=', '3').",
    ]

    symbols = parser._extract_symbols(clauses)

    assert symbols == ["+", "1", "2", "3", "=", "add"]


# ----------------------------
# build_program
# ----------------------------


def test_build_program_single_line(wam_config):
    program, binding = parser.build_new_program("parent(john, mary)", wam_config)

    assert isinstance(binding, SymbolTable)
    assert set(binding.values()) == {"parent", "john", "mary"}


def test_build_program_multiline_string(wam_config):
    source = """
    parent(john, mary).
    parent(mary, alice).
    """
    program, binding = parser.build_new_program(source, wam_config)

    # Assert bindings
    assert set(binding.values()) == {"parent", "john", "mary", "alice"}

    # Assert clauses were loaded
    # (exact assertion depends on Program internals)


def test_build_program_rejects_list_input():
    with pytest.raises(TypeError):
        parser.build_new_program(
            [
                "p(a).",
                "p(b).",
            ]
        )


def test_parse_with_symbol_table_extends_table_and_preserves_ids(wam_config):
    initial_source = """
    male(anakin).
    parent(anakin, luke).
    father(A, B) :- parent(A, B), male(A).
    """
    _, symbol_table = wam.parse(initial_source, wam_config)
    existing_ids = {
        symbol: int(symbol_table.id_by_symbol[symbol])
        for symbol in ("anakin", "father", "luke", "male", "parent")
    }

    program, extended_table = wam.parse_with_symbol_table(
        "parent(padme, luke).", symbol_table
    )

    assert extended_table is symbol_table
    assert int(extended_table.id_by_symbol["padme"]) not in existing_ids.values()
    for symbol, symbol_id in existing_ids.items():
        assert int(extended_table.id_by_symbol[symbol]) == symbol_id

    clause_symbols = set(int(symbol) for symbol in program.symbol[0])
    assert int(extended_table.id_by_symbol["parent"]) in clause_symbols
    assert int(extended_table.id_by_symbol["padme"]) in clause_symbols
    assert int(extended_table.id_by_symbol["luke"]) in clause_symbols


# ----------------------------
# _split_clauses (new)
# ----------------------------


def test_split_clauses_single_without_dot():
    clauses = parser._split_clauses("parent(john, mary)")
    assert clauses == ["parent(john, mary)."]


def test_split_clauses_multiple_clauses_one_line():
    clauses = parser._split_clauses("p(a). q(b). r(c).")
    assert clauses == ["p(a).", "q(b).", "r(c)."]


def test_split_clauses_multiline_rule_with_commas_not_split():
    source = """
    answer(N1, '+', N2, '=', A) :-
        sym(P1, N1),
        sym(P2, N2),
        add(P1, P2, R),
        sym(R, A).
    """
    clauses = parser._split_clauses(source)
    assert len(clauses) == 1
    assert clauses[0].strip().endswith(".")


def test_split_clauses_dot_inside_quotes_not_split():
    clauses = parser._split_clauses("p('a.b', c). q(d).")
    assert clauses == ["p('a.b', c).", "q(d)."]


def test_split_clauses_dot_inside_nested_parentheses_not_split_until_depth_0():
    clauses = parser._split_clauses("p(f(g(a)), h(b)). q(c).")
    assert clauses == ["p(f(g(a)), h(b)).", "q(c)."]


def test_split_clauses_ignores_extra_whitespace_and_blank_input():
    assert parser._split_clauses("   \n  ") == []
    assert parser._split_clauses("\n\np(a).\n\n") == ["p(a)."]


def test_split_clauses_ignores_prolog_block_comments():
    clauses = parser._split_clauses("/* 0000 */ p(a). /* dot . */ q(b).")
    assert clauses == ["p(a).", "q(b)."]


# ----------------------------
# _extract_symbols (more coverage)
# ----------------------------


def test_extract_symbols_ignores_variables_and_underscore_vars():
    clauses = [
        "ancestor(X, Y) :- parent(X, Y).",
        "p(_Tmp, X, y).",
    ]
    symbols = parser._extract_symbols(clauses)
    # variables X, Y, _Tmp must not appear; lowercase atom y must appear
    assert "X" not in symbols
    assert "Y" not in symbols
    assert "_Tmp" not in symbols
    assert set(["ancestor", "parent", "p", "y"]).issubset(set(symbols))


def test_extract_symbols_handles_nested_terms_and_numbers_as_quoted_atoms():
    clauses = [
        "t(f(a), g(b, c)).",
        "sym(s(s(zero)), '2').",
    ]
    symbols = parser._extract_symbols(clauses)
    # note: '2' should be unquoted into 2
    assert set(["t", "f", "g", "a", "b", "c", "sym", "s", "zero", "2"]).issubset(set(symbols))


def test_extract_symbols_does_not_create_punctuation_tokens():
    clauses = [
        "answer(N1, '+', N2, '=', A) :- sym(P1, N1), sym(P2, N2).",
    ]
    symbols = parser._extract_symbols(clauses)
    assert "+" in symbols
    assert "=" in symbols
    # ensure no junk tokens like "N1)" ever appear
    assert not any(sym.endswith(")") for sym in symbols)


def test_extract_symbols_treats_tuple_comma_as_builtin():
    symbols = parser._extract_symbols(["next_num((zero, one))."])

    assert set(symbols) == {"next_num", "one", "zero"}


def test_parse_tuple_shorthand_as_comma_functor():
    assert parser._parse_term("(zero, one)") == (",", ["zero", "one"])
    assert parser._parse_term("((zero, one), two)") == (",", ["(zero, one)", "two"])


# ----------------------------
# build_program (more coverage)
# ----------------------------


def test_build_program_accepts_multiline_rule_without_line_dots(wam_config):
    # regression: previously line-based splitting created tokens like N1)
    source = """
    sym(zero, '0').
    add(zero, Y, Y).
    answer(N1, '+', N2, '=', A) :-
        sym(P1, N1),
        add(P1, zero, R),
        sym(R, A).
    """
    program, binding = parser.build_new_program(source, wam_config)
    # sanity: key atoms exist, and there are no corrupted symbols
    assert "answer" in binding.values()
    assert "sym" in binding.values()
    assert "add" in binding.values()
    assert "+" in binding.values()
    assert "=" in binding.values()
    assert not any(v.endswith(")") for v in binding.values())


def test_build_program_multiple_clauses_one_line(wam_config):
    program, binding = parser.build_new_program("p(a). q(b).", wam_config)
    assert set(binding.values()) == {"p", "q", "a", "b"}


def test_build_program_accepts_tuple_shorthand_without_comma_binding(wam_config):
    program, binding = parser.build_new_program("next_num((zero, one)).", wam_config)

    assert set(binding.values()) == {"next_num", "one", "zero"}


def test_build_program_ignores_prolog_block_comment_prefix(wam_config):
    program, binding = parser.build_new_program("/* 0000 */ p(a).", wam_config)
    assert set(binding.values()) == {"p", "a"}


def test_extract_symbols_ignores_true_builtin():
    clauses = [
        "p :- true.",
        "q(X) :- true, p(X).",
    ]

    symbols = parser._extract_symbols(clauses)

    assert "true" not in symbols
    assert "p" in symbols
    assert "q" in symbols


def test_build_program_accepts_true_builtin(wam_config):
    program, binding = parser.build_new_program("p :- true.", wam_config)

    # true must NOT be in bindings
    assert "true" not in binding.values()

    # p must still be bound
    assert "p" in binding.values()


def test_true_is_lowered_to_gid_true(wam_config):
    from wampy.frontend.ast.symbol_id import SymID

    program, _ = parser.build_new_program("p :- true.", wam_config)

    assert program.symbol[0, 0] == SymID.CLAUSE
    assert any(program.symbol[0, i] == SymID.TRUE for i in range(program.symbol.shape[1]))
    assert all(program.symbol[tid, 0] != 209 for tid in range(program.symbol.shape[0]))


def test_true_and_goal_both_present_in_ast(wam_config):
    from wampy.frontend.ast.symbol_id import SymID
    from wampy.frontend.ast.program import NONE

    program, binding = parser.build_new_program("p :- true, q.", wam_config)

    # find the clause term_id
    clause_tid = next(tid for tid in range(program.symbol.shape[0]) if program.symbol[tid, 0] == SymID.CLAUSE)

    # TODO: add testcases


def test_build_query_anonymous_vars_are_distinct(wam_config):
    from wampy.frontend.ast.symbol_id import SymID

    _, binding = parser.build_new_program("p(a, b).", wam_config)
    q = parser.build_query("p(_, _).", binding, wam_config)

    assert q.arity == 2
    s0 = int(q.program.symbol[int(q.term_id), int(q.args[0])])
    s1 = int(q.program.symbol[int(q.term_id), int(q.args[1])])
    assert SymID.VAR_FIRST <= s0 <= SymID.VAR_LAST
    assert SymID.VAR_FIRST <= s1 <= SymID.VAR_LAST
    assert s0 != s1


def test_build_query_named_underscore_var_is_reused(wam_config):
    _, binding = parser.build_new_program("p(a, b).", wam_config)
    q = parser.build_query("p(_Tmp, _Tmp).", binding, wam_config)

    assert q.arity == 2
    s0 = int(q.program.symbol[int(q.term_id), int(q.args[0])])
    s1 = int(q.program.symbol[int(q.term_id), int(q.args[1])])
    assert s0 == s1


def test_build_query_quoted_zero_arity_predicate(wam_config):
    program, binding = parser.build_new_program("'2'.", wam_config)

    q = parser.build_query("'2'.", binding, wam_config)

    assert q.arity == 0
    assert q.fun == binding.id_by_symbol["2"]


def test_build_query_quoted_zero_arity_predicate_fails_without_symbol_registration(wam_config):
    # only prove that quoted predicate must resolve against known symbols
    program, binding = parser.build_new_program("p(a).", wam_config)

    with pytest.raises(RuntimeError, match="UNDEFINED_PREDICATE"):
        parser.build_query("'2'.", binding, wam_config)


def test_build_queries_accepts_quoted_zero_arity_predicate(wam_config):
    _, binding = parser.build_new_program("'2'.\np(a).", wam_config)

    queries = parser.build_queries(("'2'.",), binding, wam_config)
    assert len(queries) == 1
    assert queries[0].arity == 0
    assert queries[0].fun == binding.id_by_symbol["2"]
