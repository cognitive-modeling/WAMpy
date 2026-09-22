import pytest
import wampy as wam
import wampy.frontend.parser as parser
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import (
    get_query_argument,
    get_query_arity,
    get_query_functor,
    get_term_capacity,
    get_term_count,
)
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.symbol_table import SymbolTable, empty_symbol_table


class DummyProgram:
    def __init__(self):
        pass


@pytest.fixture
def dummy_program():
    return DummyProgram()


@pytest.fixture(scope="session")
def wam_config():
    return WAMConfig()


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
# parse
# ----------------------------


def test_parse_single_line(wam_config):
    symbol_table = empty_symbol_table(wam_config.frontend.ast)
    program, binding = parser.parse("parent(john, mary)", symbol_table, wam_config)

    assert binding is symbol_table
    assert isinstance(binding, SymbolTable)
    assert set(binding.values()) == {"parent", "john", "mary"}


def test_parse_keeps_configured_capacity_and_tracks_allocated_terms(wam_config):
    program, _ = parser.parse(
        "p(a). q(b). r(c).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    assert get_term_capacity(program) == wam_config.frontend.ast.max_terms
    assert get_term_count(program) == 3


def test_empty_parse_has_no_allocated_terms(wam_config):
    program, _ = parser.parse("", empty_symbol_table(wam_config.frontend.ast), wam_config)

    assert get_term_capacity(program) == wam_config.frontend.ast.max_terms
    assert get_term_count(program) == 0


def test_first_user_atom_uses_user_symbol_first(wam_config):
    _program, binding = parser.parse(
        "p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    assert int(binding.id_by_symbol["a"]) == binding.first_user_symbol_id
    assert int(binding.id_by_symbol["p"]) == binding.first_user_symbol_id + 1


def test_clause_variables_reset_to_the_fixed_range(wam_config):
    program, _binding = parser.parse(
        "p(X). q(Y).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    variable_ids_by_term = []
    for term_id in range(get_term_count(program)):
        if program.node_symbols[term_id, 0] == 0:
            continue
        variable_ids_by_term.append(
            sorted(
                {
                    int(symbol_id)
                    for symbol_id in program.node_symbols[term_id]
                    if (
                        program.first_variable_symbol_id
                        <= int(symbol_id)
                        < program.first_user_symbol_id
                    )
                }
            )
        )

    assert variable_ids_by_term[:2] == [
        [program.first_variable_symbol_id],
        [program.first_variable_symbol_id],
    ]


def test_maximum_variables_are_allocated_without_overflow():
    config = WAMConfig()
    variable_names = ", ".join(f"V{i}" for i in range(config.frontend.ast.max_variables_per_term))

    program, _binding = parser.parse(
        f"p({variable_names}).", empty_symbol_table(config.frontend.ast), config
    )
    allocated = sorted(
        {
            int(symbol_id)
            for symbol_id in program.node_symbols[0]
            if (program.first_variable_symbol_id <= int(symbol_id) < program.first_user_symbol_id)
        }
    )

    assert allocated == list(range(program.first_variable_symbol_id, program.first_user_symbol_id))

    with pytest.raises(
        RuntimeError,
        match=f"maximum is {config.frontend.ast.max_variables_per_term}",
    ):
        parser.parse(f"p({variable_names}, V32).", empty_symbol_table(config.frontend.ast), config)


def test_parse_multiple_clauses_one_line(wam_config):
    _program, binding = parser.parse(
        "p(a). q(b).", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    assert set(binding.values()) == {"p", "q", "a", "b"}


def test_parse_multiline_string(wam_config):
    source = """
    parent(john, mary).
    parent(mary, alice).
    """
    program, binding = parser.parse(source, empty_symbol_table(wam_config.frontend.ast), wam_config)

    # Assert bindings
    assert set(binding.values()) == {"parent", "john", "mary", "alice"}

    # Assert clauses were loaded
    # (exact assertion depends on Program internals)


def test_parse_accepts_tuple_shorthand_without_comma_binding(wam_config):
    _program, binding = parser.parse(
        "next_num((zero, one)).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    assert set(binding.values()) == {"next_num", "one", "zero"}


def test_parse_ignores_prolog_block_comment_prefix(wam_config):
    _program, binding = parser.parse(
        "/* 0000 */ p(a).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    assert set(binding.values()) == {"p", "a"}


def test_parse_rejects_list_input():
    with pytest.raises(TypeError):
        parser.parse(
            [  # ty: ignore[invalid-argument-type]
                "p(a).",
                "p(b).",
            ],
            empty_symbol_table(),
        )


def test_parse_requires_an_existing_symbol_table():
    with pytest.raises(TypeError, match="symbol_table"):
        parser.parse("p(a).")  # ty: ignore[missing-argument]


@pytest.mark.parametrize(
    "removed_name",
    ["build_new_program", "build_program", "parse_with_symbol_table"],
)
def test_parse_is_the_only_program_loading_entry_point(removed_name):
    assert not hasattr(parser, removed_name)
    assert not hasattr(wam, removed_name)


def test_parse_extends_table_and_preserves_ids(wam_config):
    initial_source = """
    male(anakin).
    parent(anakin, luke).
    father(A, B) :- parent(A, B), male(A).
    """
    _, symbol_table = wam.parse(
        initial_source, empty_symbol_table(wam_config.frontend.ast), wam_config
    )
    existing_ids = {
        symbol: int(symbol_table.id_by_symbol[symbol])
        for symbol in ("anakin", "father", "luke", "male", "parent")
    }

    program, extended_table = wam.parse("parent(padme, luke).", symbol_table, wam_config)

    assert extended_table is symbol_table
    assert int(extended_table.id_by_symbol["padme"]) not in existing_ids.values()
    for symbol, symbol_id in existing_ids.items():
        assert int(extended_table.id_by_symbol[symbol]) == symbol_id

    clause_symbols = set(int(symbol) for symbol in program.node_symbols[0])
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
    assert parser._parse_term("(zero, one, two)") == (",", ["zero", "(one, two)"])
    assert parser._parse_term(r"\+ (zero, one)") == (r"\+", ["(zero, one)"])
    assert parser._parse_term("not((zero, one))") == (r"\+", ["(zero, one)"])


# ----------------------------
# parse (more coverage)
# ----------------------------


def test_parse_accepts_multiline_rule_without_line_dots(wam_config):
    # regression: previously line-based splitting created tokens like N1)
    source = """
    sym(zero, '0').
    add(zero, Y, Y).
    answer(N1, '+', N2, '=', A) :-
        sym(P1, N1),
        add(P1, zero, R),
        sym(R, A).
    """
    program, binding = parser.parse(source, empty_symbol_table(wam_config.frontend.ast), wam_config)
    # sanity: key atoms exist, and there are no corrupted symbols
    assert "answer" in binding.values()
    assert "sym" in binding.values()
    assert "add" in binding.values()
    assert "+" in binding.values()
    assert "=" in binding.values()
    assert not any(v.endswith(")") for v in binding.values())


@pytest.mark.parametrize(
    ("case_id", "source", "exception_type"),
    [
        pytest.param("empty_negation_operand", r"p :- \+.", ValueError),
        pytest.param("variable_negation_operand", r"p(X) :- \+ X.", ValueError),
    ],
    ids=[
        "empty_negation_operand",
        "variable_negation_operand",
    ],
)
def test_invalid_negation_source(case_id, source, exception_type, wam_config):
    with pytest.raises(exception_type):
        program, _ = parser.parse(source, empty_symbol_table(wam_config.frontend.ast), wam_config)
        compiled_program = init_compiled_program(wam_config)
        compile_program(
            program,
            compiled_program=compiled_program,
            compiler_state=init_compiler_state(wam_config),
            config=wam_config,
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        pytest.param("p([a, b]).", "Unknown functor", id="list_syntax"),
        pytest.param("p :- q ; r.", "Unknown functor", id="disjunction_syntax"),
    ],
)
def test_unsupported_syntax_is_rejected(source, message, wam_config):
    with pytest.raises(KeyError, match=message):
        parser.parse(source, empty_symbol_table(wam_config.frontend.ast), wam_config)


def test_parse_accepts_explicit_cut(wam_config):
    program, _ = parser.parse(
        "p :- !.",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    assert any(int(symbol) == int(CoreSymID.CUT) for symbol in program.node_symbols.flat)


def test_extract_symbols_ignores_true_builtin():
    clauses = [
        "p :- true.",
        "q(X) :- true, p(X).",
    ]

    symbols = parser._extract_symbols(clauses)

    assert "true" not in symbols
    assert "p" in symbols
    assert "q" in symbols


def test_parse_accepts_true_builtin(wam_config):
    program, binding = parser.parse(
        "p :- true.", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    # true must NOT be in bindings
    assert "true" not in binding.values()

    # p must still be bound
    assert "p" in binding.values()


def test_true_is_lowered_to_gid_true(wam_config):
    from wampy.frontend.ast.symbol_id import CoreSymID

    program, _ = parser.parse("p :- true.", empty_symbol_table(wam_config.frontend.ast), wam_config)

    assert program.node_symbols[0, 0] == CoreSymID.CLAUSE
    assert any(
        program.node_symbols[0, i] == CoreSymID.TRUE for i in range(program.node_symbols.shape[1])
    )
    assert all(program.node_symbols[tid, 0] != 209 for tid in range(get_term_count(program)))


def test_true_and_goal_both_present_in_ast(wam_config):
    from wampy.frontend.ast.program import NO_NODE
    from wampy.frontend.ast.symbol_id import CoreSymID

    program, binding = parser.parse(
        "p :- true, q.", empty_symbol_table(wam_config.frontend.ast), wam_config
    )

    clause_tid = next(
        tid
        for tid in range(get_term_count(program))
        if program.node_symbols[tid, 0] == CoreSymID.CLAUSE
    )

    def children(node_id):
        return [int(child) for child in program.node_links[clause_tid, node_id] if child != NO_NODE]

    head_id, body_id = children(0)
    assert int(program.node_symbols[clause_tid, head_id]) == int(binding.id_by_symbol["p"])
    assert children(head_id) == []

    assert int(program.node_symbols[clause_tid, body_id]) == int(CoreSymID.CONJUNCTION)
    first_goal, remaining_goals = children(body_id)
    assert int(program.node_symbols[clause_tid, first_goal]) == int(CoreSymID.TRUE)
    assert int(program.node_symbols[clause_tid, remaining_goals]) == int(CoreSymID.CONJUNCTION)

    second_goal, terminator = children(remaining_goals)
    assert int(program.node_symbols[clause_tid, second_goal]) == int(binding.id_by_symbol["q"])
    assert children(second_goal) == []
    assert int(program.node_symbols[clause_tid, terminator]) == int(CoreSymID.TRUE)


def test_parse_query_anonymous_vars_are_distinct(wam_config):
    _, binding = parser.parse("p(a, b).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    program, _ = parser.parse("?- p(_, _).", binding, wam_config)

    assert get_query_arity(program, 0) == 2
    s0 = int(program.node_symbols[0, int(get_query_argument(program, 0, 0))])
    s1 = int(program.node_symbols[0, int(get_query_argument(program, 0, 1))])
    assert program.first_variable_symbol_id <= s0 < program.first_user_symbol_id
    assert program.first_variable_symbol_id <= s1 < program.first_user_symbol_id
    assert s0 != s1


def test_parse_query_named_underscore_var_is_reused(wam_config):
    _, binding = parser.parse("p(a, b).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    program, _ = parser.parse("?- p(_Tmp, _Tmp).", binding, wam_config)

    assert get_query_arity(program, 0) == 2
    s0 = int(program.node_symbols[0, int(get_query_argument(program, 0, 0))])
    s1 = int(program.node_symbols[0, int(get_query_argument(program, 0, 1))])
    assert s0 == s1


def test_parse_query_quoted_zero_arity_predicate(wam_config):
    _, binding = parser.parse("'2'.", empty_symbol_table(wam_config.frontend.ast), wam_config)

    program, _ = parser.parse("?- '2'.", binding, wam_config)

    assert get_query_arity(program, 0) == 0
    assert get_query_functor(program, 0) == binding.id_by_symbol["2"]


def test_parse_query_interns_quoted_zero_arity_predicate(wam_config):
    _, binding = parser.parse(
        "p(a).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    assert "2" not in binding.id_by_symbol

    program, _ = parser.parse(
        "?- '2'.",
        binding,
        wam_config,
    )

    assert "2" in binding.id_by_symbol
    assert get_query_arity(program, 0) == 0
    assert get_query_functor(program, 0) == binding.id_by_symbol["2"]


def test_parse_query_parses_and_interns_quoted_zero_arity_predicate(wam_config):
    _, binding = parser.parse(
        "p(a).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    query, _ = parser.parse("?- '2'.", binding, wam_config)

    assert binding.id_by_symbol["2"] == get_query_functor(query, 0)
    assert get_query_arity(query, 0) == 0
