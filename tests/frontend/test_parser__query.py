import numpy as np
import wampy.frontend.parser as parser
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.ast.ops.analysis import (
    get_query_argument,
    get_query_arity,
    get_query_functor,
    get_term_count,
)
from wampy.frontend.ast.program import ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.symbol_table import empty_symbol_table


def _symbol_table():
    _, symbol_table = parser.parse(
        "p(a). q(b).", empty_symbol_table(DEFAULT_CONFIG.frontend.ast), DEFAULT_CONFIG
    )
    return symbol_table


def test_parse_explicit_ground_query():
    symbol_table = _symbol_table()

    program, returned_table = parser.parse("?- p(a).", symbol_table, DEFAULT_CONFIG)

    assert isinstance(program, ASTProgram)
    assert returned_table is symbol_table
    assert int(program.node_symbols[0, 0]) == int(CoreSymID.QUERY)
    assert get_query_arity(program, 0) == 1
    assert int(get_query_functor(program, 0)) == int(symbol_table.id_by_symbol["p"])
    argument = get_query_argument(program, 0, 0)
    assert int(program.node_symbols[0, int(argument)]) == int(symbol_table.id_by_symbol["a"])


def test_parse_explicit_query_with_variables_and_metadata():
    symbol_table = _symbol_table()
    variable_names_by_term = []

    program, _ = parser.parse(
        "?- p(X).",
        symbol_table,
        DEFAULT_CONFIG,
        variable_names_by_term=variable_names_by_term,
    )
    argument = get_query_argument(program, 0, 0)
    argument_symbol = int(program.node_symbols[0, int(argument)])

    assert get_query_arity(program, 0) == 1
    assert program.first_variable_symbol_id <= argument_symbol < program.first_user_symbol_id
    assert variable_names_by_term == [{"X": argument_symbol}]


def test_parse_stores_query_variable_name_ids_in_the_ast():
    symbol_table = _symbol_table()

    program, symbol_table = parser.parse(
        "?- p(X, Y).",
        symbol_table,
        DEFAULT_CONFIG,
    )

    assert int(program.variable_name_ids[0, 0]) == symbol_table.id_by_symbol["X"]
    assert int(program.variable_name_ids[0, 1]) == symbol_table.id_by_symbol["Y"]
    assert int(program.variable_name_ids[0, 2]) == np.iinfo(program.variable_name_ids.dtype).max


def test_parse_does_not_name_anonymous_variables():
    symbol_table = _symbol_table()

    program, symbol_table = parser.parse(
        "?- p(_, X).",
        symbol_table,
        DEFAULT_CONFIG,
    )

    invalid_symbol_id = np.iinfo(program.variable_name_ids.dtype).max
    assert int(program.variable_name_ids[0, 0]) == invalid_symbol_id
    assert int(program.variable_name_ids[0, 1]) == symbol_table.id_by_symbol["X"]


def test_same_variable_spelling_in_different_terms_keeps_local_variable_slots():
    program, symbol_table = parser.parse(
        "p(X). q(X).",
        empty_symbol_table(DEFAULT_CONFIG.frontend.ast),
        DEFAULT_CONFIG,
    )

    assert symbol_table.id_by_symbol["X"] in symbol_table.symbol_by_id
    assert int(program.variable_name_ids[0, 0]) == symbol_table.id_by_symbol["X"]
    assert int(program.variable_name_ids[1, 0]) == symbol_table.id_by_symbol["X"]

    variable_ids = []
    for term_id in range(get_term_count(program)):
        variable_ids.append(
            sorted(
                int(symbol_id)
                for symbol_id in program.node_symbols[term_id]
                if program.first_variable_symbol_id <= int(symbol_id) < program.first_user_symbol_id
            )
        )

    assert variable_ids == [[program.first_variable_symbol_id]] * 2


def test_parse_explicit_conjunction_query():
    symbol_table = _symbol_table()

    program, _ = parser.parse("?- p(X), q(X).", symbol_table, DEFAULT_CONFIG)

    query_root = int(program.node_links[0, 0, 0])
    assert int(program.node_symbols[0, 0]) == int(CoreSymID.QUERY)
    assert int(program.node_symbols[0, query_root]) == int(CoreSymID.CONJUNCTION)


def test_parse_mixed_clauses_and_queries_preserves_term_kinds():
    source = """
    p(a).
    ?- p(X).
    q(b).
    ?- q(Y).
    """

    program, _ = parser.parse(
        source,
        empty_symbol_table(DEFAULT_CONFIG.frontend.ast),
        DEFAULT_CONFIG,
    )

    assert get_term_count(program) == 4
    assert [int(program.node_symbols[i, 0]) for i in range(4)] == [
        int(CoreSymID.CLAUSE),
        int(CoreSymID.QUERY),
        int(CoreSymID.CLAUSE),
        int(CoreSymID.QUERY),
    ]


def test_parse_resizes_for_query_batches():
    symbol_table = _symbol_table()
    config = DEFAULT_CONFIG._replace(
        frontend=DEFAULT_CONFIG.frontend._replace(
            ast=DEFAULT_CONFIG.frontend.ast._replace(max_terms=1),
        ),
    )

    program, _ = parser.parse(
        "?- p(a).\n?- q(b).",
        symbol_table,
        config,
    )

    assert program.node_symbols.shape[0] == 2
    assert get_term_count(program) == 2


def test_parse_explicit_query_interns_query_only_symbols():
    program, symbol_table = parser.parse(
        "p(a).", empty_symbol_table(DEFAULT_CONFIG.frontend.ast), DEFAULT_CONFIG
    )

    query_program, returned_table = parser.parse("?- p(b).", symbol_table, DEFAULT_CONFIG)

    assert query_program is not program
    assert returned_table is symbol_table
    assert "b" in symbol_table.values()
    assert "p" in symbol_table.values()


def test_parse_is_the_only_query_parsing_entry_point():
    for removed_name in (
        "build_query",
        "build_queries",
        "build_query_with_metadata",
        "ensure_query_symbols",
    ):
        assert not hasattr(parser, removed_name)
