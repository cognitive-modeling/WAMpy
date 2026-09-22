import numpy as np
import pytest
import wampy
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import get_child_at, get_node_symbol, get_term_count
from wampy.frontend.ast.program import init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.parser import loader
from wampy.frontend.symbol_table import empty_symbol_table

SOURCE = """
% WAMpy owns comments, clause splitting, and multiline syntax.
parent(alice, bob).

grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
"""


def test_load_prolog_parses_once_and_loads_into_existing_program():
    config = WAMConfig()
    program = init_ast_program(config)
    symbol_table = empty_symbol_table(config.frontend.ast)
    program.term_count[0] = 2

    original_parse = loader.parse
    calls = []

    def counted_parse(*args, **kwargs):
        calls.append(args[0])
        return original_parse(*args, **kwargs)

    # load_prolog must pass the complete source to WAMpy's parser once.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(loader, "parse", counted_parse)
        imported = wampy.load_prolog(
            SOURCE,
            program,
            symbol_table,
            config,
            start_term_id=2,
        )

    assert calls == [SOURCE]
    assert imported == [2, 3]
    assert get_term_count(program) == 4
    assert program.node_symbols[0, 0] == CoreSymID.NO_SYMBOL
    assert program.node_symbols[1, 0] == CoreSymID.NO_SYMBOL

    parent_id = symbol_table.id_by_symbol["parent"]
    parent_head_id = get_child_at(program, 2, 0, 0)
    assert get_node_symbol(program, 2, parent_head_id) == parent_id


def test_load_prolog_preserves_parser_ast_and_symbol_identity():
    config = WAMConfig()
    parsed_symbols = empty_symbol_table(config.frontend.ast)
    parsed_program, _ = wampy.parse(SOURCE, parsed_symbols, config)

    destination_symbols = empty_symbol_table(config.frontend.ast)
    destination_program = init_ast_program(config)
    imported = wampy.load_prolog(
        SOURCE,
        destination_program,
        destination_symbols,
        config,
        start_term_id=5,
    )

    assert imported == [5, 6]
    assert destination_symbols.id_by_symbol == parsed_symbols.id_by_symbol
    for source_term, destination_term in zip(range(2), imported, strict=True):
        assert np.array_equal(
            parsed_program.node_symbols[source_term],
            destination_program.node_symbols[destination_term],
        )
        assert np.array_equal(
            parsed_program.node_links[source_term],
            destination_program.node_links[destination_term],
        )
        assert np.array_equal(
            parsed_program.node_arities[source_term],
            destination_program.node_arities[destination_term],
        )
        assert np.array_equal(
            parsed_program.variable_name_ids[source_term],
            destination_program.variable_name_ids[destination_term],
        )


def test_copy_term_between_programs_is_public_ast_operation():
    config = WAMConfig()
    source_symbols = empty_symbol_table(config.frontend.ast)
    source, _ = wampy.parse("p(a).", source_symbols, config)
    destination = init_ast_program(config)

    wampy.ast.copy_term_between_programs(source, 0, destination, 3)

    assert callable(wampy.ast.copy_term_between_programs)
    assert np.array_equal(source.node_symbols[0], destination.node_symbols[3])
    assert np.array_equal(source.node_links[0], destination.node_links[3])
