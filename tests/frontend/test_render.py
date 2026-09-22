import numpy as np
import pytest
import wampy as wam
import wampy.frontend as frontend
from wampy.config import WAMConfig
from wampy.frontend.ast.ops.analysis import (
    get_node_symbol,
    get_term_count,
)
from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.frontend.parser import parse
from wampy.frontend.symbol_table import empty_symbol_table


@pytest.fixture(scope="session")
def wam_config():
    return WAMConfig()


def _first_clause_term_id(program):
    for term_id in range(get_term_count(program)):
        if get_node_symbol(program, term_id, 0) == CoreSymID.CLAUSE:
            return term_id

    raise AssertionError("No clause term found")


def test_frontend_exports_only_tree_and_jit():
    assert hasattr(frontend, "display_as_tree")
    assert hasattr(frontend, "display")
    assert hasattr(frontend, "render")
    assert not hasattr(frontend, "debug_program")
    assert not hasattr(frontend, "debug_program_as_tree")
    assert not hasattr(frontend, "debug_program_jit")


def test_wam_does_not_export_public_display_dispatcher():
    assert not hasattr(wam, "display")
    assert not hasattr(wam, "debug_program")
    assert not hasattr(wam, "debug_program_as_tree")
    assert not hasattr(wam, "debug_program_jit")


def test_display_as_tree_prints_clause(capsys, wam_config):
    program, symbol_table = parse("p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    clause_tid = _first_clause_term_id(program)

    frontend.display_as_tree(program, symbol_table=symbol_table, term_id=clause_tid)
    out = capsys.readouterr().out

    assert "p" in out
    assert "a" in out
    assert "└──" in out


def test_display_prints_clause_line(capsys, wam_config):
    program, symbol_table = parse("p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    clause_tid = _first_clause_term_id(program)

    frontend.display(program, symbol_table, term_id=clause_tid)
    out = capsys.readouterr().out

    assert "/* 000" in out
    assert "p(a)" in out


def test_render_returns_clause_line_without_printing(capsys, wam_config):
    program, symbol_table = parse("p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    clause_tid = _first_clause_term_id(program)

    text = frontend.render(program, symbol_table, term_id=clause_tid)

    assert text.startswith("/* 000")
    assert "p(a)" in text
    assert not text.endswith("\n")
    assert capsys.readouterr().out == ""


def test_display_parenthesizes_comma_terms_in_arguments(capsys, wam_config):
    program, symbol_table = parse(
        "p((a, true)). q((a, (b, c))) :- p((a, true)).",
        empty_symbol_table(wam_config.frontend.ast),
        wam_config,
    )

    frontend.display(program, symbol_table)
    out = capsys.readouterr().out

    assert "p((a, true))." in out
    assert "q((a, (b, c))) :- p((a, true))." in out


def test_display_shows_clause_without_name_marker_rows(capsys, wam_config):
    program, symbol_table = parse("p(a).", empty_symbol_table(wam_config.frontend.ast), wam_config)
    usage = np.zeros(program.node_symbols.shape[0], dtype=np.int64)
    usage[0] = 4

    frontend.display(program, symbol_table, term_usage=usage)
    out = capsys.readouterr().out

    assert "/* 0000 */  p(a)." in out
    assert "/* 0000 */  a." not in out
