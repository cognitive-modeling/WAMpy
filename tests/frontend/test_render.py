import warnings

import numpy as np
import pytest

import wampy as wam
import wampy.frontend as frontend
from minimalist_machine.core.model import Model, display_program
from wampy.config import load_config_toml
from wampy.frontend.ast.symbol_id import SymID
from wampy.frontend.parser import build_new_program

CONFIG_TOML = """
[ast]
num_terms = 12
max_terms_nodes = 40
max_terms_nodes_childs = 5
"""


@pytest.fixture(scope="session")
def wam_config():
    return load_config_toml(CONFIG_TOML)


def _first_clause_term_id(program):
    for tid in range(program.symbol.shape[0]):
        if int(program.symbol[tid, 0]) == int(SymID.CLAUSE):
            return tid
    raise AssertionError("Expected at least one clause term")


def test_frontend_exports_only_tree_and_jit():
    assert hasattr(frontend, "display_as_tree")
    assert hasattr(frontend, "display")
    assert not hasattr(frontend, "debug_program")
    assert not hasattr(frontend, "debug_program_as_tree")
    assert not hasattr(frontend, "debug_program_jit")

    with pytest.raises(ImportError):
        from wampy.frontend import debug_program  # noqa: F401


def test_wam_exports_only_jit_printer():
    assert hasattr(wam, "display")
    assert not hasattr(wam, "debug_program")
    assert not hasattr(wam, "debug_program_as_tree")
    assert not hasattr(wam, "debug_program_jit")


def test_display_as_tree_prints_clause(capsys, wam_config):
    program, symbol_table = build_new_program("p(a).", wam_config)
    clause_tid = _first_clause_term_id(program)

    frontend.display_as_tree(program, symbol_table=symbol_table, term_id=clause_tid)
    out = capsys.readouterr().out

    assert "p" in out
    assert "a" in out
    assert "└──" in out


def test_display_prints_clause_line(capsys, wam_config):
    program, symbol_table = build_new_program("p(a).", wam_config)
    clause_tid = _first_clause_term_id(program)

    frontend.display(program, symbol_table, term_id=clause_tid)
    out = capsys.readouterr().out

    assert "/* 000" in out
    assert "p(a)" in out


def test_display_parenthesizes_comma_terms_in_arguments(capsys, wam_config):
    program, symbol_table = build_new_program("p((a, true)). q((a, (b, c))) :- p((a, true)).", wam_config)

    frontend.display(program, symbol_table)
    out = capsys.readouterr().out

    assert "p((a, true)) :- true." in out
    assert "q((a, (b, c))) :- p((a, true)), true." in out


def test_display_program_warns_for_ignored_args(capsys, wam_config):
    program, symbol_table = build_new_program("p(a).", wam_config)
    model = Model(settings=None, program_state=program, symbol_table=symbol_table)

    with pytest.warns(UserWarning, match="multihead_mode"):
        display_program(model, multihead_mode="split")
    with pytest.warns(UserWarning, match="print_every_term"):
        display_program(model, print_every_term=False)

    out = capsys.readouterr().out
    assert "/* 000" in out


def test_display_program_default_has_no_warning(capsys, wam_config):
    program, symbol_table = build_new_program("p(a).", wam_config)
    model = Model(settings=None, program_state=program, symbol_table=symbol_table)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        display_program(model)

    assert len(caught) == 0
    out = capsys.readouterr().out
    assert "/* 000" in out


def test_display_shows_clause_without_name_marker_rows(capsys, wam_config):
    program, symbol_table = build_new_program("p(a).", wam_config)
    usage = np.zeros(program.symbol.shape[0], dtype=np.int64)
    usage[0] = 4

    frontend.display(program, symbol_table, term_usage=usage)
    out = capsys.readouterr().out

    assert "/* 0000 */  p(a) :- true." in out
    assert "/* 0000 */  a." not in out
