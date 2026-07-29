import pytest
from numba import jit
from numba.typed import Dict

import wampy as wam
from wampy.frontend.symbol_table import (
    SymbolTable,
    build_symbol_table,
    intern_symbol,
)


@pytest.mark.requires_numba_jit
def test_build_symbol_table_returns_bidirectional_maps():
    table = build_symbol_table(("a", "p", "q"))

    assert isinstance(table, SymbolTable)
    assert SymbolTable._fields == ("symbol_by_id", "id_by_symbol")
    assert isinstance(table.symbol_by_id, Dict)
    assert isinstance(table.id_by_symbol, Dict)
    assert set(table.symbol_by_id.values()) == {"a", "p", "q"}
    assert set(table.id_by_symbol.keys()) == {"a", "p", "q"}

    for sid, symbol in table.symbol_by_id.items():
        assert table.id_by_symbol[symbol] == int(sid)


def test_intern_symbol_reuses_existing_symbol_id():
    table = build_symbol_table(("a",))

    first = intern_symbol(table, "p")
    second = intern_symbol(table, "p")

    assert first == second
    assert table.symbol_by_id[first] == "p"
    assert table.id_by_symbol["p"] == first


def test_import_path_symbol_table_only():
    from wampy.frontend.symbol_table import (
        SymbolTable as SymbolTableNew,
        build_symbol_table as build_symbol_table_new,
    )

    table = build_symbol_table_new(("x", "y"))

    assert isinstance(table, SymbolTableNew)
    assert set(table.symbol_by_id.values()) == {"x", "y"}


def test_wam_exports_symbol_table_api():
    assert hasattr(wam, "SymbolTable")
    assert hasattr(wam, "build_symbol_table")

    table = wam.build_symbol_table(("n", "m"))
    assert isinstance(table, wam.SymbolTable)


@jit
def _jit_lookup_symbol(symbol_by_id, sid: int) -> str:
    if sid in symbol_by_id:
        return symbol_by_id[sid]
    return ""


@jit
def _jit_lookup_id(id_by_symbol, symbol: str) -> int:
    if symbol in id_by_symbol:
        return int(id_by_symbol[symbol])
    return -1


@pytest.mark.requires_numba_jit
def test_build_symbol_table_maps_are_jit_compatible():
    table = build_symbol_table(("alpha", "beta"))

    alpha_id = int(table.id_by_symbol["alpha"])
    beta_id = int(table.id_by_symbol["beta"])

    assert _jit_lookup_symbol(table.symbol_by_id, alpha_id) == "alpha"
    assert _jit_lookup_id(table.id_by_symbol, "beta") == beta_id
    assert _jit_lookup_symbol.nopython_signatures
    assert _jit_lookup_id.nopython_signatures
