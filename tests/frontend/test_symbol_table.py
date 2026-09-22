import pytest
import wampy as wam
from wampy.frontend.symbol_table import (
    SymbolTable,
    build_symbol_table,
    intern_symbol,
)


def test_build_symbol_table_returns_bidirectional_maps():
    table = build_symbol_table(("a", "p", "q"))

    assert isinstance(table, SymbolTable)
    assert SymbolTable._fields == (
        "symbol_by_id",
        "id_by_symbol",
        "next_user_symbol_id",
        "first_user_symbol_id",
        "user_symbol_id_limit",
    )
    assert isinstance(table.symbol_by_id, dict)
    assert isinstance(table.id_by_symbol, dict)
    assert set(table.symbol_by_id.values()) == {"a", "p", "q"}
    assert set(table.id_by_symbol.keys()) == {"a", "p", "q"}

    for sid, symbol in table.symbol_by_id.items():
        assert table.id_by_symbol[symbol] == int(sid)
    assert int(table.next_user_symbol_id[0]) == table.first_user_symbol_id + 3


def test_symbol_allocation_uses_the_selected_upper_limit():
    table = build_symbol_table(
        ("a", "p"),
        max_symbol_id=49,
    )

    assert int(table.id_by_symbol["a"]) == table.first_user_symbol_id
    assert int(table.id_by_symbol["p"]) == table.first_user_symbol_id + 1

    with pytest.raises(RuntimeError, match="configured symbol range"):
        intern_symbol(table, "q", max_symbol_id=49)


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
    )
    from wampy.frontend.symbol_table import (
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


# @jit
# def _jit_lookup_symbol(symbol_by_id, sid: int) -> str:
#     if sid in symbol_by_id:
#         return symbol_by_id[sid]
#     return ""


# @jit
# def _jit_lookup_id(id_by_symbol, symbol: str) -> int:
#     if symbol in id_by_symbol:
#         return int(id_by_symbol[symbol])
#     return -1


# @jit
# def _jit_lookup_symbol_table(symbol_table: SymbolTable, sid: int) -> str:
#     return symbol_for_id_jit(symbol_table, sid)


# @pytest.mark.requires_numba_jit
# def test_build_symbol_table_maps_are_jit_compatible():
#     table = build_symbol_table(("alpha", "beta"))

#     alpha_id = int(table.id_by_symbol["alpha"])
#     beta_id = int(table.id_by_symbol["beta"])

#     assert _jit_lookup_symbol(table.symbol_by_id, alpha_id) == "alpha"
#     assert _jit_lookup_id(table.id_by_symbol, "beta") == beta_id
#     assert _jit_lookup_symbol_table(table, alpha_id) == "alpha"
#     assert _jit_lookup_symbol.nopython_signatures
#     assert _jit_lookup_id.nopython_signatures
#     assert _jit_lookup_symbol_table.nopython_signatures
