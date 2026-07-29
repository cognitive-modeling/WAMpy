from typing import Iterable, Mapping, NamedTuple

from numba import types
from numba.typed import Dict

from wampy.frontend.ast.symbol_id import SymID


class SymbolTable(NamedTuple):
    symbol_by_id: Dict
    id_by_symbol: Dict

    def keys(self):
        return self.symbol_by_id.keys()

    def values(self):
        return self.symbol_by_id.values()

    def items(self):
        return self.symbol_by_id.items()

    def get(self, symbol_id: int, default=None):
        return self.symbol_by_id.get(int(symbol_id), default)


def empty_symbol_table() -> SymbolTable:
    return SymbolTable(
        symbol_by_id=Dict.empty(key_type=types.int64, value_type=types.unicode_type),
        id_by_symbol=Dict.empty(key_type=types.unicode_type, value_type=types.int64),
    )


def build_symbol_table(symbols: Iterable[str]) -> SymbolTable:
    """
    Allocate symbol IDs and build a bidirectional symbol table backed by
    numba.typed.Dict for JIT compatibility.
    """
    symbol_table = empty_symbol_table()

    for symbol in symbols:
        intern_symbol(symbol_table, str(symbol))

    return symbol_table


def intern_symbol(symbol_table: SymbolTable, symbol: str) -> int:
    symbol_text = str(symbol)
    if symbol_text in symbol_table.id_by_symbol:
        return int(symbol_table.id_by_symbol[symbol_text])

    symbol_id = _next_available_symbol_id(symbol_table)
    symbol_table.symbol_by_id[symbol_id] = symbol_text
    symbol_table.id_by_symbol[symbol_text] = symbol_id
    return symbol_id


def symbol_table_from_mapping(symbol_by_id: Mapping[int, str]) -> SymbolTable:
    symbol_table = empty_symbol_table()
    for symbol_id, symbol in symbol_by_id.items():
        sid = int(symbol_id)
        symbol_text = str(symbol)
        symbol_table.symbol_by_id[sid] = symbol_text
        symbol_table.id_by_symbol[symbol_text] = sid
    return symbol_table


def merge_symbol_tables(target: SymbolTable, source: SymbolTable) -> SymbolTable:
    """
    Merge symbols from `source` into `target`.

    Existing symbols in `target` are preserved.
    """
    target_symbol_by_id = target.symbol_by_id
    target_id_by_symbol = target.id_by_symbol

    for source_symbol_id, source_symbol in source.symbol_by_id.items():
        symbol_text = str(source_symbol)
        if symbol_text in target_id_by_symbol:
            continue

        source_id = int(source_symbol_id)
        if source_id not in target_symbol_by_id:
            target_symbol_by_id[source_id] = symbol_text
            target_id_by_symbol[symbol_text] = source_id
            continue

        symbol_id = _next_available_symbol_id(target)
        target_symbol_by_id[symbol_id] = symbol_text
        target_id_by_symbol[symbol_text] = symbol_id

    return target


def _next_available_symbol_id(symbol_table: SymbolTable) -> int:
    for symbol_id in range(SymID.SYMBOL_FIRST, SymID.SYMBOL_LAST + 1):
        if symbol_id not in symbol_table.symbol_by_id:
            return symbol_id
    raise RuntimeError("No available symbol IDs.")
