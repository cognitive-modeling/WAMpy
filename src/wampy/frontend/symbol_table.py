from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np

from wampy.config import ASTConfig
from wampy.frontend.ast.symbol_id import calculate_symbol_id_positions


class SymbolTable(NamedTuple):
    symbol_by_id: dict[int, str]
    id_by_symbol: dict[str, int]

    # A one-element array keeps the counter mutable while preserving the
    # Numba-compatible NamedTuple used by the renderer and runtime helpers.
    next_user_symbol_id: np.ndarray

    first_user_symbol_id: int
    user_symbol_id_limit: int

    # These are Python convenience methods.
    # Numba code should access symbol_by_id directly.
    def keys(self):
        return self.symbol_by_id.keys()

    def values(self):
        return self.symbol_by_id.values()

    def items(self):
        return self.symbol_by_id.items()

    def get(self, symbol_id: int, default=None):
        return self.symbol_by_id.get(int(symbol_id), default)


def _resolve_layout(
    config: ASTConfig | None = None,
    first_user_symbol_id: int | None = None,
    user_symbol_id_limit: int | None = None,
) -> tuple[int, int]:
    if config is None:
        config = ASTConfig()

    _, default_first_user_symbol_id = calculate_symbol_id_positions(
        config.max_variables_per_term,
    )

    if first_user_symbol_id is None:
        first_user_symbol_id = default_first_user_symbol_id

    if user_symbol_id_limit is None:
        user_symbol_id_limit = first_user_symbol_id + config.max_user_symbols

    if first_user_symbol_id < 0:
        raise ValueError("first_user_symbol_id must be >= 0")

    if user_symbol_id_limit <= first_user_symbol_id:
        raise ValueError(
            "user_symbol_id_limit must exceed first_user_symbol_id",
        )

    return int(first_user_symbol_id), int(user_symbol_id_limit)


def _empty_symbol_table_resolved(
    first_user_symbol_id: int,
    user_symbol_id_limit: int,
) -> SymbolTable:
    """
    Low-level constructor suitable for both CPython and Numba.

    In CPython these are ordinary dicts.

    When compiled by Numba, the temporary entries establish:

        symbol_by_id: int64 -> unicode
        id_by_symbol: unicode -> int64

    before clear() makes them empty again.
    """

    symbol_by_id = {
        np.int64(0): "",
    }
    symbol_by_id.clear()

    id_by_symbol = {
        "": np.int64(0),
    }
    id_by_symbol.clear()

    return SymbolTable(
        symbol_by_id=symbol_by_id,
        id_by_symbol=id_by_symbol,
        next_user_symbol_id=np.array(
            [first_user_symbol_id],
            dtype=np.int64,
        ),
        first_user_symbol_id=int(first_user_symbol_id),
        user_symbol_id_limit=int(user_symbol_id_limit),
    )


def empty_symbol_table(
    config: ASTConfig | None = None,
    *,
    first_user_symbol_id: int | None = None,
    user_symbol_id_limit: int | None = None,
) -> SymbolTable:
    first_user_symbol_id, user_symbol_id_limit = _resolve_layout(
        config,
        first_user_symbol_id,
        user_symbol_id_limit,
    )

    return _empty_symbol_table_resolved(
        first_user_symbol_id,
        user_symbol_id_limit,
    )


def build_symbol_table(
    symbols: Iterable[str],
    max_symbol_id: int | None = None,
    config: ASTConfig | None = None,
    *,
    first_user_symbol_id: int | None = None,
    user_symbol_id_limit: int | None = None,
) -> SymbolTable:
    """Allocate symbols using the configured user-symbol range."""

    # Keep accepting build_symbol_table(symbols, ast_config) while the
    # second positional argument remains the legacy dtype upper bound.
    if isinstance(max_symbol_id, ASTConfig):
        if config is not None:
            raise TypeError("Specify the AST config only once")

        config = max_symbol_id
        max_symbol_id = None

    symbol_table = empty_symbol_table(
        config,
        first_user_symbol_id=first_user_symbol_id,
        user_symbol_id_limit=user_symbol_id_limit,
    )

    for symbol in symbols:
        intern_symbol(
            symbol_table,
            str(symbol),
            max_symbol_id=max_symbol_id,
        )

    return symbol_table


def intern_symbol(
    symbol_table: SymbolTable,
    symbol: str,
    max_symbol_id: int | None = None,
) -> int:
    """
    Intern an already-normalized string.

    Keep this function Numba-compatible. Conversion to str() belongs at
    Python-facing call sites such as build_symbol_table().
    """

    if symbol in symbol_table.id_by_symbol:
        return int(symbol_table.id_by_symbol[symbol])

    symbol_id = _allocate_next_user_symbol_id(
        symbol_table,
        max_symbol_id,
    )

    symbol_table.symbol_by_id[symbol_id] = symbol
    symbol_table.id_by_symbol[symbol] = symbol_id
    symbol_table.next_user_symbol_id[0] = symbol_id + 1

    return symbol_id


def symbol_table_from_mapping(
    symbol_by_id: Mapping[int, str],
    config: ASTConfig | None = None,
    *,
    first_user_symbol_id: int | None = None,
    user_symbol_id_limit: int | None = None,
) -> SymbolTable:
    symbol_table = empty_symbol_table(
        config,
        first_user_symbol_id=first_user_symbol_id,
        user_symbol_id_limit=user_symbol_id_limit,
    )

    next_user_symbol_id = symbol_table.first_user_symbol_id

    for symbol_id, symbol in symbol_by_id.items():
        sid = int(symbol_id)
        symbol_text = str(symbol)

        symbol_table.symbol_by_id[sid] = symbol_text
        symbol_table.id_by_symbol[symbol_text] = sid

        if sid >= next_user_symbol_id:
            next_user_symbol_id = sid + 1

    symbol_table.next_user_symbol_id[0] = next_user_symbol_id

    return symbol_table


def _check_layout_compatibility(
    target: SymbolTable,
    source: SymbolTable,
) -> None:
    if (
        target.first_user_symbol_id != source.first_user_symbol_id
        or target.user_symbol_id_limit != source.user_symbol_id_limit
    ):
        raise ValueError(
            "Cannot combine symbol tables created with different user-symbol layouts",
        )


def merge_symbol_tables(
    target: SymbolTable,
    source: SymbolTable,
    max_symbol_id: int | None = None,
) -> SymbolTable:
    """Merge symbols from source into target."""

    _check_layout_compatibility(target, source)

    target_symbol_by_id = target.symbol_by_id
    target_id_by_symbol = target.id_by_symbol

    for source_symbol_id, source_symbol in source.symbol_by_id.items():
        # Do not call str() here. Keeping the value as unicode makes this
        # function usable from nopython code as well.
        symbol_text = source_symbol

        if symbol_text in target_id_by_symbol:
            continue

        source_id = int(source_symbol_id)

        source_id_is_allocated_user_id = (
            target.first_user_symbol_id <= source_id < target.user_symbol_id_limit
        )

        if source_id_is_allocated_user_id and source_id not in target_symbol_by_id:
            if max_symbol_id is not None and source_id > max_symbol_id:
                source_id_is_allocated_user_id = False
            else:
                target_symbol_by_id[source_id] = symbol_text
                target_id_by_symbol[symbol_text] = source_id

                if source_id >= int(
                    target.next_user_symbol_id[0],
                ):
                    target.next_user_symbol_id[0] = source_id + 1

                continue

        symbol_id = _allocate_next_user_symbol_id(
            target,
            max_symbol_id,
        )

        target_symbol_by_id[symbol_id] = symbol_text
        target_id_by_symbol[symbol_text] = symbol_id
        target.next_user_symbol_id[0] = symbol_id + 1

    return target


def _allocate_next_user_symbol_id(
    symbol_table: SymbolTable,
    max_symbol_id: int | None = None,
) -> int:
    symbol_id = max(
        symbol_table.first_user_symbol_id,
        int(symbol_table.next_user_symbol_id[0]),
    )

    exclusive_limit = symbol_table.user_symbol_id_limit

    if max_symbol_id is not None:
        exclusive_limit = min(
            exclusive_limit,
            int(max_symbol_id) + 1,
        )

    while symbol_id in symbol_table.symbol_by_id:
        symbol_id += 1

    if symbol_id >= exclusive_limit:
        raise RuntimeError(
            "No available user symbol IDs for the configured symbol range.",
        )

    return symbol_id
