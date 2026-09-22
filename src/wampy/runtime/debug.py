from wampy.frontend.symbol_table import SymbolTable
from wampy.runtime.machine.heap import deref
from wampy.runtime.tags import TAG


def record_path_usage(state, usage_row) -> None:
    """Record traced clause-term usage for runtime diagnostics."""

    for index in range(usage_row.shape[0]):
        usage_row[index] = 0

    depth = int(state.path_depth[0])
    for index in range(depth):
        term_id = state.path_terms[index]
        if 0 <= term_id < usage_row.shape[0]:
            usage_row[term_id] += 1


def format_heap_term(
    state,
    memory,
    address: int,
    symbol_table: SymbolTable,
) -> str:
    """Format a live WAM heap term for runtime diagnostics."""

    address = int(deref(state, memory, address))
    heap = memory.heap
    tag = heap.tags[address]

    if tag == TAG.REF and heap.cells[address] == address:
        return "_"

    if tag == TAG.CON:
        symbol_id = int(heap.cells[address])
        return symbol_table.get(symbol_id, str(symbol_id))

    if tag == TAG.STR:
        functor_address = int(heap.cells[address])
        symbol_id = int(heap.cells[functor_address])
        arity = int(heap.cells[functor_address + 1])

        symbol = symbol_table.get(symbol_id, str(symbol_id))
        arguments = [
            format_heap_term(
                state,
                memory,
                address + 1 + index,
                symbol_table,
            )
            for index in range(arity)
        ]

        return f"{symbol}({', '.join(arguments)})"

    return "<?>"
