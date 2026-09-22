"""Solution values and WAM-term decoding for the WAMpy frontend.

decode_term       heap address -> Python value
decode_variables  successful Machine -> [value, ...]
decode_bindings   successful Machine -> {variable_name: value}
"""

from wampy.frontend.ast.symbol_id import CoreSymID
from wampy.runtime.machine import Machine
from wampy.runtime.machine.heap import deref
from wampy.runtime.term_decode import read_term

type SolutionType = dict[str, object]

# Runtime compatibility: `Solution` remains the built-in dict class.
Solution = dict


def _symbol_name(symbol_table, symbol_id: int) -> str:
    if symbol_id == int(CoreSymID.TRUE):
        return "true"
    if symbol_id == int(CoreSymID.NEGATION_AS_FAILURE):
        return "not"
    if symbol_id == int(CoreSymID.CUT):
        return "!"
    if symbol_id == int(CoreSymID.CONJUNCTION):
        return ","
    return str(symbol_table.symbol_by_id.get(int(symbol_id), f"<SYM:{symbol_id}>"))


def _format_decoded_term(term, symbol_table, unresolved_names):
    kind = term[0]
    if kind == "VAR":
        return unresolved_names.get(int(term[1]), f"_G{int(term[1])}")
    if kind == "CONST":
        return _symbol_name(symbol_table, int(term[1]))

    functor = _symbol_name(symbol_table, int(term[1]))
    children = [_format_decoded_term(child, symbol_table, unresolved_names) for child in term[2]]
    if functor == "," and len(children) == 2:
        return f"({children[0]}, {children[1]})"
    return f"{functor}({', '.join(map(str, children))})"


def decode_variables(
    machine: Machine,
    compiled_query,
    symbol_table,
) -> list[object]:
    """Decode all valid query-variable registers without source names."""

    values = []

    for register in compiled_query.variable_registers:
        register = int(register)

        if register < 0:
            continue

        address = int(machine.X[register])

        values.append(decode_term(machine, address, symbol_table))

    return values


def decode_term(
    machine: Machine,
    address: int,
    symbol_table,
    unresolved_names: dict[int, str] | None = None,
) -> object:
    """Decode one WAM heap term into a Python-owned value."""

    term = read_term(machine.registers, machine, int(address))

    return _format_decoded_term(
        term,
        symbol_table,
        unresolved_names or {},
    )


def decode_bindings(
    machine: Machine,
    compiled_query,
    symbol_table,
) -> SolutionType:
    """Decode named query-variable bindings from a successful machine state."""

    state = machine.registers
    variable_registers = compiled_query.variable_registers
    variable_name_ids = compiled_query.variable_name_ids

    unresolved_names: dict[int, str] = {}
    roots_by_name: list[tuple[str, int]] = []

    for slot, register_value in enumerate(variable_registers):
        register = int(register_value)

        if register < 0:
            continue

        name_id = int(variable_name_ids[slot])
        if name_id not in symbol_table.symbol_by_id:
            continue

        name = str(symbol_table.symbol_by_id[name_id])
        root = int(machine.X[register])
        resolved = int(deref(state, machine, root))

        unresolved_names.setdefault(resolved, name)
        roots_by_name.append((name, root))

    result: SolutionType = {}

    for name, root in roots_by_name:
        result[name] = decode_term(
            machine,
            root,
            symbol_table,
            unresolved_names,
        )

    return result
