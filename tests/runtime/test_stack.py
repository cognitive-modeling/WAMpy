import numpy as np
from wampy.config import DEFAULT_CONFIG
from wampy.frontend.symbol_table import empty_symbol_table
from wampy.runtime.debug import format_heap_term
from wampy.runtime.machine import (
    ChoicePointKind,
    ChoiceSlot,
    Machine,
    Registers,
    Stack,
    init_machine,
)
from wampy.runtime.machine.heap import (
    deref,
    make_const,
    make_functor,
    make_structure,
    make_var,
)
from wampy.runtime.machine.trail import trail_var
from wampy.runtime.tags import TAG


def test_init_machine_parts():
    """Machine state and memory initialization."""
    memory = init_machine()
    state = memory.registers

    assert state.H[0] == 0
    assert state.TR[0] == 0
    assert state.B[0] == -1
    assert state.pdl_top[0] == 0
    assert memory.stack.max_x_registers == DEFAULT_CONFIG.runtime.max_x_registers
    assert memory.stack.cells.shape[0] > DEFAULT_CONFIG.runtime.environment_size
    assert state.local_top[0] == 0
    assert memory.pdl.structure_S.shape[0] == DEFAULT_CONFIG.runtime.max_structure_depth
    assert state.step_limit == DEFAULT_CONFIG.runtime.max_steps
    assert state.unify_step_limit == DEFAULT_CONFIG.runtime.max_unify_steps

    # All heap tags initialized as REF
    assert np.all(memory.heap.tags == TAG.REF)


def test_machine_named_tuple_layouts():
    assert tuple(slot.name for slot in ChoiceSlot) == (
        "TR",
        "BP",
        "CP",
        "CE",
        "PREVIOUS_B",
        "H",
        "B0",
        "DEPTH",
        "RETURN_DEPTH",
        "KIND",
        "A",
    )
    assert tuple(int(slot) for slot in ChoiceSlot) == tuple(range(11))
    assert tuple(kind.name for kind in ChoicePointKind) == (
        "NORMAL",
        "NAF",
        "NAF_TRUNCATED",
    )
    assert tuple(int(kind) for kind in ChoicePointKind) == (0, 1, 2)
    assert Machine._fields == (
        "registers",
        "X",
        "heap",
        "stack",
        "trail",
        "pdl",
        "depth",
        "depth_limit",
        "return_depth",
    )
    assert Registers._fields[:11] == (
        "P",
        "CP",
        "E",
        "B",
        "B0",
        "H",
        "HB",
        "TR",
        "S",
        "mode",
        "local_top",
    )
    assert Stack._fields == (
        "cells",
        "choice_point_frame_size",
        "max_x_registers",
    )
    assert hasattr(Machine, "heap")


# -------------------------------------------------
# Variable creation and dereferencing
# -------------------------------------------------


def test_make_var_creates_self_reference():
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)

    assert memory.heap.cells[x] == x
    assert memory.heap.tags[x] == TAG.REF
    assert state.H[0] == 1


def test_deref_unbound_variable():
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    assert deref(state, memory, x) == x


def test_deref_reference_chain():
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)

    # Make x point to y
    memory.heap.cells[x] = y

    assert deref(state, memory, x) == y
    assert deref(state, memory, y) == y


def test_deref_reference_cycle_returns_invalid_address():
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)
    memory.heap.cells[x] = y
    memory.heap.cells[y] = x

    assert deref(state, memory, x) == state.H[0]


# -------------------------------------------------
# Constants
# -------------------------------------------------


def test_make_const():
    memory = init_machine()
    state = memory.registers

    c = make_const(state, memory, 42)

    assert memory.heap.cells[c] == 42
    assert memory.heap.tags[c] == TAG.CON
    assert state.H[0] == 1


# -------------------------------------------------
# Functors and structures
# -------------------------------------------------


def test_make_functor():
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=1, arity=2)

    assert memory.heap.tags[f] == TAG.FUN
    assert memory.heap.cells[f] == 1
    assert memory.heap.cells[f + 1] == 2
    assert state.H[0] == 2


def test_make_structure_with_variables():
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)

    f = make_functor(state, memory, symbol_id=1, arity=2)
    s = make_structure(state, memory, f, np.array([x, y], dtype=np.uint16))

    assert memory.heap.tags[s] == TAG.STR
    assert memory.heap.cells[s] == f
    assert state.H[0] == 2 + 2 + 1 + 2  # vars + functor + structure header + args


# -------------------------------------------------
# Dereferencing edge cases
# -------------------------------------------------


def test_deref_long_reference_chain():
    """
    Edge case: dereferencing a deep REF → REF → ... chain.

    Ensures deref follows references transitively and terminates
    at the final self-referential variable.
    """
    memory = init_machine()
    state = memory.registers

    vars_ = [make_var(state, memory) for _ in range(10)]

    for i in range(9):
        memory.heap.cells[vars_[i]] = vars_[i + 1]

    assert deref(state, memory, vars_[0]) == vars_[9]


def test_deref_stops_at_constant():
    """
    Edge case: dereferencing a variable bound directly to a constant.

    deref must stop when a non-REF tag is encountered.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    c = make_const(state, memory, 99)

    memory.heap.cells[x] = c

    assert deref(state, memory, x) == c
    assert memory.heap.tags[c] == TAG.CON


# -------------------------------------------------
# Structure construction edge cases
# -------------------------------------------------


def test_zero_arity_functor_structure():
    """
    Edge case: zero-arity functor (atom-like structure).

    A structure cell is still allocated even with no arguments.
    """
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=1, arity=0)
    s = make_structure(state, memory, f, np.empty(0, dtype=np.uint16))

    assert memory.heap.tags[s] == TAG.STR
    assert memory.heap.cells[s] == f


def test_structure_keeps_raw_argument_addresses():
    """
    Edge case: structure arguments are stored verbatim.

    Arguments are not dereferenced at construction time.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)

    memory.heap.cells[x] = y  # x → y

    f = make_functor(state, memory, 1, 1)
    s = make_structure(state, memory, f, np.array([x], dtype=np.uint16))

    arg_cell = s + 1
    assert memory.heap.cells[arg_cell] == x


# -------------------------------------------------
# Heap behavior invariants
# -------------------------------------------------


def test_heap_top_monotonicity():
    """
    Invariant: heap_top must grow monotonically.
    """
    memory = init_machine()
    state = memory.registers

    tops = []
    for _ in range(5):
        make_var(state, memory)
        tops.append(int(state.H[0]))

    assert tops == sorted(tops)


def test_multiple_structures_do_not_overlap():
    """
    Invariant: creating multiple structures must not overwrite
    earlier heap regions.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    f = make_functor(state, memory, 1, 1)

    s1 = make_structure(state, memory, f, np.array([x], dtype=np.uint16))
    top_after_s1 = int(state.H[0])

    s2 = make_structure(state, memory, f, np.array([x], dtype=np.uint16))

    assert memory.heap.tags[s1] == TAG.STR
    assert s2 >= top_after_s1


# -------------------------------------------------
# Debug printer edge cases
# -------------------------------------------------


def test_debug_nested_structure():
    """
    Edge case: nested structures render correctly.
    """
    memory = init_machine()
    state = memory.registers

    atoms = empty_symbol_table()
    atoms.symbol_by_id[1] = "f"
    atoms.symbol_by_id[2] = "g"
    atoms.symbol_by_id[10] = "a"

    a = make_const(state, memory, 10)

    g = make_functor(state, memory, 2, 1)
    f = make_functor(state, memory, 1, 1)

    g_a = make_structure(state, memory, g, np.array([a], dtype=np.uint16))
    f_ga = make_structure(state, memory, f, np.array([g_a], dtype=np.uint16))

    result = format_heap_term(state, memory, f_ga, atoms)
    assert result == "f(g(a))"


def test_debug_shared_variable():
    """
    Edge case: shared variables appear consistently in debug output.
    """
    memory = init_machine()
    state = memory.registers

    atoms = empty_symbol_table()
    atoms.symbol_by_id[1] = "f"

    x = make_var(state, memory)
    f = make_functor(state, memory, 1, 2)
    s = make_structure(state, memory, f, np.array([x, x], dtype=np.uint16))

    result = format_heap_term(state, memory, s, atoms)
    assert result == "f(_, _)"


# -------------------------------------------------
# Trail stack edge cases
# -------------------------------------------------


def test_trail_var_records_index():
    """
    Edge case: trailing a variable records its heap index.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    trail_var(state, memory, x)

    assert state.TR[0] == 1
    assert memory.trail[0] == x


def test_trail_multiple_vars_ordered():
    """
    Invariant: trail preserves insertion order.
    """
    memory = init_machine()
    state = memory.registers

    xs = [make_var(state, memory) for _ in range(3)]
    for x in xs:
        trail_var(state, memory, x)

    assert list(memory.trail[:3]) == xs


# -------------------------------------------------
# Functor layout invariants
# -------------------------------------------------


def test_functor_header_layout():
    """
    Invariant: functor header consists of two consecutive FUN cells
    storing (symbol_id, arity).
    """
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=42, arity=3)

    assert memory.heap.tags[f] == TAG.FUN
    assert memory.heap.tags[f + 1] == TAG.FUN
    assert memory.heap.cells[f] == 42
    assert memory.heap.cells[f + 1] == 3


# -------------------------------------------------
# Preserved duplicate definitions from the original file
# -------------------------------------------------


def test_deref_long_reference_chain_repeated_case():
    """
    Edge case: dereferencing a deep REF → REF → ... chain.

    Ensures deref follows references transitively and terminates
    at the final self-referential variable.
    """
    memory = init_machine()
    state = memory.registers

    vars_ = [make_var(state, memory) for _ in range(10)]

    for i in range(9):
        memory.heap.cells[vars_[i]] = vars_[i + 1]

    assert deref(state, memory, vars_[0]) == vars_[9]


def test_deref_stops_at_constant_repeated_case():
    """
    Edge case: dereferencing a variable bound directly to a constant.

    deref must stop when a non-REF tag is encountered.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    c = make_const(state, memory, 99)

    memory.heap.cells[x] = c

    assert deref(state, memory, x) == c
    assert memory.heap.tags[c] == TAG.CON


def test_zero_arity_functor_structure_repeated_case():
    """
    Edge case: zero-arity functor (atom-like structure).

    A structure cell is still allocated even with no arguments.
    """
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=1, arity=0)
    s = make_structure(state, memory, f, np.empty(0, dtype=np.uint16))

    assert memory.heap.tags[s] == TAG.STR
    assert memory.heap.cells[s] == f


def test_structure_keeps_raw_argument_addresses_repeated_case():
    """
    Edge case: structure arguments are stored verbatim.

    Arguments are not dereferenced at construction time.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)

    memory.heap.cells[x] = y  # x → y

    f = make_functor(state, memory, 1, 1)
    s = make_structure(state, memory, f, np.array([x], dtype=np.uint16))

    arg_cell = s + 1
    assert memory.heap.cells[arg_cell] == x


def test_heap_top_monotonicity_repeated_case():
    """
    Invariant: heap_top must grow monotonically.
    """
    memory = init_machine()
    state = memory.registers

    tops = []
    for _ in range(5):
        make_var(state, memory)
        tops.append(int(state.H[0]))

    assert tops == sorted(tops)


def test_multiple_structures_do_not_overlap_repeated_case():
    """
    Invariant: creating multiple structures must not overwrite
    earlier heap regions.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    f = make_functor(state, memory, 1, 1)

    s1 = make_structure(state, memory, f, np.array([x], dtype=np.uint16))
    top_after_s1 = int(state.H[0])

    s2 = make_structure(state, memory, f, np.array([x], dtype=np.uint16))

    assert memory.heap.tags[s1] == TAG.STR
    assert s2 >= top_after_s1


def test_debug_nested_structure_repeated_case():
    """
    Edge case: nested structures render correctly.
    """
    memory = init_machine()
    state = memory.registers

    atoms = empty_symbol_table()
    atoms.symbol_by_id[1] = "f"
    atoms.symbol_by_id[2] = "g"
    atoms.symbol_by_id[10] = "a"

    a = make_const(state, memory, 10)

    g = make_functor(state, memory, 2, 1)
    f = make_functor(state, memory, 1, 1)

    g_a = make_structure(state, memory, g, np.array([a], dtype=np.uint16))
    f_ga = make_structure(state, memory, f, np.array([g_a], dtype=np.uint16))

    result = format_heap_term(state, memory, f_ga, atoms)
    assert result == "f(g(a))"


def test_debug_shared_variable_repeated_case():
    """
    Edge case: shared variables appear consistently in debug output.
    """
    memory = init_machine()
    state = memory.registers

    atoms = empty_symbol_table()
    atoms.symbol_by_id[1] = "f"

    x = make_var(state, memory)
    f = make_functor(state, memory, 1, 2)
    s = make_structure(state, memory, f, np.array([x, x], dtype=np.uint16))

    result = format_heap_term(state, memory, s, atoms)
    assert result == "f(_, _)"


def test_trail_var_records_index_repeated_case():
    """
    Edge case: trailing a variable records its heap index.
    """
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    trail_var(state, memory, x)

    assert state.TR[0] == 1
    assert memory.trail[0] == x


def test_trail_multiple_vars_ordered_repeated_case():
    """
    Invariant: trail preserves insertion order.
    """
    memory = init_machine()
    state = memory.registers

    xs = [make_var(state, memory) for _ in range(3)]
    for x in xs:
        trail_var(state, memory, x)

    assert list(memory.trail[:3]) == xs


def test_functor_header_layout_repeated_case():
    """
    Invariant: functor header consists of two consecutive FUN cells
    storing (symbol_id, arity).
    """
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=42, arity=3)

    assert memory.heap.tags[f] == TAG.FUN
    assert memory.heap.tags[f + 1] == TAG.FUN
    assert memory.heap.cells[f] == 42
    assert memory.heap.cells[f + 1] == 3


def test_make_var_increments_heap_top():
    """make_var allocates consecutive heap cells and advances heap_top."""
    memory = init_machine()
    state = memory.registers

    h0 = state.H[0]
    v1 = make_var(state, memory)
    v2 = make_var(state, memory)

    assert v1 == h0
    assert v2 == h0 + 1
    assert state.H[0] == h0 + 2
    assert memory.heap.cells[v1] == v1
    assert memory.heap.tags[v1] == TAG.REF


def test_make_const_tag_and_value():
    """make_const stores the value and tags the heap cell as CON."""
    memory = init_machine()
    state = memory.registers

    c = make_const(state, memory, 42)

    assert memory.heap.cells[c] == 42
    assert memory.heap.tags[c] == TAG.CON
    assert state.H[0] == c + 1


def test_make_functor_allocates_two_cells():
    """make_functor allocates symbol and arity cells with FUN tags."""
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=7, arity=3)

    assert memory.heap.cells[f] == 7
    assert memory.heap.cells[f + 1] == 3
    assert memory.heap.tags[f] == TAG.FUN
    assert memory.heap.tags[f + 1] == TAG.FUN
    assert state.H[0] == f + 2


def test_make_structure_zero_arity():
    """make_structure supports zero-arity functors."""
    memory = init_machine()
    state = memory.registers

    f = make_functor(state, memory, symbol_id=1, arity=0)
    s = make_structure(state, memory, f, np.empty((0,), dtype=np.uint16))

    assert memory.heap.tags[s] == TAG.STR
    assert memory.heap.cells[s] == f
    assert state.H[0] == s + 1


def test_make_structure_keeps_raw_argument_addresses():
    """make_structure stores argument addresses without dereferencing."""
    memory = init_machine()
    state = memory.registers

    x = make_var(state, memory)
    y = make_var(state, memory)
    f = make_functor(state, memory, symbol_id=9, arity=2)

    s = make_structure(state, memory, f, np.array([x, y], dtype=np.uint16))

    assert memory.heap.tags[s] == TAG.STR
    assert memory.heap.cells[s + 1] == x
    assert memory.heap.cells[s + 2] == y


def test_deref_stops_at_constant_short_chain():
    """deref stops when encountering a non-REF (constant) cell."""
    memory = init_machine()
    state = memory.registers

    c = make_const(state, memory, 99)
    v = make_var(state, memory)

    memory.heap.cells[v] = c  # bind variable to constant

    assert deref(state, memory, v) == c


def test_deref_long_reference_chain_three_nodes():
    """deref follows a chain of variable references to the final target."""
    memory = init_machine()
    state = memory.registers

    a = make_var(state, memory)
    b = make_var(state, memory)
    c = make_var(state, memory)

    memory.heap.cells[a] = b
    memory.heap.cells[b] = c

    assert deref(state, memory, a) == c


def test_trail_var_records_index_and_advances_top():
    """trail_var records variable indices and increments trail_top."""
    memory = init_machine()
    state = memory.registers

    trail_var(state, memory, 3)
    trail_var(state, memory, 7)

    assert memory.trail[0] == 3
    assert memory.trail[1] == 7
    assert state.TR[0] == 2
