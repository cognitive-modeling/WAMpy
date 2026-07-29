import numpy as np

from wampy.config import DEFAULT_CONFIG
from wampy.runtime.stack import (
    init_stack,
    TAG,
    deref,
    make_var,
    make_const,
    make_functor,
    make_structure,
    trail_var,
)
from wampy.utils.debug import debug_term_from_stack

# -------------------------------------------------

# -------------------------------------------------


def test_init_stack():
    """Stack initialization"""
    stack = init_stack()

    assert stack.heap_top[0] == 0
    assert stack.trail_top[0] == 0
    assert stack.cp_top[0] == 0
    assert stack.u_top[0] == 0
    assert stack.choice_X.shape[1] == DEFAULT_CONFIG.limits.max_x
    assert stack.call_X.shape[1] == DEFAULT_CONFIG.limits.max_x
    assert stack.struct_stack_size == DEFAULT_CONFIG.limits.max_struct
    assert stack.step_limit == DEFAULT_CONFIG.solver.max_steps
    assert stack.unify_step_limit == DEFAULT_CONFIG.solver.max_unify_steps

    # All heap tags initialized as REF
    assert np.all(stack.tags == TAG.REF)


# -------------------------------------------------
# Variable creation and dereferencing
# -------------------------------------------------


def test_make_var_creates_self_reference():
    stack = init_stack()

    x = make_var(stack)

    assert stack.heap[x] == x
    assert stack.tags[x] == TAG.REF
    assert stack.heap_top[0] == 1


def test_deref_unbound_variable():
    stack = init_stack()

    x = make_var(stack)
    assert deref(stack, x) == x


def test_deref_reference_chain():
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)

    # Make x point to y
    stack.heap[x] = y

    assert deref(stack, x) == y
    assert deref(stack, y) == y


def test_deref_reference_cycle_returns_invalid_address():
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)
    stack.heap[x] = y
    stack.heap[y] = x

    assert deref(stack, x) == stack.heap_top[0]


# -------------------------------------------------
# Constants
# -------------------------------------------------


def test_make_const():
    stack = init_stack()

    c = make_const(stack, 42)

    assert stack.heap[c] == 42
    assert stack.tags[c] == TAG.CON
    assert stack.heap_top[0] == 1


# -------------------------------------------------
# Functors and structures
# -------------------------------------------------


def test_make_functor():
    stack = init_stack()

    f = make_functor(stack, symbol_id=1, arity=2)

    assert stack.tags[f] == TAG.FUN
    assert stack.heap[f] == 1
    assert stack.heap[f + 1] == 2
    assert stack.heap_top[0] == 2


def test_make_structure_with_variables():
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)

    f = make_functor(stack, symbol_id=1, arity=2)
    s = make_structure(stack, f, np.array([x, y], dtype=np.uint16))

    assert stack.tags[s] == TAG.STR
    assert stack.heap[s] == f
    assert stack.heap_top[0] == 2 + 2 + 1 + 2  # vars + functor + structure header + args


# -------------------------------------------------
# Dereferencing edge cases
# -------------------------------------------------


def test_deref_long_reference_chain():
    """
    Edge case: dereferencing a deep REF → REF → ... chain.

    Ensures deref follows references transitively and terminates
    at the final self-referential variable.
    """
    stack = init_stack()

    vars_ = [make_var(stack) for _ in range(10)]

    for i in range(9):
        stack.heap[vars_[i]] = vars_[i + 1]

    assert deref(stack, vars_[0]) == vars_[9]


def test_deref_stops_at_constant():
    """
    Edge case: dereferencing a variable bound directly to a constant.

    deref must stop when a non-REF tag is encountered.
    """
    stack = init_stack()

    x = make_var(stack)
    c = make_const(stack, 99)

    stack.heap[x] = c

    assert deref(stack, x) == c
    assert stack.tags[c] == TAG.CON


# -------------------------------------------------
# Structure construction edge cases
# -------------------------------------------------


def test_zero_arity_functor_structure():
    """
    Edge case: zero-arity functor (atom-like structure).

    A structure cell is still allocated even with no arguments.
    """
    stack = init_stack()

    f = make_functor(stack, symbol_id=1, arity=0)
    s = make_structure(stack, f, np.empty([], dtype=np.uint16))

    assert stack.tags[s] == TAG.STR
    assert stack.heap[s] == f


def test_structure_keeps_raw_argument_addresses():
    """
    Edge case: structure arguments are stored verbatim.

    Arguments are not dereferenced at construction time.
    """
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)

    stack.heap[x] = y  # x → y

    f = make_functor(stack, 1, 1)
    s = make_structure(stack, f, np.array([x], dtype=np.uint16))

    arg_cell = s + 1
    assert stack.heap[arg_cell] == x


# -------------------------------------------------
# Heap behavior invariants
# -------------------------------------------------


def test_heap_top_monotonicity():
    """
    Invariant: heap_top must grow monotonically.
    """
    stack = init_stack()

    tops = []
    for _ in range(5):
        make_var(stack)
        tops.append(int(stack.heap_top[0]))

    assert tops == sorted(tops)


def test_multiple_structures_do_not_overlap():
    """
    Invariant: creating multiple structures must not overwrite
    earlier heap regions.
    """
    stack = init_stack()

    x = make_var(stack)
    f = make_functor(stack, 1, 1)

    s1 = make_structure(stack, f, np.array([x], dtype=np.uint16))
    top_after_s1 = int(stack.heap_top[0])

    s2 = make_structure(stack, f, np.array([x], dtype=np.uint16))

    assert s2 >= top_after_s1


# -------------------------------------------------
# Debug printer edge cases
# -------------------------------------------------


def test_debug_nested_structure():
    """
    Edge case: nested structures render correctly.
    """
    stack = init_stack()

    ATOMS = {1: "f", 2: "g", 10: "a"}

    a = make_const(stack, 10)

    g = make_functor(stack, 2, 1)
    f = make_functor(stack, 1, 1)

    g_a = make_structure(stack, g, np.array([a], dtype=np.uint16))
    f_ga = make_structure(stack, f, np.array([g_a], dtype=np.uint16))

    result = debug_term_from_stack(stack, f_ga, ATOMS)
    assert result == "f(g(a))"


def test_debug_shared_variable():
    """
    Edge case: shared variables appear consistently in debug output.
    """
    stack = init_stack()

    ATOMS = {1: "f"}

    x = make_var(stack)
    f = make_functor(stack, 1, 2)
    s = make_structure(stack, f, np.array([x, x], dtype=np.uint16))

    result = debug_term_from_stack(stack, s, ATOMS)
    assert result == "f(_, _)"


# -------------------------------------------------
# Trail stack edge cases
# -------------------------------------------------


def test_trail_var_records_index():
    """
    Edge case: trailing a variable records its heap index.
    """
    stack = init_stack()

    x = make_var(stack)
    trail_var(stack, x)

    assert stack.trail_top[0] == 1
    assert stack.trail[0] == x


def test_trail_multiple_vars_ordered():
    """
    Invariant: trail preserves insertion order.
    """
    stack = init_stack()

    xs = [make_var(stack) for _ in range(3)]
    for x in xs:
        trail_var(stack, x)

    assert list(stack.trail[:3]) == xs


# -------------------------------------------------
# Functor layout invariants
# -------------------------------------------------


def test_functor_header_layout():
    """
    Invariant: functor header consists of two consecutive FUN cells
    storing (symbol_id, arity).
    """
    stack = init_stack()

    f = make_functor(stack, symbol_id=42, arity=3)

    assert stack.tags[f] == TAG.FUN
    assert stack.tags[f + 1] == TAG.FUN
    assert stack.heap[f] == 42
    assert stack.heap[f + 1] == 3


# -------------------------------------------------
# Dereferencing edge cases
# -------------------------------------------------


def test_deref_long_reference_chain():
    """
    Edge case: dereferencing a deep REF → REF → ... chain.

    Ensures deref follows references transitively and terminates
    at the final self-referential variable.
    """
    stack = init_stack()

    vars_ = [make_var(stack) for _ in range(10)]

    for i in range(9):
        stack.heap[vars_[i]] = vars_[i + 1]

    assert deref(stack, vars_[0]) == vars_[9]


def test_deref_stops_at_constant():
    """
    Edge case: dereferencing a variable bound directly to a constant.

    deref must stop when a non-REF tag is encountered.
    """
    stack = init_stack()

    x = make_var(stack)
    c = make_const(stack, 99)

    stack.heap[x] = c

    assert deref(stack, x) == c
    assert stack.tags[c] == TAG.CON


# -------------------------------------------------
# Structure construction edge cases
# -------------------------------------------------


def test_zero_arity_functor_structure():
    """
    Edge case: zero-arity functor (atom-like structure).

    A structure cell is still allocated even with no arguments.
    """
    stack = init_stack()

    f = make_functor(stack, symbol_id=1, arity=0)
    s = make_structure(stack, f, np.empty(0, dtype=np.uint16))

    assert stack.tags[s] == TAG.STR
    assert stack.heap[s] == f


def test_structure_keeps_raw_argument_addresses():
    """
    Edge case: structure arguments are stored verbatim.

    Arguments are not dereferenced at construction time.
    """
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)

    stack.heap[x] = y  # x → y

    f = make_functor(stack, 1, 1)
    s = make_structure(stack, f, np.array([x], dtype=np.uint16))

    arg_cell = s + 1
    assert stack.heap[arg_cell] == x


# -------------------------------------------------
# Heap behavior invariants
# -------------------------------------------------


def test_heap_top_monotonicity():
    """
    Invariant: heap_top must grow monotonically.
    """
    stack = init_stack()

    tops = []
    for _ in range(5):
        make_var(stack)
        tops.append(int(stack.heap_top[0]))

    assert tops == sorted(tops)


def test_multiple_structures_do_not_overlap():
    """
    Invariant: creating multiple structures must not overwrite
    earlier heap regions.
    """
    stack = init_stack()

    x = make_var(stack)
    f = make_functor(stack, 1, 1)

    s1 = make_structure(stack, f, np.array([x], dtype=np.uint16))
    top_after_s1 = int(stack.heap_top[0])

    s2 = make_structure(stack, f, np.array([x], dtype=np.uint16))

    assert s2 >= top_after_s1


# -------------------------------------------------
# Debug printer edge cases
# -------------------------------------------------


def test_debug_nested_structure():
    """
    Edge case: nested structures render correctly.
    """
    stack = init_stack()

    ATOMS = {1: "f", 2: "g", 10: "a"}

    a = make_const(stack, 10)

    g = make_functor(stack, 2, 1)
    f = make_functor(stack, 1, 1)

    g_a = make_structure(stack, g, np.array([a], dtype=np.uint16))
    f_ga = make_structure(stack, f, np.array([g_a], dtype=np.uint16))

    result = debug_term_from_stack(stack, f_ga, ATOMS)
    assert result == "f(g(a))"


def test_debug_shared_variable():
    """
    Edge case: shared variables appear consistently in debug output.
    """
    stack = init_stack()

    ATOMS = {1: "f"}

    x = make_var(stack)
    f = make_functor(stack, 1, 2)
    s = make_structure(stack, f, np.array([x, x], dtype=np.uint16))

    result = debug_term_from_stack(stack, s, ATOMS)
    assert result == "f(_, _)"


# -------------------------------------------------
# Trail stack edge cases
# -------------------------------------------------


def test_trail_var_records_index():
    """
    Edge case: trailing a variable records its heap index.
    """
    stack = init_stack()

    x = make_var(stack)
    trail_var(stack, x)

    assert stack.trail_top[0] == 1
    assert stack.trail[0] == x


def test_trail_multiple_vars_ordered():
    """
    Invariant: trail preserves insertion order.
    """
    stack = init_stack()

    xs = [make_var(stack) for _ in range(3)]
    for x in xs:
        trail_var(stack, x)

    assert list(stack.trail[:3]) == xs


def test_functor_header_layout():
    """
    Invariant: functor header consists of two consecutive FUN cells
    storing (symbol_id, arity).
    """
    stack = init_stack()

    f = make_functor(stack, symbol_id=42, arity=3)

    assert stack.tags[f] == TAG.FUN
    assert stack.tags[f + 1] == TAG.FUN
    assert stack.heap[f] == 42
    assert stack.heap[f + 1] == 3


def test_make_var_increments_heap_top():
    """make_var allocates consecutive heap cells and advances heap_top."""
    stack = init_stack()

    h0 = stack.heap_top[0]
    v1 = make_var(stack)
    v2 = make_var(stack)

    assert v1 == h0
    assert v2 == h0 + 1
    assert stack.heap_top[0] == h0 + 2
    assert stack.heap[v1] == v1
    assert stack.tags[v1] == TAG.REF


def test_make_const_tag_and_value():
    """make_const stores the value and tags the heap cell as CON."""
    stack = init_stack()

    c = make_const(stack, 42)

    assert stack.heap[c] == 42
    assert stack.tags[c] == TAG.CON
    assert stack.heap_top[0] == c + 1


def test_make_functor_allocates_two_cells():
    """make_functor allocates symbol and arity cells with FUN tags."""
    stack = init_stack()

    f = make_functor(stack, symbol_id=7, arity=3)

    assert stack.heap[f] == 7
    assert stack.heap[f + 1] == 3
    assert stack.tags[f] == TAG.FUN
    assert stack.tags[f + 1] == TAG.FUN
    assert stack.heap_top[0] == f + 2


def test_make_structure_zero_arity():
    """make_structure supports zero-arity functors."""
    stack = init_stack()

    f = make_functor(stack, symbol_id=1, arity=0)
    s = make_structure(stack, f, np.empty((0,), dtype=np.uint16))

    assert stack.tags[s] == TAG.STR
    assert stack.heap[s] == f
    assert stack.heap_top[0] == s + 1


def test_make_structure_keeps_raw_argument_addresses():
    """make_structure stores argument addresses without dereferencing."""
    stack = init_stack()

    x = make_var(stack)
    y = make_var(stack)
    f = make_functor(stack, symbol_id=9, arity=2)

    s = make_structure(stack, f, np.array([x, y], dtype=np.uint16))

    assert stack.tags[s] == TAG.STR
    assert stack.heap[s + 1] == x
    assert stack.heap[s + 2] == y


def test_deref_stops_at_constant():
    """deref stops when encountering a non-REF (constant) cell."""
    stack = init_stack()

    c = make_const(stack, 99)
    v = make_var(stack)

    stack.heap[v] = c  # bind variable to constant

    assert deref(stack, v) == c


def test_deref_long_reference_chain():
    """deref follows a chain of variable references to the final target."""
    stack = init_stack()

    a = make_var(stack)
    b = make_var(stack)
    c = make_var(stack)

    stack.heap[a] = b
    stack.heap[b] = c

    assert deref(stack, a) == c


def test_trail_var_records_index_and_advances_top():
    """trail_var records variable indices and increments trail_top."""
    stack = init_stack()

    trail_var(stack, 3)
    trail_var(stack, 7)

    assert stack.trail[0] == 3
    assert stack.trail[1] == 7
    assert stack.trail_top[0] == 2
