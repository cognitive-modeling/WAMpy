import numpy as np
from wampy.config import load_config_toml
from wampy.runtime.machine import init_machine
from wampy.runtime.machine.heap import (
    deref,
    make_const,
    make_functor,
    make_structure,
    make_var,
)
from wampy.runtime.machine.stack import ChoiceSlot
from wampy.runtime.tags import TAG
from wampy.runtime.unification import bind_var, unify
from wampy.status import WAMStatus


def unify_status(state, memory, left, right) -> WAMStatus:
    return WAMStatus(int(unify(state, memory, left, right)))


def helpertest_backtrack(state, memory):
    """
    Low-level backtracking helper for unit tests.
    Mirrors WAM untrailing semantics over separate state and memory.
    """
    if state.B[0] < 0:
        return

    address = state.B[0]
    state.B[0] = memory.stack.cells[address + ChoiceSlot.PREVIOUS_B]
    state.local_top[0] = address

    saved_h = memory.stack.cells[address + ChoiceSlot.H]
    target = memory.stack.cells[address + ChoiceSlot.TR]

    # untrail
    while state.TR[0] > target:
        state.TR[0] -= 1
        x = memory.trail[state.TR[0]]
        memory.heap.cells[x] = x
        memory.heap.tags[x] = TAG.REF

    # restore heap
    state.H[0] = saved_h


def helpertest_push_choice_point(state, memory):
    """
    Low-level choice point.
    Used by unification tests.
    """
    address = state.local_top[0]
    memory.stack.cells[address + ChoiceSlot.TR] = state.TR[0]
    memory.stack.cells[address + ChoiceSlot.H] = state.H[0]
    memory.stack.cells[address + ChoiceSlot.PREVIOUS_B] = state.B[0]
    state.B[0] = address
    state.local_top[0] = address + memory.stack.choice_point_frame_size


# ---------------------------------------------------------------------------
# Basic unification tests
# ---------------------------------------------------------------------------


def test_unify_same_variable():
    memory = init_machine()
    state = memory.registers
    X = make_var(state, memory)

    assert unify_status(state, memory, X, X) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == X


def test_unify_variable_with_constant():
    memory = init_machine()
    state = memory.registers
    X = make_var(state, memory)
    c1 = make_const(state, memory, 1)

    assert unify_status(state, memory, X, c1) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == c1


def test_unify_two_different_constants_fails():
    memory = init_machine()
    state = memory.registers
    c1 = make_const(state, memory, 1)
    c2 = make_const(state, memory, 2)

    assert unify_status(state, memory, c1, c2) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Structure unification
# ---------------------------------------------------------------------------


def test_unify_simple_structure():
    """
    f(X, Y) = f(a, b)
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)

    F = 1  # f/2
    a_id = 10
    b_id = 11

    f_fun = make_functor(state, memory, F, 2)
    a = make_const(state, memory, a_id)
    b = make_const(state, memory, b_id)

    left = make_structure(state, memory, f_fun, np.array([X, Y], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([a, b], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == a
    assert deref(state, memory, Y) == b


def test_unify_structure_functor_mismatch():
    """
    f(X) != g(X)
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)

    F = 1  # f/1
    G = 2  # g/1

    f_fun = make_functor(state, memory, F, 1)
    g_fun = make_functor(state, memory, G, 1)

    f_struct = make_structure(state, memory, f_fun, np.array([X], dtype=np.uint16))
    g_struct = make_structure(state, memory, g_fun, np.array([X], dtype=np.uint16))

    assert unify_status(state, memory, f_struct, g_struct) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Nested / shared variable unification
# ---------------------------------------------------------------------------


def test_unify_nested_shared_variables():
    """
    f(g(X), X) = f(g(a), a)
    """

    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)

    F = 1  # f/2
    G = 2  # g/1
    a_id = 10

    f_fun = make_functor(state, memory, F, 2)
    g_fun = make_functor(state, memory, G, 1)
    a = make_const(state, memory, a_id)

    g_X = make_structure(state, memory, g_fun, np.array([X], dtype=np.uint16))
    left = make_structure(state, memory, f_fun, np.array([g_X, X], dtype=np.uint16))

    g_a = make_structure(state, memory, g_fun, np.array([a], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([g_a, a], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == a


# ---------------------------------------------------------------------------
# Backtracking behavior
# ---------------------------------------------------------------------------


def test_backtracking_restores_variable():
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    c1 = make_const(state, memory, 1)

    helpertest_push_choice_point(state, memory)
    assert unify_status(state, memory, X, c1) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == c1

    helpertest_backtrack(state, memory)

    # After backtracking, X should be unbound again
    assert deref(state, memory, X) == X
    assert memory.heap.tags[X] == TAG.REF


def test_multiple_choice_points():
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    c1 = make_const(state, memory, 1)
    c2 = make_const(state, memory, 2)

    helpertest_push_choice_point(state, memory)
    bind_var(state, memory, X, c1)
    assert deref(state, memory, X) == c1

    helpertest_backtrack(state, memory)
    assert deref(state, memory, X) == X

    helpertest_push_choice_point(state, memory)
    bind_var(state, memory, X, c2)
    assert deref(state, memory, X) == c2

    helpertest_backtrack(state, memory)
    assert deref(state, memory, X) == X


# ---------------------------------------------------------------------------
# Variable aliasing and chains
# ---------------------------------------------------------------------------


def test_variable_variable_aliasing():
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)

    assert unify_status(state, memory, X, Y) == WAMStatus.SUCCESS

    # X and Y should now dereference to the same root
    assert deref(state, memory, X) == deref(state, memory, Y)


def test_variable_chain_binding():
    """
    X = Y, Y = Z, Z = a
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)
    Z = make_var(state, memory)
    a = make_const(state, memory, 10)

    assert unify_status(state, memory, X, Y) == WAMStatus.SUCCESS
    assert unify_status(state, memory, Y, Z) == WAMStatus.SUCCESS
    assert unify_status(state, memory, Z, a) == WAMStatus.SUCCESS

    assert deref(state, memory, X) == a
    assert deref(state, memory, Y) == a
    assert deref(state, memory, Z) == a


# ---------------------------------------------------------------------------
# Repeated variable constraints
# ---------------------------------------------------------------------------


def test_repeated_variable_success():
    """
    f(X, X) = f(a, a)
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)

    F = 1
    a = make_const(state, memory, 10)

    f_fun = make_functor(state, memory, F, 2)

    left = make_structure(state, memory, f_fun, np.array([X, X], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([a, a], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == a


def test_repeated_variable_failure():
    """
    f(X, X) = f(a, b)  should fail
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)

    F = 1
    a = make_const(state, memory, 10)
    b = make_const(state, memory, 11)

    f_fun = make_functor(state, memory, F, 2)

    left = make_structure(state, memory, f_fun, np.array([X, X], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([a, b], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.EXHAUSTED


def test_variable_can_bind_to_structure_containing_itself_without_freezing():
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    f_fun = make_functor(state, memory, 1, 1)
    f_x = make_structure(state, memory, f_fun, np.array([X], dtype=np.uint16))

    assert unify_status(state, memory, X, f_x) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == f_x


def test_unify_step_limit_bounds_large_work():
    config = load_config_toml(
        """
        [runtime]
        max_unify_steps = 1
        """
    )
    memory = init_machine(config.runtime)
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)
    f_fun = make_functor(state, memory, 1, 2)
    left = make_structure(state, memory, f_fun, np.array([X, Y], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([X, Y], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.UNIFY_STEP_LIMIT


# ---------------------------------------------------------------------------
# Arity mismatch
# ---------------------------------------------------------------------------


def test_structure_arity_mismatch():
    """
    f(X) != f(X, Y)
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)

    F = 1

    f1_fun = make_functor(state, memory, F, 1)
    f2_fun = make_functor(state, memory, F, 2)

    s1 = make_structure(state, memory, f1_fun, np.array([X], dtype=np.uint16))
    s2 = make_structure(state, memory, f2_fun, np.array([X, Y], dtype=np.uint16))

    assert unify_status(state, memory, s1, s2) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Deep nesting
# ---------------------------------------------------------------------------


def test_deeply_nested_structures():
    """
    f(g(h(X))) = f(g(h(a)))
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)

    F = 1
    G = 2
    H = 3
    a = make_const(state, memory, 10)

    f_fun = make_functor(state, memory, F, 1)
    g_fun = make_functor(state, memory, G, 1)
    h_fun = make_functor(state, memory, H, 1)

    h_X = make_structure(state, memory, h_fun, np.array([X], dtype=np.uint16))
    g_h_X = make_structure(state, memory, g_fun, np.array([h_X], dtype=np.uint16))
    left = make_structure(state, memory, f_fun, np.array([g_h_X], dtype=np.uint16))

    h_a = make_structure(state, memory, h_fun, np.array([a], dtype=np.uint16))
    g_h_a = make_structure(state, memory, g_fun, np.array([h_a], dtype=np.uint16))
    right = make_structure(state, memory, f_fun, np.array([g_h_a], dtype=np.uint16))

    assert unify_status(state, memory, left, right) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == a


# ---------------------------------------------------------------------------
# Backtracking with multiple bindings
# ---------------------------------------------------------------------------


def test_backtracking_multiple_bindings():
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    Y = make_var(state, memory)

    a = make_const(state, memory, 10)
    b = make_const(state, memory, 11)

    helpertest_push_choice_point(state, memory)

    assert unify_status(state, memory, X, a) == WAMStatus.SUCCESS
    assert unify_status(state, memory, Y, b) == WAMStatus.SUCCESS

    assert deref(state, memory, X) == a
    assert deref(state, memory, Y) == b

    helpertest_backtrack(state, memory)

    assert deref(state, memory, X) == X
    assert deref(state, memory, Y) == Y


# ---------------------------------------------------------------------------
# Failure should not corrupt heap
# ---------------------------------------------------------------------------


def test_failed_unification_does_not_bind():
    """
    X = a, then try unify(X, b) -> fail, X should remain bound to a
    """
    memory = init_machine()
    state = memory.registers

    X = make_var(state, memory)
    a = make_const(state, memory, 10)
    b = make_const(state, memory, 11)

    assert unify_status(state, memory, X, a) == WAMStatus.SUCCESS
    assert deref(state, memory, X) == a

    assert unify_status(state, memory, X, b) == WAMStatus.EXHAUSTED
    assert deref(state, memory, X) == a
