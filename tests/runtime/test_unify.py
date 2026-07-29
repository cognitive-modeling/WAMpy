import numpy as np

from wampy.config import load_config_toml
from wampy.status import WAMStatus
from wampy.runtime.unify import unify, bind_var
from wampy.runtime.stack import (
    TAG,
    deref,
    init_stack,
    make_const,
    make_functor,
    make_structure,
    make_var,
)


def unify_status(stack, left, right) -> WAMStatus:
    return WAMStatus(int(unify(stack, left, right)))


def helpertest_backtrack(stack):
    """
    Stack-only backtracking helper for unit tests.
    Mirrors WAM untrailing semantics without a Machine.
    """
    if stack.cp_top[0] == 0:
        return

    stack.cp_top[0] -= 1
    i = stack.cp_top[0]

    saved_heap_top = stack.choice_heap_tops[i]
    target = stack.choice_points[i]

    # untrail
    while stack.trail_top[0] > target:
        stack.trail_top[0] -= 1
        x = stack.trail[stack.trail_top[0]]
        stack.heap[x] = x
        stack.tags[x] = TAG.REF

    # restore heap
    stack.heap_top[0] = saved_heap_top


def helpertest_push_choice_point(stack):
    """
    Stack-only choice point.
    Used by unification tests.
    """
    i = stack.cp_top[0]
    stack.choice_points[i] = stack.trail_top[0]
    stack.choice_heap_tops[i] = stack.heap_top[0]
    stack.cp_top[0] = i + 1


# ---------------------------------------------------------------------------
# Basic unification tests
# ---------------------------------------------------------------------------


def test_unify_same_variable():
    stack = init_stack()
    X = make_var(stack)

    assert unify_status(stack, X, X) == WAMStatus.SUCCESS
    assert deref(stack, X) == X


def test_unify_variable_with_constant():
    stack = init_stack()
    X = make_var(stack)
    c1 = make_const(stack, 1)

    assert unify_status(stack, X, c1) == WAMStatus.SUCCESS
    assert deref(stack, X) == c1


def test_unify_two_different_constants_fails():
    stack = init_stack()
    c1 = make_const(stack, 1)
    c2 = make_const(stack, 2)

    assert unify_status(stack, c1, c2) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Structure unification
# ---------------------------------------------------------------------------


def test_unify_simple_structure():
    """
    f(X, Y) = f(a, b)
    """
    stack = init_stack()

    X = make_var(stack)
    Y = make_var(stack)

    F = 1  # f/2
    a_id = 10
    b_id = 11

    f_fun = make_functor(stack, F, 2)
    a = make_const(stack, a_id)
    b = make_const(stack, b_id)

    left = make_structure(stack, f_fun, np.array([X, Y], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([a, b], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.SUCCESS
    assert deref(stack, X) == a
    assert deref(stack, Y) == b


def test_unify_structure_functor_mismatch():
    """
    f(X) != g(X)
    """
    stack = init_stack()

    X = make_var(stack)

    F = 1  # f/1
    G = 2  # g/1

    f_fun = make_functor(stack, F, 1)
    g_fun = make_functor(stack, G, 1)

    f_struct = make_structure(stack, f_fun, np.array([X], dtype=np.uint16))
    g_struct = make_structure(stack, g_fun, np.array([X], dtype=np.uint16))

    assert unify_status(stack, f_struct, g_struct) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Nested / shared variable unification
# ---------------------------------------------------------------------------


def test_unify_nested_shared_variables():
    """
    f(g(X), X) = f(g(a), a)
    """

    stack = init_stack()

    X = make_var(stack)

    F = 1  # f/2
    G = 2  # g/1
    a_id = 10

    f_fun = make_functor(stack, F, 2)
    g_fun = make_functor(stack, G, 1)
    a = make_const(stack, a_id)

    g_X = make_structure(stack, g_fun, np.array([X], dtype=np.uint16))
    left = make_structure(stack, f_fun, np.array([g_X, X], dtype=np.uint16))

    g_a = make_structure(stack, g_fun, np.array([a], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([g_a, a], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.SUCCESS
    assert deref(stack, X) == a


# ---------------------------------------------------------------------------
# Backtracking behavior
# ---------------------------------------------------------------------------


def test_backtracking_restores_variable():
    stack = init_stack()

    X = make_var(stack)
    c1 = make_const(stack, 1)

    helpertest_push_choice_point(stack)
    assert unify_status(stack, X, c1) == WAMStatus.SUCCESS
    assert deref(stack, X) == c1

    helpertest_backtrack(stack)

    # After backtracking, X should be unbound again
    assert deref(stack, X) == X
    assert stack.tags[X] == TAG.REF


def test_multiple_choice_points():
    stack = init_stack()

    X = make_var(stack)
    c1 = make_const(stack, 1)
    c2 = make_const(stack, 2)

    helpertest_push_choice_point(stack)
    bind_var(stack, X, c1)
    assert deref(stack, X) == c1

    helpertest_backtrack(stack)
    assert deref(stack, X) == X

    helpertest_push_choice_point(stack)
    bind_var(stack, X, c2)
    assert deref(stack, X) == c2

    helpertest_backtrack(stack)
    assert deref(stack, X) == X


# ---------------------------------------------------------------------------
# Variable aliasing and chains
# ---------------------------------------------------------------------------


def test_variable_variable_aliasing():
    stack = init_stack()

    X = make_var(stack)
    Y = make_var(stack)

    assert unify_status(stack, X, Y) == WAMStatus.SUCCESS

    # X and Y should now dereference to the same root
    assert deref(stack, X) == deref(stack, Y)


def test_variable_chain_binding():
    """
    X = Y, Y = Z, Z = a
    """
    stack = init_stack()

    X = make_var(stack)
    Y = make_var(stack)
    Z = make_var(stack)
    a = make_const(stack, 10)

    assert unify_status(stack, X, Y) == WAMStatus.SUCCESS
    assert unify_status(stack, Y, Z) == WAMStatus.SUCCESS
    assert unify_status(stack, Z, a) == WAMStatus.SUCCESS

    assert deref(stack, X) == a
    assert deref(stack, Y) == a
    assert deref(stack, Z) == a


# ---------------------------------------------------------------------------
# Repeated variable constraints
# ---------------------------------------------------------------------------


def test_repeated_variable_success():
    """
    f(X, X) = f(a, a)
    """
    stack = init_stack()

    X = make_var(stack)

    F = 1
    a = make_const(stack, 10)

    f_fun = make_functor(stack, F, 2)

    left = make_structure(stack, f_fun, np.array([X, X], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([a, a], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.SUCCESS
    assert deref(stack, X) == a


def test_repeated_variable_failure():
    """
    f(X, X) = f(a, b)  should fail
    """
    stack = init_stack()

    X = make_var(stack)

    F = 1
    a = make_const(stack, 10)
    b = make_const(stack, 11)

    f_fun = make_functor(stack, F, 2)

    left = make_structure(stack, f_fun, np.array([X, X], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([a, b], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.EXHAUSTED


def test_variable_can_bind_to_structure_containing_itself_without_freezing():
    stack = init_stack()

    X = make_var(stack)
    f_fun = make_functor(stack, 1, 1)
    f_x = make_structure(stack, f_fun, np.array([X], dtype=np.uint16))

    assert unify_status(stack, X, f_x) == WAMStatus.SUCCESS
    assert deref(stack, X) == f_x


def test_unify_step_limit_bounds_large_work():
    config = load_config_toml(
        """
        [solver]
        max_unify_steps = 1
        """
    )
    stack = init_stack(config)

    X = make_var(stack)
    Y = make_var(stack)
    f_fun = make_functor(stack, 1, 2)
    left = make_structure(stack, f_fun, np.array([X, Y], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([X, Y], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.UNIFY_STEP_LIMIT


# ---------------------------------------------------------------------------
# Arity mismatch
# ---------------------------------------------------------------------------


def test_structure_arity_mismatch():
    """
    f(X) != f(X, Y)
    """
    stack = init_stack()

    X = make_var(stack)
    Y = make_var(stack)

    F = 1

    f1_fun = make_functor(stack, F, 1)
    f2_fun = make_functor(stack, F, 2)

    s1 = make_structure(stack, f1_fun, np.array([X], dtype=np.uint16))
    s2 = make_structure(stack, f2_fun, np.array([X, Y], dtype=np.uint16))

    assert unify_status(stack, s1, s2) == WAMStatus.EXHAUSTED


# ---------------------------------------------------------------------------
# Deep nesting
# ---------------------------------------------------------------------------


def test_deeply_nested_structures():
    """
    f(g(h(X))) = f(g(h(a)))
    """
    stack = init_stack()

    X = make_var(stack)

    F = 1
    G = 2
    H = 3
    a = make_const(stack, 10)

    f_fun = make_functor(stack, F, 1)
    g_fun = make_functor(stack, G, 1)
    h_fun = make_functor(stack, H, 1)

    h_X = make_structure(stack, h_fun, np.array([X], dtype=np.uint16))
    g_h_X = make_structure(stack, g_fun, np.array([h_X], dtype=np.uint16))
    left = make_structure(stack, f_fun, np.array([g_h_X], dtype=np.uint16))

    h_a = make_structure(stack, h_fun, np.array([a], dtype=np.uint16))
    g_h_a = make_structure(stack, g_fun, np.array([h_a], dtype=np.uint16))
    right = make_structure(stack, f_fun, np.array([g_h_a], dtype=np.uint16))

    assert unify_status(stack, left, right) == WAMStatus.SUCCESS
    assert deref(stack, X) == a


# ---------------------------------------------------------------------------
# Backtracking with multiple bindings
# ---------------------------------------------------------------------------


def test_backtracking_multiple_bindings():
    stack = init_stack()

    X = make_var(stack)
    Y = make_var(stack)

    a = make_const(stack, 10)
    b = make_const(stack, 11)

    helpertest_push_choice_point(stack)

    assert unify_status(stack, X, a) == WAMStatus.SUCCESS
    assert unify_status(stack, Y, b) == WAMStatus.SUCCESS

    assert deref(stack, X) == a
    assert deref(stack, Y) == b

    helpertest_backtrack(stack)

    assert deref(stack, X) == X
    assert deref(stack, Y) == Y


# ---------------------------------------------------------------------------
# Failure should not corrupt heap
# ---------------------------------------------------------------------------


def test_failed_unification_does_not_bind():
    """
    X = a, then try unify(X, b) -> fail, X should remain bound to a
    """
    stack = init_stack()

    X = make_var(stack)
    a = make_const(stack, 10)
    b = make_const(stack, 11)

    assert unify_status(stack, X, a) == WAMStatus.SUCCESS
    assert deref(stack, X) == a

    assert unify_status(stack, X, b) == WAMStatus.EXHAUSTED
    assert deref(stack, X) == a
