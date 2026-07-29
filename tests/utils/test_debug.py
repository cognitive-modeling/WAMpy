# utils/test_debug_wam.py

import pytest
import numpy as np


from wampy.runtime.stack import (
    init_stack,
    make_var,
    make_const,
    make_functor,
    make_structure,
)
from wampy.utils.debug import debug_term_from_stack


# -------------------------------------------------
# Debug printing
# -------------------------------------------------


def test_debug_term_variable():
    stack = init_stack()
    x = make_var(stack)

    result = debug_term_from_stack(stack, x, {})
    assert result == "_"


def test_debug_term_constant():
    stack = init_stack()
    c = make_const(stack, 10)

    result = debug_term_from_stack(stack, c, {10: "a"})
    assert result == "a"


def test_debug_term_structure():
    stack = init_stack()

    ATOMS = {1: "f", 10: "a", 20: "b"}

    x = make_var(stack)
    y = make_var(stack)

    f = make_functor(stack, 1, 2)
    a = make_const(stack, 10)
    b = make_const(stack, 20)

    s = make_structure(stack, f, np.array([a, b], dtype=np.uint16))

    result = debug_term_from_stack(stack, s, ATOMS)
    assert result == "f(a, b)"
