import numpy as np
import pytest
from wampy.frontend.symbol_table import (
    empty_symbol_table,
    symbol_table_from_mapping,
)
from wampy.runtime.debug import format_heap_term
from wampy.runtime.machine import init_machine
from wampy.runtime.machine.heap import (
    make_const,
    make_functor,
    make_structure,
    make_var,
)


@pytest.fixture
def symbol_table():
    return symbol_table_from_mapping(
        {
            1: "f",
            2: "g",
            10: "a",
            20: "b",
        }
    )


# -------------------------------------------------
# Variables
# -------------------------------------------------


def test_format_heap_term_unbound_variable():
    memory = init_machine()
    state = memory.registers
    variable = make_var(state, memory)

    result = format_heap_term(
        state,
        memory,
        variable,
        empty_symbol_table(),
    )

    assert result == "_"


def test_format_heap_term_variable_bound_to_constant(symbol_table):
    memory = init_machine()
    state = memory.registers

    variable = make_var(state, memory)
    constant = make_const(state, memory, 10)
    memory.heap.cells[variable] = constant

    result = format_heap_term(state, memory, variable, symbol_table)

    assert result == "a"


def test_format_heap_term_dereferences_reference_chain(symbol_table):
    memory = init_machine()
    state = memory.registers

    first = make_var(state, memory)
    second = make_var(state, memory)
    constant = make_const(state, memory, 10)

    memory.heap.cells[first] = second
    memory.heap.cells[second] = constant

    result = format_heap_term(state, memory, first, symbol_table)

    assert result == "a"


def test_format_heap_term_shared_unbound_variable(symbol_table):
    memory = init_machine()
    state = memory.registers

    variable = make_var(state, memory)
    functor = make_functor(state, memory, 1, 2)
    structure = make_structure(
        state,
        memory,
        functor,
        np.array([variable, variable], dtype=np.uint16),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "f(_, _)"


# -------------------------------------------------
# Constants
# -------------------------------------------------


def test_format_heap_term_known_constant(symbol_table):
    memory = init_machine()
    state = memory.registers
    constant = make_const(state, memory, 10)

    result = format_heap_term(state, memory, constant, symbol_table)

    assert result == "a"


@pytest.mark.parametrize(
    ("symbol_id", "expected"),
    [
        (0, "0"),
        (42, "42"),
        (255, "255"),
    ],
)
def test_format_heap_term_unknown_constant_uses_numeric_fallback(
    symbol_id,
    expected,
):
    memory = init_machine()
    state = memory.registers
    constant = make_const(state, memory, symbol_id)

    result = format_heap_term(
        state,
        memory,
        constant,
        empty_symbol_table(),
    )

    assert result == expected


# -------------------------------------------------
# Structures
# -------------------------------------------------


def test_format_heap_term_structure(symbol_table):
    memory = init_machine()
    state = memory.registers

    functor = make_functor(state, memory, 1, 2)
    first_argument = make_const(state, memory, 10)
    second_argument = make_const(state, memory, 20)

    structure = make_structure(
        state,
        memory,
        functor,
        np.array(
            [first_argument, second_argument],
            dtype=np.uint16,
        ),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "f(a, b)"


def test_format_heap_term_nested_structure(symbol_table):
    memory = init_machine()
    state = memory.registers

    constant = make_const(state, memory, 10)

    inner_functor = make_functor(state, memory, 2, 1)
    inner_structure = make_structure(
        state,
        memory,
        inner_functor,
        np.array([constant], dtype=np.uint16),
    )

    outer_functor = make_functor(state, memory, 1, 1)
    outer_structure = make_structure(
        state,
        memory,
        outer_functor,
        np.array([inner_structure], dtype=np.uint16),
    )

    result = format_heap_term(
        state,
        memory,
        outer_structure,
        symbol_table,
    )

    assert result == "f(g(a))"


def test_format_heap_term_structure_with_variable(symbol_table):
    memory = init_machine()
    state = memory.registers

    variable = make_var(state, memory)
    constant = make_const(state, memory, 10)
    functor = make_functor(state, memory, 1, 2)

    structure = make_structure(
        state,
        memory,
        functor,
        np.array([variable, constant], dtype=np.uint16),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "f(_, a)"


def test_format_heap_term_structure_with_bound_variable(symbol_table):
    memory = init_machine()
    state = memory.registers

    variable = make_var(state, memory)
    constant = make_const(state, memory, 10)
    memory.heap.cells[variable] = constant

    functor = make_functor(state, memory, 1, 1)
    structure = make_structure(
        state,
        memory,
        functor,
        np.array([variable], dtype=np.uint16),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "f(a)"


def test_format_heap_term_zero_arity_structure():
    memory = init_machine()
    state = memory.registers
    symbol_table = symbol_table_from_mapping({5: "unit"})

    functor = make_functor(state, memory, 5, 0)
    structure = make_structure(
        state,
        memory,
        functor,
        np.empty(0, dtype=np.uint16),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "unit()"


def test_format_heap_term_unknown_functor_uses_numeric_fallback():
    memory = init_machine()
    state = memory.registers
    symbol_table = symbol_table_from_mapping({10: "a"})

    argument = make_const(state, memory, 10)
    functor = make_functor(state, memory, 99, 1)
    structure = make_structure(
        state,
        memory,
        functor,
        np.array([argument], dtype=np.uint16),
    )

    result = format_heap_term(state, memory, structure, symbol_table)

    assert result == "99(a)"


# -------------------------------------------------
# Unsupported heap cells
# -------------------------------------------------


def test_format_heap_term_unsupported_tag_returns_placeholder():
    memory = init_machine()
    state = memory.registers

    functor_address = make_functor(
        state,
        memory,
        symbol_id=1,
        arity=2,
    )

    result = format_heap_term(
        state,
        memory,
        functor_address,
        empty_symbol_table(),
    )

    assert result == "<?>"
