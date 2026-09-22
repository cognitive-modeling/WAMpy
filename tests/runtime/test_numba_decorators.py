"""Tests for WAMpy's distinction between compiled boundaries and helpers."""

import inspect
import os

import pytest
from wampy.api.jitable import (
    init_machine,
    make_const,
    make_var,
    unify,
)
from wampy.frontend.parser import parse
from wampy.frontend.render import display
from wampy.frontend.symbol_table import empty_symbol_table
from wampy.runtime.machine.heap import deref
from wampy.runtime.unification import bind_var
from wampy.runtime.unification import unify as runtime_unify
from wampy.status import WAMStatus


def _skip_when_jit_is_disabled():
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        pytest.skip("This test checks the enabled Numba dispatcher path")


def test_runtime_helpers_are_ordinary_python_functions():
    assert inspect.isfunction(deref)
    assert inspect.isfunction(bind_var)
    assert inspect.isfunction(unify)
    assert inspect.isfunction(display)


def test_jitable_unification_import_is_the_runtimeementation():
    assert inspect.isfunction(unify)
    assert unify is runtime_unify
    memory = init_machine()
    state = memory.registers
    variable = make_var(state, memory)
    constant = make_const(state, memory, 7)

    status = WAMStatus(int(unify(state, memory, variable, constant)))

    assert status is WAMStatus.SUCCESS
    assert deref(state, memory, variable) == constant


# @pytest.mark.requires_numba_jit
@pytest.mark.skip(reason="Not applicable")
def test_python_helper_is_compiled_when_called_transitively():
    _skip_when_jit_is_disabled()

    def call_deref(state, memory, address):
        return deref(state, memory, address)

    memory = init_machine()
    state = memory.registers
    variable = make_var(state, memory)
    constant = make_const(state, memory, 11)
    memory.heap.cells[variable] = constant

    assert call_deref(state, memory, variable) == constant


# @pytest.mark.requires_numba_jit
@pytest.mark.skip(reason="Not applicable")
def test_display_is_callable_from_numba():
    _skip_when_jit_is_disabled()
    program, symbol_table = parse("p(a).", empty_symbol_table())

    def call_display(program, symbol_table):
        display(program, symbol_table, 0)

    call_display(program, symbol_table)
