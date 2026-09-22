import numpy as np
import pytest
from numba import jit
from wampy.config import load_config_toml, resolve_symbol_id_dtype
from wampy.frontend.ast.ops.validation import is_valid_program
from wampy.frontend.ast.program import NO_NODE, ASTProgram, init_ast_program
from wampy.frontend.ast.symbol_id import CoreSymID, calculate_symbol_id_positions


@jit
def _init_both_from_jit(config):
    return (
        init_ast_program(config, num_terms=2),
        init_ast_program(config),
    )


@pytest.mark.parametrize(
    ("dtype_name", "expected_dtype"),
    [
        ("uint16", np.dtype(np.uint16)),
        ("uint32", np.dtype(np.uint32)),
    ],
)
def test_symbol_storage_dtype_is_configurable(dtype_name, expected_dtype):
    symbol_dtype = resolve_symbol_id_dtype(dtype_name)
    config = load_config_toml(f"[frontend.ast]\nsymbol_id_dtype = {dtype_name!r}")
    assert isinstance(config.frontend.ast.symbol_id_dtype, np.dtype)
    program = init_ast_program(config)

    assert symbol_dtype == expected_dtype
    assert program.node_symbols.dtype == expected_dtype
    assert np.all(program.node_symbols == CoreSymID.NO_SYMBOL)
    assert is_valid_program(program)
    assert np.zeros((2, 3), dtype=symbol_dtype).dtype == expected_dtype


@pytest.mark.parametrize("dtype_name", ["int16", "float32", "uint4"])
def test_invalid_symbol_storage_dtypes_are_rejected(dtype_name):
    with pytest.raises(ValueError):
        load_config_toml(f"[frontend.ast]\nsymbol_id_dtype = {dtype_name!r}")


def test_reserved_symbol_ids_do_not_contain_allocation_bounds():
    assert CoreSymID.NO_SYMBOL == 0
    assert CoreSymID.ARGS == 2
    assert set(CoreSymID.__members__) == {
        "NO_SYMBOL",
        "TRUE",
        "ARGS",
        "NEGATION_AS_FAILURE",
        "CUT",
        "CONJUNCTION",
        "CLAUSE",
        "QUERY",
    }

    assert calculate_symbol_id_positions(32) == (16, 48)

    program = init_ast_program(load_config_toml())
    assert program.first_variable_symbol_id == 16
    assert program.first_user_symbol_id == 48


@pytest.mark.parametrize(
    ("dtype_name", "expected_dtype"),
    [
        ("uint16", np.dtype(np.uint16)),
        ("uint32", np.dtype(np.uint32)),
    ],
)
def test_ast_initializer_is_callable(dtype_name, expected_dtype):
    config = load_config_toml(f"[frontend.ast]\nsymbol_id_dtype = {dtype_name!r}")

    direct_program, configured_program = _init_both_from_jit(config)

    assert direct_program.node_symbols.dtype == expected_dtype
    assert configured_program.node_symbols.dtype == expected_dtype
    for program in (direct_program, configured_program):
        assert isinstance(program, ASTProgram)
        assert program.node_links.dtype == np.dtype(np.uint16)
        assert program.node_links.shape[2] == 2
        assert program.free_node_stack.dtype == np.dtype(np.uint16)
        assert program.free_node_count.dtype == np.dtype(np.uint16)
        assert np.all(program.node_symbols == CoreSymID.NO_SYMBOL)
        assert np.all(program.node_links == NO_NODE)


def test_ast_initialization_has_one_dtype_genericementation():
    import wampy.frontend.ast.program as program_module

    assert not hasattr(program_module, "init_ast_structure")
    assert not hasattr(program_module, "init_program")
    assert not any(name.startswith("_init_ast_structure_") for name in dir(program_module))
