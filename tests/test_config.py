import numpy as np
import pytest
import wampy as wam
from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiled_query import init_compiled_query
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.query import compile_query
from wampy.config import (
    DEFAULT_CONFIG,
    ASTConfig,
    CompilerConfig,
    FrontendConfig,
    RuntimeConfig,
    WAMConfig,
    apply_overrides,
    debug_config,
    load_config_toml,
)
from wampy.runtime.interpreter import run
from wampy.runtime.machine import init_machine
from wampy.status import WAMStatus


def test_default_config_has_only_subsystem_sections():
    assert WAMConfig._fields == ("frontend", "compiler", "runtime")

    assert ASTConfig().symbol_id_dtype == np.dtype(np.uint16)
    assert ASTConfig().child_block_size == 4

    assert WAMConfig().compiler.max_arity == 16
    assert WAMConfig().runtime.max_steps == 10_000_000

    assert isinstance(DEFAULT_CONFIG.frontend, FrontendConfig)
    assert isinstance(DEFAULT_CONFIG.frontend.ast, ASTConfig)
    assert isinstance(DEFAULT_CONFIG.compiler, CompilerConfig)
    assert isinstance(DEFAULT_CONFIG.runtime, RuntimeConfig)

    assert DEFAULT_CONFIG == WAMConfig()
    assert load_config_toml() == WAMConfig()
    assert load_config_toml("  \n") == WAMConfig()


@pytest.mark.parametrize(
    "dtype",
    [np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.uint32)],
)
def test_direct_ast_config_is_accepted_by_jit_init(dtype):
    from wampy.frontend.ast.program import init_ast_program

    config = WAMConfig(
        frontend=FrontendConfig(ast=ASTConfig(symbol_id_dtype=dtype)),
    )
    program = init_ast_program(config)

    assert program.node_symbols.dtype == dtype


def test_default_ast_config_is_accepted_by_jit_init():
    from wampy.frontend.ast.program import init_ast_program

    program = init_ast_program(WAMConfig())

    assert program.node_symbols.dtype == np.dtype(np.uint16)


# @pytest.mark.requires_numba_jit
# def test_default_ast_config_init_ast_program_compiles_in_nopython():
#     from wampy.frontend.ast.program import init_ast_program

#     init_ast_program(WAMConfig())

#     assert init_ast_program.nopython_signatures


def test_symbol_dtype_override_is_resolved_before_jit_init():
    from wampy.frontend.ast.program import init_ast_program

    config = apply_overrides(
        None,
        {
            "frontend": {
                "ast": {
                    "symbol_id_dtype": "uint8",
                    "max_user_symbols": 192,
                }
            }
        },
    )

    assert config.frontend.ast.symbol_id_dtype == np.dtype(np.uint8)
    assert init_ast_program(config).node_symbols.dtype == np.dtype(np.uint8)
    assert load_config_toml(debug_config(config)) == config


def test_load_config_toml_applies_nested_overrides():
    config = load_config_toml(
        """
        [frontend.ast]
        max_terms = 200
        max_nodes_per_term = 600
        child_block_size = 4

        [compiler]
        max_arity = 16
        max_instructions = 8192

        [runtime]
        heap_size = 131072
        max_steps = 50000
        max_x_registers = 512
        max_answers = 20
        trace = true
        """
    )

    assert config.frontend.ast.max_terms == 200
    assert config.frontend.ast.max_nodes_per_term == 600
    assert config.frontend.ast.child_block_size == 4
    assert config.compiler.max_arity == 16
    assert config.compiler.max_instructions == 8192
    assert config.runtime.heap_size == 131072
    assert config.runtime.max_steps == 50_000
    assert config.runtime.max_x_registers == 512
    assert config.runtime.max_answers == 20
    assert config.runtime.trace is True
    assert config.runtime.trail_size == WAMConfig().runtime.trail_size


def test_load_config_toml_accepts_wam_wrapper_table():
    config = load_config_toml(
        """
        [wam.frontend.ast]
        max_terms = 40

        [wam.runtime]
        heap_size = 2048
        max_steps = 3000
        """
    )

    assert config.frontend.ast.max_terms == 40
    assert config.runtime.heap_size == 2048
    assert config.runtime.max_steps == 3000


def test_load_config_toml_merges_into_base_without_mutating_it():
    base = load_config_toml(
        """
        [runtime]
        max_steps = 25000
        """
    )

    config = load_config_toml(
        """
        [runtime]
        heap_size = 4096
        """,
        base=base,
    )

    assert config.runtime.heap_size == 4096
    assert config.runtime.max_steps == 25_000
    assert base.runtime.heap_size == WAMConfig().runtime.heap_size


def test_apply_overrides_merges_nested_mappings():
    config = apply_overrides(
        None,
        {
            "compiler": {"max_arity": 8},
            "runtime": {"max_answers": 20, "trace": True},
        },
    )

    assert config.compiler.max_arity == 8
    assert config.runtime.max_answers == 20
    assert config.runtime.trace is True
    assert config.frontend == WAMConfig().frontend


@pytest.mark.parametrize(
    "toml_text",
    [
        "unknown = 1",
        "[runtime]\nunknown = 1",
        "[wam.compiler]\nunknown = 1",
        "[frontend.ast]\nold_max_terms = 1",
    ],
)
def test_load_config_toml_rejects_unknown_keys(toml_text):
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config_toml(toml_text)


def test_apply_overrides_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_overrides(None, {"runtime": {"unknown": 1}})


@pytest.mark.parametrize(
    ("toml_text", "message"),
    [
        ("wam = 1", "'wam' must be a table/mapping"),
        ("runtime = 1", "Expected table for 'runtime'"),
    ],
)
def test_load_config_toml_rejects_invalid_table_shapes(toml_text, message):
    with pytest.raises((TypeError, ValueError), match=message):
        load_config_toml(toml_text)


def test_load_config_toml_wraps_toml_parse_errors():
    with pytest.raises(ValueError, match="Invalid TOML"):
        load_config_toml("[runtime")


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("frontend.ast", "max_terms"),
        ("frontend.ast", "max_nodes_per_term"),
        ("frontend.ast", "child_block_size"),
        ("frontend.ast", "max_variables_per_term"),
        ("frontend.ast", "max_user_symbols"),
        ("compiler", "max_instructions"),
        ("compiler", "max_predicates"),
        ("compiler", "max_arity"),
        ("compiler", "max_goal_depth"),
        ("compiler", "max_structure_depth"),
        ("runtime", "heap_size"),
        ("runtime", "trail_size"),
        ("runtime", "choice_point_size"),
        ("runtime", "unify_stack_size"),
        ("runtime", "max_x_registers"),
        ("runtime", "max_structure_depth"),
        ("runtime", "max_steps"),
        ("runtime", "max_unify_steps"),
        ("runtime", "max_answers"),
        ("runtime", "max_answer_nodes"),
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_load_config_toml_rejects_non_positive_limits(section, key, value):
    with pytest.raises(ValueError, match=rf"{section}\.{key} must be > 0"):
        load_config_toml(f"[{section}]\n{key} = {value}")


def test_load_config_toml_validates_a_supplied_base():
    invalid_base = WAMConfig(
        runtime=WAMConfig().runtime._replace(heap_size=0),
    )

    with pytest.raises(ValueError, match=r"runtime\.heap_size must be > 0"):
        load_config_toml(base=invalid_base)


def test_load_config_toml_rejects_compiler_arity_above_runtime_registers():
    with pytest.raises(
        ValueError,
        match="compiler.max_arity must not exceed runtime.max_x_registers",
    ):
        load_config_toml(
            """
            [compiler]
            max_arity = 5

            [runtime]
            max_x_registers = 4
            """
        )


def test_debug_config_round_trips_through_toml():
    config = load_config_toml(
        """
        [frontend.ast]
        max_terms = 50
        max_nodes_per_term = 150

        [compiler]
        max_goal_depth = 128

        [runtime]
        max_answers = 3
        trace = true
        """
    )

    assert load_config_toml(debug_config(config)) == config


def test_debug_config_formats_booleans_as_toml_literals():
    rendered = debug_config(WAMConfig(runtime=WAMConfig().runtime._replace(trace=True)))

    assert "trace = true" in rendered
    assert "trace = false" not in rendered


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_debug_config_rejects_non_finite_floats(value):
    with pytest.raises(ValueError, match="TOML does not support NaN/Inf"):
        debug_config({"value": value})


def test_debug_config_rejects_unsupported_values():
    with pytest.raises(TypeError, match="Unsupported TOML value type"):
        debug_config({"value": object()})


def test_ast_symbol_dtype_is_preserved_when_config_changes():
    from wampy.frontend.ast.program import init_ast_program

    default_program = init_ast_program(DEFAULT_CONFIG)
    narrow_config = DEFAULT_CONFIG._replace(
        frontend=DEFAULT_CONFIG.frontend._replace(
            ast=DEFAULT_CONFIG.frontend.ast._replace(
                max_terms=12,
                symbol_id_dtype=np.dtype(np.uint8),
            ),
        ),
    )
    narrow_program = init_ast_program(narrow_config)
    assert default_program.node_symbols.dtype == np.uint16
    assert narrow_program.node_symbols.dtype == np.uint8


def test_public_operations_accept_one_complete_wam_config() -> None:
    config = load_config_toml(
        """
        [frontend.ast]
        max_terms = 8

        [compiler]
        max_instructions = 128

        [runtime]
        heap_size = 128
        max_answers = 1
        max_answer_nodes = 32
        """
    )

    program, symbols = wam.parse(
        "parent(anakin, luke).",
        wam.empty_symbol_table(config.frontend.ast),
        config=config,
    )

    compiler_state = init_compiler_state(config)

    compiled_program = init_compiled_program(config)
    compile_program(
        program,
        compiled_program,
        compiler_state,
    )

    query_program, _ = wam.parse(
        "?- parent(X, luke).",
        symbols,
        config,
    )

    compiled_query = init_compiled_query(config)
    compile_query(
        query_program,
        compiled_query,
        compiled_program,
        compiler_state,
        config,
    )

    machine = init_machine(config.runtime)

    status = run(
        machine,
        compiled_program,
        compiled_query,
    )

    assert status == WAMStatus.SUCCESS
