import inspect

import wampy as wam
import wampy.api.ast as public_ast
import wampy.frontend.ast as frontend_ast
from wampy.config import WAMConfig
from wampy.frontend.ast.ops import analysis, transforms


def test_ast_exposes_flat_operation_api():
    assert callable(public_ast.get_clause_head_body)
    assert callable(public_ast.get_child_at)
    assert callable(public_ast.get_child_count)
    assert callable(public_ast.get_first_child)
    assert callable(public_ast.count_clauses)
    assert callable(public_ast.is_variable_symbol)
    assert callable(public_ast.add_child_node)
    assert callable(public_ast.remove_descendants)
    assert callable(public_ast.copy_term)


def test_public_ast_is_a_static_facade_over_frontend_ast():
    assert public_ast is not frontend_ast
    assert wam.ast is public_ast
    assert public_ast.get_child_count is analysis.get_child_count
    assert public_ast.copy_term is transforms.copy_term
    assert inspect.signature(public_ast.get_child_count) == inspect.signature(
        analysis.get_child_count
    )
    assert not hasattr(frontend_ast, "get_child_count")
    assert not hasattr(frontend_ast, "copy_term")


def test_from_wampy_import_ast_is_canonical():
    from wampy.api import ast

    assert ast is public_ast


def test_flat_analysis_api_executes():
    program, _symbol_table = wam.parse("male(anakin).", wam.empty_symbol_table())

    head_id, body_id = public_ast.get_clause_head_body(program, 0)

    assert head_id >= 0
    assert body_id >= 0
    assert public_ast.get_child_count(program, 0, head_id) == 1


def test_removed_operation_names_are_not_exported():
    removed_names = (
        "get_" + "arity_of_node",
        "clause_" + "head_body",
        "get_" + "count_clauses",
        "is_sym_" + "a_var",
        "is_" + "compound",
        "is_structure_" + "of_terms_equal",
        "remove_all_" + "descendant",
        "copy_and_" + "override_term",
        "reset_metadata_term",
        "copy_term_metadata",
    )

    for name in removed_names:
        assert not hasattr(public_ast, name)
        assert not hasattr(frontend_ast, name)


def test_operation_modules_are_not_part_of_public_api():
    for name in ("analysis", "mutation", "validation", "transforms"):
        assert not hasattr(public_ast, name)
        assert not hasattr(frontend_ast, name)


def test_ast_all_contains_only_supported_public_names():
    assert "get_child_at" in public_ast.__all__
    assert "get_child_count" in public_ast.__all__
    assert "get_first_child" in public_ast.__all__
    assert "add_child_node" in public_ast.__all__
    assert "copy_term" in public_ast.__all__

    assert "get_" + "arity_of_node" not in public_ast.__all__
    assert "analysis" not in public_ast.__all__
    assert "mutation" not in public_ast.__all__


def test_public_ast_mutation_runs_on_an_independent_program():
    program = public_ast.init_ast_program(WAMConfig())

    child_id, ok = public_ast.add_child_node(
        program,
        0,
        0,
        program.first_variable_symbol_id,
    )

    assert ok
    assert child_id != public_ast.NO_NODE
    assert program.first_variable_symbol_id == 16
