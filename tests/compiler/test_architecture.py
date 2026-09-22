import importlib

from wampy.compiler.codegen.clause import compile_clause
from wampy.compiler.codegen.goal import compile_goal
from wampy.compiler.compiler import compile_program
from wampy.compiler.predicates import find_predicate_slot
from wampy.compiler.program_plan import build_program_compile_plan
from wampy.compiler.query import compile_query


def test_compiler_functions_live_in_their_canonical_modules() -> None:
    assert compile_program.__module__ == "wampy.compiler.compiler"
    assert compile_query.__module__ == "wampy.compiler.query"
    assert find_predicate_slot.__module__ == "wampy.compiler.predicates"
    assert build_program_compile_plan.__module__ == "wampy.compiler.program_plan"
    assert compile_clause.__module__ == "wampy.compiler.codegen.clause"
    assert compile_goal.__module__ == "wampy.compiler.codegen.goal"


def test_compiler_module_does_not_reexport_moved_implementations() -> None:
    compiler = importlib.import_module("wampy.compiler.compiler")

    assert not hasattr(compiler, "compile_query")
    assert not hasattr(compiler, "compile_clause")
    assert not hasattr(compiler, "find_predicate_slot")
