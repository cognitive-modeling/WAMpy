import pytest
import wampy
from wampy.api.views import ASTView, CodeView


class PrettyPrinter:
    def __init__(self) -> None:
        self.output = ""

    def text(self, value: str) -> None:
        self.output += value


@pytest.fixture
def prolog() -> wampy.Prolog:
    return wampy.Prolog("parent(anakin, luke).")


def test_prolog_exposes_contextual_ast_and_code_views(prolog: wampy.Prolog) -> None:
    ast_view = prolog.ast
    code_view = prolog.code

    assert isinstance(ast_view, ASTView)
    assert ast_view.program is prolog.program
    assert ast_view.symbol_table is prolog.symbol_table

    assert isinstance(code_view, CodeView)
    assert code_view.compiled_program is prolog.compiled_program
    assert code_view.compiler_state is prolog.compiler_state
    assert code_view.symbol_table is prolog.symbol_table


def test_ast_view_renders_with_str_and_print(
    prolog: wampy.Prolog,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rendered = str(prolog.ast)

    assert "parent(anakin, luke)" in rendered
    assert capsys.readouterr().out == ""

    print(prolog.ast)

    assert capsys.readouterr().out == rendered + "\n"


def test_code_view_renders_with_str_and_print(
    prolog: wampy.Prolog,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rendered = str(prolog.code)

    assert "=== Predicate entry points ===" in rendered
    assert "parent/2 -> PC" in rendered
    assert "=== WAM instructions ===" in rendered
    assert capsys.readouterr().out == ""

    print(prolog.code)

    assert capsys.readouterr().out == rendered + "\n"


@pytest.mark.parametrize("attribute", ["ast", "code"])
def test_views_support_ipython_pretty_protocol(
    prolog: wampy.Prolog,
    attribute: str,
) -> None:
    view = getattr(prolog, attribute)
    printer = PrettyPrinter()

    view._repr_pretty_(printer, cycle=False)

    assert printer.output == str(view)


def test_view_repr_is_concise(prolog: wampy.Prolog) -> None:
    assert repr(prolog.ast) == "ASTView(program=..., symbol_table=...)"
    assert repr(prolog.code) == (
        "CodeView(compiled_program=..., compiler_state=..., symbol_table=...)"
    )
