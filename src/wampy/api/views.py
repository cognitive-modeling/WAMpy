"""Self-rendering views for high-level WAMpy objects."""

__all__ = [
    "ASTView",
    "CodeView",
]

from dataclasses import dataclass

from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiler_state import CompilerState
from wampy.compiler.diagnostics import render_compiler_state
from wampy.frontend.ast.program import ASTProgram
from wampy.frontend.render import render
from wampy.frontend.symbol_table import SymbolTable


@dataclass(frozen=True, slots=True)
class ASTView:
    """An AST together with the symbols needed to render it."""

    program: ASTProgram
    symbol_table: SymbolTable

    def __str__(self) -> str:
        return render(self.program, self.symbol_table)

    def __repr__(self) -> str:
        return "ASTView(program=..., symbol_table=...)"

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text(str(self))


@dataclass(frozen=True, slots=True)
class CodeView:
    """A compiler state together with the symbols needed to render it."""

    compiled_program: CompiledProgram
    compiler_state: CompilerState
    symbol_table: SymbolTable

    def __str__(self) -> str:
        return render_compiler_state(
            self.compiled_program,
            self.compiler_state,
            self.symbol_table,
        )

    def __repr__(self) -> str:
        return "CodeView(compiled_program=..., compiler_state=..., symbol_table=...)"

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text(str(self))
