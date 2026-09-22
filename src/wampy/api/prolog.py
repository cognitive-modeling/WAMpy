"""Compiled knowledge bases for the high-level WAMpy API."""

import numpy as np

from wampy.compiler.compiled_program import init_compiled_program
from wampy.compiler.compiled_query import CompiledQuery, init_compiled_query
from wampy.compiler.compiler import compile_program
from wampy.compiler.compiler_state import init_compiler_state
from wampy.compiler.query import compile_query
from wampy.config import DEFAULT_CONFIG, WAMConfig, load_config_toml
from wampy.frontend.ast.ops.analysis import get_term_count
from wampy.frontend.parser import parse
from wampy.frontend.solution import SolutionType
from wampy.frontend.symbol_table import empty_symbol_table

from .query import Query
from .views import ASTView, CodeView


def _query_source(src: str) -> str:
    """Return one explicit Prolog query term for the source string."""
    stripped = src.strip()
    if not stripped:
        return src
    if stripped.startswith("?-"):
        return stripped
    if stripped.endswith("."):
        stripped = stripped[:-1].rstrip()
    return f"?- {stripped}."


class Prolog:
    """A compiled knowledge base that can own many independent queries."""

    def __init__(self, src: str = "", config: WAMConfig = DEFAULT_CONFIG):

        self.config = load_config_toml(base=config)

        # 1. Parse
        self.symbol_table = empty_symbol_table(self.config.frontend.ast)
        self.program, self.symbol_table = parse(src, self.symbol_table, self.config)

        # 2. Compile
        self.compiled_program = init_compiled_program(self.config)
        self.compiler_state = init_compiler_state(self.config)
        compile_program(
            program=self.program,
            compiled_program=self.compiled_program,
            compiler_state=self.compiler_state,
            config=self.config,
        )

    @property
    def ast(self) -> ASTView:
        """Return the program and symbol context needed for AST display."""

        return ASTView(
            self.program,
            self.symbol_table,
        )

    @property
    def code(self) -> CodeView:
        """Return the compiler and symbol context needed for code display."""

        return CodeView(
            self.compiled_program,
            self.compiler_state,
            self.symbol_table,
        )

    def prepare(self, src: str) -> CompiledQuery:
        """Parse and compile one query against this compiled knowledge base."""

        program, self.symbol_table = parse(
            _query_source(src),
            self.symbol_table,
            self.config,
        )
        if get_term_count(program) != 1:
            if not src.strip():
                raise ValueError("Empty query")
            raise ValueError("Query must contain exactly one term")

        compiled_query = init_compiled_query(self.config)
        compile_query(
            program,
            compiled_query,
            self.compiled_program,
            self.compiler_state,
            self.config,
        )
        return compiled_query

    def query(self, src: str | CompiledQuery) -> Query:
        """Return a lazy iterator for one source or prepared query."""

        compiled_query = self.prepare(src) if isinstance(src, str) else src

        return Query(self, compiled_query)

    def query_once(self, src: str | CompiledQuery) -> SolutionType | None:
        query = self.query(src)
        try:
            return next(query)
        except StopIteration:
            return None
        finally:
            query.close()

    def query_all(self, src: str | CompiledQuery) -> list[SolutionType]:
        query: Query = self.query(src)
        try:
            return list(query)
        finally:
            query.close()

    def exists(self, src: str | CompiledQuery) -> bool:
        query: Query = self.query(src=src)
        try:
            try:
                next(query)
            except StopIteration:
                return False
            return True
        finally:
            query.close()

    def query_batch(
        self,
        queries,
        mode: str = "exists",
        parallel: bool = False,
    ):
        """Evaluate independent queries in a compact, serial batch shape.

        A later batch executor may add parallel lanes without changing the
        result shapes. Until then, ``parallel=True`` is rejected explicitly.
        """

        if parallel:
            raise NotImplementedError(
                "parallel batch execution is not implemented; use parallel=False"
            )
        mode = str(mode).lower()
        if mode not in {"exists", "first", "count", "all"}:
            raise ValueError("mode must be one of: exists, first, count, all")

        results = []
        for query_source in queries:
            if mode == "exists":
                results.append(self.exists(query_source))
            elif mode == "first":
                results.append(self.query_once(query_source))
            elif mode == "count":
                results.append(len(self.query_all(query_source)))
            else:
                results.append(self.query_all(query_source))

        if mode == "exists":
            return np.asarray(results, dtype=np.bool_)
        if mode == "count":
            return np.asarray(results, dtype=np.int64)
        return results
