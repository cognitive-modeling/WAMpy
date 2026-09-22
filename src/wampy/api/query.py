"""Stateful, resumable WAM query iterators."""

from collections.abc import Iterator
from enum import Enum, unique
from typing import TYPE_CHECKING

from wampy.compiler.compiled_query import CompiledQuery
from wampy.frontend.solution import SolutionType, decode_bindings
from wampy.runtime.interpreter import redo, run
from wampy.runtime.machine import init_machine, reset_machine
from wampy.status import WAMStatus

if TYPE_CHECKING:
    from .prolog import Prolog


class QueryError(RuntimeError):
    """A WAM execution error raised while advancing a ``Query``."""

    def __init__(self, status: WAMStatus):
        self.status = WAMStatus(int(status))
        super().__init__(f"WAM error: {self.status.name}")


@unique
class QueryPhase(Enum):
    NEW = 0
    ACTIVE = 1
    EXHAUSTED = 2
    CLOSED = 3
    FAILED = 4


class Query(Iterator[SolutionType]):
    """One private, resumable WAM execution."""

    def __init__(self, prolog: "Prolog", compiled_query: CompiledQuery):
        self._prolog: Prolog = prolog
        self._compiled_query = compiled_query
        self._phase = QueryPhase.NEW
        self._status: WAMStatus | None = None
        self._closed = False
        self._machine = None

    def __iter__(self):
        return self

    def __next__(self) -> SolutionType:
        if self._phase in (QueryPhase.CLOSED, QueryPhase.EXHAUSTED, QueryPhase.FAILED):
            raise StopIteration

        try:
            if self._phase is QueryPhase.NEW:
                status = self._start()
            else:
                assert self._machine is not None
                status = WAMStatus(
                    int(
                        redo(
                            self._machine,
                            self._prolog.compiled_program,
                            self._compiled_query,
                        )
                    )
                )
        except Exception:
            self._phase = QueryPhase.FAILED
            self._release_state()
            raise

        self._status = status
        if status == WAMStatus.SUCCESS:
            self._phase = QueryPhase.ACTIVE
            return self._snapshot_bindings()

        if status == WAMStatus.EXHAUSTED:
            self._phase = QueryPhase.EXHAUSTED
            self._release_state()
            raise StopIteration

        self._phase = QueryPhase.FAILED
        self._release_state()
        raise QueryError(status)

    def next(self) -> SolutionType | None:
        """Return the next solution, or ``None`` when the query is exhausted.

        Unlike Python's :func:`next`, this convenience method does not raise
        :class:`StopIteration` when no further solutions are available.
        """

        try:
            return self.__next__()
        except StopIteration:
            return None

    def close(self) -> None:
        """Release this query's machine state; safe to call repeatedly."""

        if self._closed:
            return
        if self._phase not in (QueryPhase.EXHAUSTED, QueryPhase.FAILED):
            self._phase = QueryPhase.CLOSED
        self._release_state()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def phase(self) -> QueryPhase:
        return self._phase

    @property
    def status(self) -> WAMStatus | None:
        return self._status

    @property
    def state(self) -> QueryPhase:
        """Alias for callers that prefer ``query.state`` to ``phase``."""

        return self._phase

    def __enter__(self) -> "Query":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _start(self) -> WAMStatus:
        self._machine = init_machine(self._prolog.config.runtime)

        state = self._machine.registers
        state.trace_enabled[0] = int(self._prolog.config.runtime.trace)

        status: WAMStatus = run(
            self._machine,
            self._prolog.compiled_program,
            self._compiled_query,
        )
        return status

    def _snapshot_bindings(self) -> SolutionType:
        if self._machine is None:
            return {}

        return decode_bindings(
            self._machine,
            self._compiled_query,
            self._prolog.symbol_table,
        )

    def _release_state(self) -> None:
        if self._machine is not None:
            try:
                reset_machine(self._machine)
            except Exception:
                # Cleanup must not mask the original query error.
                pass
        self._machine = None
        self._closed = True

    def __del__(self):
        try:
            self._release_state()
        except Exception:
            pass
