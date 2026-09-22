"""Pytest collection and execution for native WAMpy query fixtures."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from wampy.config import DEFAULT_CONFIG

from .helpers import solve_from_str
from .plunit_loader import PlUnitCase, parse_plunit


def _case_names(case: PlUnitCase) -> tuple[str, str]:
    """Return the PlUnit unit name and test name encoded in a case ID."""

    unit_id, test_name = case.id.rsplit("::", 1)
    unit_name = unit_id.rsplit("/", 1)[-1]
    return unit_name, test_name


def _run_case(case: PlUnitCase) -> None:
    """Execute and validate a normal PlUnit case."""

    solutions = solve_from_str(
        case.program,
        case.query,
        DEFAULT_CONFIG,
    )

    if case.mode == "det":
        assert len(solutions) == 1

    elif case.mode == "fail":
        assert solutions == []

    elif case.mode in {"true", "all"}:
        assert case.template is not None
        actual = [
            _canonical_term(_instantiate_template(case.template, solution))
            for solution in solutions
        ]
        expected = [_canonical_term(value) for value in case.expected]
        assert actual == expected

    elif case.mode == "throws":
        raise AssertionError("runtime-error cases must use _run_error_case")

    else:
        raise AssertionError(f"unexpected PlUnit mode: {case.mode}")


def _instantiate_template(template: str, solution: dict[str, object]) -> str:
    """Render a fixture template from decoded high-level bindings."""

    variable_pattern = re.compile(r"\b[A-Z_][A-Za-z0-9_]*\b")

    def replace(match: re.Match[str]) -> str:
        name = match.group(0)
        return str(solution.get(name, name))

    return f"{variable_pattern.sub(replace, template)}."


def _canonical_term(value: str) -> str:
    return "".join(value.removesuffix(".").split())


def _run_error_case(case: PlUnitCase) -> None:
    """Execute a PlUnit case that expects a runtime error."""

    assert case.mode == "throws"

    with pytest.raises(RuntimeError) as exc:
        solve_from_str(
            case.program,
            case.query,
            DEFAULT_CONFIG,
        )

    assert case.expected_error is not None
    assert case.expected_error in str(exc.value)


class PlUnitTestModule(pytest.Module):
    """Collect Python tests and logical PlUnit fixtures below test_query.py."""

    def __init__(
        self,
        *,
        cases_root: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cases_root = (
            cases_root if cases_root is not None else self.path.parent.parent / "plunit" / "cases"
        )

    def collect(self) -> Iterator[pytest.Collector | pytest.Item]:
        # Ordinary Python tests remain direct children of test_query.py.
        yield from super().collect()

        if not self.cases_root.is_dir():
            return

        for fixture_path in sorted(self.cases_root.rglob("*.plt")):
            if not fixture_path.is_file():
                continue

            yield PlUnitFixture.from_parent(
                self,
                # Include the extension in the Test Explorer label.
                name=fixture_path.name,
                fixture_path=fixture_path,
            )


class _VSCodeGroup(pytest.Class):
    """Logical group retained as a nested node by VS Code Test Explorer."""

    def _getobj(self):
        # pytest.Class normally resolves an actual Python class from its
        # parent module. These groups are synthetic, so provide one.
        python_name = self.name.replace(".", "_")
        return type(
            python_name,
            (),
            {"__module__": __name__},
        )


class PlUnitFixture(_VSCodeGroup):
    """One logical ``.plt`` fixture below test_query.py."""

    def __init__(
        self,
        *,
        fixture_path: Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fixture_path = fixture_path

    def collect(self) -> Iterator[pytest.Collector]:
        cases = parse_plunit(
            self.fixture_path.read_text(encoding="utf-8"),
            filename=self.fixture_path.as_posix(),
            fixture_id=self.fixture_path.stem,
        )

        units: dict[str, list[PlUnitCase]] = {}

        for case in cases:
            unit_name, _ = _case_names(case)
            units.setdefault(unit_name, []).append(case)

        for unit_name, unit_cases in units.items():
            yield PlUnitUnit.from_parent(
                self,
                name=unit_name,
                cases=tuple(unit_cases),
            )


class PlUnitUnit(_VSCodeGroup):
    """One ``begin_tests(...)`` unit inside a logical fixture."""

    def __init__(
        self,
        *,
        cases: tuple[PlUnitCase, ...],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cases = cases

    def collect(self) -> Iterator[pytest.Item]:
        for case in self.cases:
            _, test_name = _case_names(case)

            yield PlUnitItem.from_parent(
                self,
                name=test_name,
                case=case,
            )


class PlUnitItem(pytest.Item):
    """One executable PlUnit ``test(...)`` clause."""

    def __init__(
        self,
        *,
        case: PlUnitCase,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.case = case

    def runtest(self) -> None:
        if self.case.mode == "throws":
            _run_error_case(self.case)
        else:
            _run_case(self.case)

    def reportinfo(self):
        return (
            self.path,
            0,
            f"PlUnit test {self.case.id}",
        )


__all__ = [
    "PlUnitFixture",
    "PlUnitItem",
    "PlUnitTestModule",
    "PlUnitUnit",
]
