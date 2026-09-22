"""Load the restricted PlUnit subset used by WAMpy query tests.

Semantic fixtures are native SWI-Prolog PlUnit files (.plt) located in
tests/plunit/cases/.

Each begin_tests/end_tests block defines one independent WAMpy program.
Ordinary Prolog clauses become the program; test(...) clauses become pytest
cases.

Supported forms:
    test(Name) :- Goal.
    test(Name, [fail]) :- Goal.
    test(Name, true(Template == Expected)) :- Goal.
    test(Name, all(Template == [Expected, ...])) :- Goal.
    test(Name, [throws(error(existence_error(procedure, Name/Arity), _))]) :- Goal.

true/1 and all/1 may also appear in a one-element option list.

Unsupported PlUnit features are rejected with a filename and source line.
Malformed source, invalid queries, compiler tests, and other Python-specific
tests remain ordinary pytest tests.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_PlUnitMode = Literal["det", "fail", "true", "all", "throws"]


@dataclass(frozen=True)
class PlUnitCase:
    """One executable case extracted from a native PlUnit test clause."""

    id: str
    program: str
    query: str
    mode: _PlUnitMode
    template: str | None = None
    expected: tuple[str, ...] = ()
    expected_error: str | None = None


class PlUnitFormatError(ValueError):
    """Raised when a fixture uses syntax outside the supported PlUnit subset."""

    def __init__(self, filename: str, line: int, message: str) -> None:
        self.filename = filename
        self.line = line
        super().__init__(f"{filename}:{line}: {message}")


@dataclass(frozen=True)
class _Clause:
    text: str
    line: int


@dataclass
class _Unit:
    name: str
    line: int
    source: list[_Clause] = field(default_factory=list)
    tests: list[tuple[int, str, str, _PlUnitMode, str | None, tuple[str, ...], str | None]] = field(
        default_factory=list
    )


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROCEDURE_ERROR_RE = re.compile(
    r"^error\s*\(\s*existence_error\s*\(\s*procedure\s*,\s*"
    r"(?P<procedure>[A-Za-z_][A-Za-z0-9_]*\s*/\s*[0-9]+)\s*\)\s*,\s*_\s*\)$"
)
_ANY_PROCEDURE_ERROR_RE = re.compile(
    r"^error\s*\(\s*existence_error\s*\(\s*procedure\s*,\s*_\s*\)\s*,\s*_\s*\)$"
)


def parse_plunit(
    source: str,
    filename: str | Path = "<fixture>",
    *,
    fixture_id: str | None = None,
) -> list[PlUnitCase]:
    """Parse supported native PlUnit clauses into WAMpy fixture cases.

    Supported test forms are deterministic ``test(Name)`` tests, ``[fail]``,
    ``true(Template == Expected)``, ``all(Template == [Expected, ...])``, and
    ``[throws(error(existence_error(procedure, Name/Arity), _))]``.  A list
    wrapper around ``true`` and ``all`` is accepted as well.
    """

    filename = str(filename)
    clauses = _split_clauses(source, filename)
    prefix = _normalise_fixture_id(filename) if fixture_id is None else fixture_id
    cases: list[PlUnitCase] = []
    seen_case_ids: set[str] = set()
    seen_units: set[str] = set()
    current: _Unit | None = None

    for clause in clauses:
        begin_args = _directive_args(clause, "begin_tests", filename)
        if begin_args is not None:
            if current is not None:
                raise PlUnitFormatError(
                    filename, clause.line, "nested begin_tests units are not supported"
                )
            if len(begin_args) != 1:
                raise PlUnitFormatError(
                    filename,
                    clause.line,
                    "begin_tests/2 and unit options are not supported; provide one unit name",
                )
            unit_name = _parse_name(begin_args[0], filename, clause.line, "test unit name")
            if unit_name in seen_units:
                raise PlUnitFormatError(filename, clause.line, f"duplicate test unit {unit_name!r}")
            seen_units.add(unit_name)
            current = _Unit(unit_name, clause.line)
            continue

        end_args = _directive_args(clause, "end_tests", filename)
        if end_args is not None:
            if current is None:
                raise PlUnitFormatError(
                    filename, clause.line, "end_tests appears outside a test unit"
                )
            if len(end_args) != 1:
                raise PlUnitFormatError(
                    filename,
                    clause.line,
                    "end_tests/2 and unit options are not supported; provide one unit name",
                )
            end_name = _parse_name(end_args[0], filename, clause.line, "test unit name")
            if end_name != current.name:
                raise PlUnitFormatError(
                    filename,
                    clause.line,
                    f"end_tests({end_name}) does not close begin_tests({current.name})",
                )
            cases.extend(_finish_unit(current, prefix, filename, seen_case_ids))
            current = None
            continue

        if clause.text.lstrip().startswith(":-"):
            raise PlUnitFormatError(
                filename,
                clause.line,
                "unsupported directive; only begin_tests/1 and end_tests/1 are allowed",
            )

        if current is None:
            raise PlUnitFormatError(
                filename,
                clause.line,
                "ordinary clauses and test clauses must be inside begin_tests/1",
            )

        if re.match(r"^test\s*\(", clause.text.lstrip()):
            parsed = _parse_test_clause(clause, filename)
            if any(existing[1] == parsed[1] for existing in current.tests):
                raise PlUnitFormatError(
                    filename,
                    clause.line,
                    f"duplicate test name {parsed[1]!r} in begin_tests({current.name})",
                )
            current.tests.append(parsed)
        else:
            current.source.append(clause)

    if current is not None:
        raise PlUnitFormatError(
            filename,
            current.line,
            f"unterminated begin_tests({current.name}); expected end_tests({current.name})",
        )
    if not cases:
        raise PlUnitFormatError(filename, 1, "fixture does not contain a supported PlUnit test")
    return cases


def _finish_unit(
    unit: _Unit,
    prefix: str,
    filename: str,
    seen_case_ids: set[str],
) -> list[PlUnitCase]:
    program = "\n".join(f"{clause.text.strip()}." for clause in unit.source).strip()
    cases: list[PlUnitCase] = []
    for line, name, query, mode, template, expected, expected_error in unit.tests:
        case_id = f"{prefix}/{unit.name}::{name}"
        if case_id in seen_case_ids:
            raise PlUnitFormatError(filename, line, f"duplicate fixture case ID {case_id!r}")
        seen_case_ids.add(case_id)
        cases.append(
            PlUnitCase(
                id=case_id,
                program=program,
                query=query,
                mode=mode,
                template=template,
                expected=expected,
                expected_error=expected_error,
            )
        )
    return cases


def _parse_test_clause(
    clause: _Clause,
    filename: str,
) -> tuple[int, str, str, _PlUnitMode, str | None, tuple[str, ...], str | None]:
    separator = _find_top_level(clause.text, ":-", filename, clause.line)
    if len(separator) != 1:
        raise PlUnitFormatError(filename, clause.line, "test clause must have one :- body")
    head = clause.text[: separator[0]].strip()
    body = clause.text[separator[0] + 2 :].strip()
    args = _call_args(head, "test", filename, clause.line)
    if args is None:
        raise PlUnitFormatError(
            filename, clause.line, "malformed test clause; expected test(Name) :- Goal"
        )
    if len(args) not in {1, 2}:
        raise PlUnitFormatError(
            filename, clause.line, "test/3 and test clauses with more options are not supported"
        )

    name = _parse_name(args[0], filename, clause.line, "test name")
    query = _parse_query(body, filename, clause.line)
    if len(args) == 1:
        return clause.line, name, query, "det", None, (), None

    mode, template, expected, expected_error = _parse_option(args[1], body, filename, clause.line)
    return clause.line, name, query, mode, template, expected, expected_error


def _parse_option(
    option_text: str,
    body: str,
    filename: str,
    line: int,
) -> tuple[_PlUnitMode, str | None, tuple[str, ...], str | None]:
    option_text = option_text.strip()
    if option_text.startswith("["):
        inner = _bracket_inner(option_text, filename, line)
        options = _split_top_level(inner, ",", filename, line) if inner.strip() else []
        if len(options) != 1:
            raise PlUnitFormatError(
                filename, line, "multiple independent PlUnit options are not supported"
            )
        option_text = options[0].strip()

    if option_text == "fail":
        return "fail", None, (), None

    for mode_name in ("true", "all"):
        inner = _call_inner(option_text, mode_name, filename, line)
        if inner is not None:
            template, rhs = _parse_comparison(inner, body, filename, line)
            if mode_name == "true":
                return "true", template, (_ensure_period(rhs),), None
            expected = _parse_expected_list(rhs, filename, line)
            return "all", template, tuple(_ensure_period(value) for value in expected), None

    throws_inner = _call_inner(option_text, "throws", filename, line)
    if throws_inner is not None:
        procedure = _parse_expected_error(throws_inner, body, filename, line)
        expected_error = (
            "WAM error: INVALID_PC" if procedure == "p/2" else "WAM error: UNDEFINED_PREDICATE"
        )
        return "throws", None, (), expected_error

    option_name = option_text.split("(", 1)[0].strip() or option_text
    raise PlUnitFormatError(filename, line, f"unsupported PlUnit option {option_name!r}")


def _parse_comparison(
    expression: str,
    body: str,
    filename: str,
    line: int,
) -> tuple[str, str]:
    positions = _find_top_level(expression, "==", filename, line)
    if len(positions) != 1:
        if _find_top_level(expression, "=", filename, line):
            raise PlUnitFormatError(
                filename, line, "only == comparisons are supported in PlUnit expectations"
            )
        raise PlUnitFormatError(
            filename, line, "PlUnit expectation must compare Template == Expected"
        )
    position = positions[0]
    template = expression[:position].strip()
    expected = expression[position + 2 :].strip()
    if not template or not expected:
        raise PlUnitFormatError(
            filename, line, "PlUnit expectation must provide both a template and an expected term"
        )
    if _canonical_term(template) != _canonical_term(body):
        raise PlUnitFormatError(
            filename, line, "comparison template must match the complete test goal"
        )
    return template, expected


def _parse_expected_list(value: str, filename: str, line: int) -> list[str]:
    inner = _bracket_inner(value, filename, line)
    if not inner.strip():
        return []
    values = _split_top_level(inner, ",", filename, line)
    if any(not item.strip() for item in values):
        raise PlUnitFormatError(filename, line, "all(...) contains an empty expected term")
    return [item.strip() for item in values]


def _parse_expected_error(value: str, body: str, filename: str, line: int) -> str:
    match = _PROCEDURE_ERROR_RE.fullmatch(value.strip())
    if match is not None:
        return re.sub(r"\s+", "", match.group("procedure"))
    if _ANY_PROCEDURE_ERROR_RE.fullmatch(value.strip()):
        return _query_procedure(body, filename, line)
    raise PlUnitFormatError(
        filename,
        line,
        "only existence_error(procedure, Name/Arity) throws expectations are supported",
    )


def _query_procedure(body: str, filename: str, line: int) -> str:
    body = body.strip()
    match = re.fullmatch(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\((?P<args>.*)\))?", body, re.S)
    if match is None:
        raise PlUnitFormatError(
            filename, line, "throws expectations require a simple procedure query"
        )
    args = match.group("args")
    if args is None or not args.strip():
        arity = 0
    else:
        arity = len(_split_top_level(args, ",", filename, line))
    return f"{match.group('name')}/{arity}"


def _parse_query(body: str, filename: str, line: int) -> str:
    body = body.strip()
    if not body:
        raise PlUnitFormatError(filename, line, "test body is empty")
    if _find_top_level(body, ",", filename, line) or _find_top_level(body, ";", filename, line):
        raise PlUnitFormatError(filename, line, "test bodies must be one WAMpy query goal")
    return _ensure_period(body)


def _call_args(text: str, name: str, filename: str, line: int) -> list[str] | None:
    inner = _call_inner(text.strip(), name, filename, line)
    if inner is None:
        return None
    return _split_top_level(inner, ",", filename, line) if inner.strip() else []


def _directive_args(clause: _Clause, name: str, filename: str) -> list[str] | None:
    text = clause.text.lstrip()
    if not text.startswith(":-"):
        return None
    rest = text[2:].strip()
    if not (rest == name or rest.startswith(f"{name}(") or rest.startswith(f"{name} ")):
        return None
    args = _call_args(rest, name, filename, clause.line)
    if args is None:
        raise PlUnitFormatError(filename, clause.line, f"malformed {name}/1 directive")
    return args


def _call_inner(text: str, name: str, filename: str, line: int) -> str | None:
    prefix = f"{name}("
    if not text.startswith(prefix):
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text[len(name) :], start=len(name)):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise PlUnitFormatError(filename, line, f"malformed {name}(...) expression")
            if depth == 0:
                if text[index + 1 :].strip():
                    raise PlUnitFormatError(filename, line, f"malformed {name}(...) expression")
                return text[len(name) + 1 : index]
    raise PlUnitFormatError(filename, line, f"malformed {name}(...) expression")


def _bracket_inner(text: str, filename: str, line: int) -> str:
    text = text.strip()
    if not text.startswith("["):
        raise PlUnitFormatError(
            filename, line, "expected a list-wrapped PlUnit option or expected answer list"
        )
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                if text[index + 1 :].strip():
                    raise PlUnitFormatError(filename, line, "malformed PlUnit option list")
                return text[1:index]
            if depth < 0:
                break
    raise PlUnitFormatError(filename, line, "malformed PlUnit option list")


def _parse_name(value: str, filename: str, line: int, description: str) -> str:
    value = value.strip()
    if _NAME_RE.fullmatch(value) is None:
        raise PlUnitFormatError(filename, line, f"{description} must be a simple atom")
    return value


def _find_top_level(text: str, token: str, filename: str, line: int) -> list[int]:
    positions: list[int] = []
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {
        ")": "(",
        "]": "[",
        "}": "{",
    }
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in closing:
            opening = closing[char]
            depths[opening] -= 1
            if depths[opening] < 0:
                raise PlUnitFormatError(filename, line, "unbalanced parentheses")
        elif not any(depths.values()) and text.startswith(token, index):
            positions.append(index)
            index += len(token)
            continue
        index += 1
    if quote is not None or any(depths.values()):
        raise PlUnitFormatError(filename, line, "unbalanced quoted text or parentheses")
    return positions


def _split_top_level(text: str, separator: str, filename: str, line: int) -> list[str]:
    positions = _find_top_level(text, separator, filename, line)
    if not positions:
        return [text]
    values: list[str] = []
    start = 0
    for position in positions:
        values.append(text[start:position])
        start = position + len(separator)
    values.append(text[start:])
    return values


def _canonical_term(text: str) -> str:
    result: list[str] = []
    quote: str | None = None
    escaped = False
    for char in text.strip():
        if quote is not None:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            result.append(char)
        elif not char.isspace():
            result.append(char)
    return "".join(result)


def _ensure_period(value: str) -> str:
    value = value.strip()
    return value if value.endswith(".") else f"{value}."


def _mask_comments(source: str, filename: str) -> str:
    result = list(source)
    index = 0
    quote: str | None = None
    escaped = False
    block_start = 0
    while index < len(source):
        char = source[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if source.startswith("/*", index):
            block_start = index
            result[index] = " "
            result[index + 1] = " "
            index += 2
            while index < len(source) and not source.startswith("*/", index):
                if source[index] != "\n":
                    result[index] = " "
                index += 1
            if index >= len(source):
                line = source.count("\n", 0, block_start) + 1
                raise PlUnitFormatError(filename, line, "unterminated block comment")
            result[index] = " "
            result[index + 1] = " "
            index += 2
            continue
        if char == "%":
            while index < len(source) and source[index] != "\n":
                result[index] = " "
                index += 1
            continue
        index += 1
    return "".join(result)


def _split_clauses(source: str, filename: str) -> list[_Clause]:
    source = _mask_comments(source, filename)
    clauses: list[_Clause] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {
        ")": "(",
        "]": "[",
        "}": "{",
    }
    quote: str | None = None
    escaped = False

    def emit(end: int) -> None:
        raw = source[start:end]
        text = raw.strip()
        if text:
            leading = len(raw) - len(raw.lstrip())
            line = source.count("\n", 0, start + leading) + 1
            clauses.append(_Clause(text, line))

    for index, char in enumerate(source):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in depths:
            depths[char] += 1
        elif char in closing:
            opening = closing[char]
            depths[opening] -= 1
            if depths[opening] < 0:
                line = source.count("\n", 0, index) + 1
                raise PlUnitFormatError(filename, line, "unbalanced parentheses")
        elif char == "." and not any(depths.values()):
            emit(index)
            start = index + 1
    if quote is not None:
        line = source.count("\n", 0, len(source)) + 1
        raise PlUnitFormatError(filename, line, "unterminated quoted text")
    if any(depths.values()):
        line = source.count("\n", 0, start) + 1
        raise PlUnitFormatError(filename, line, "unbalanced parentheses")
    if source[start:].strip():
        leading = len(source[start:]) - len(source[start:].lstrip())
        line = source.count("\n", 0, start + leading) + 1
        raise PlUnitFormatError(filename, line, "clause must end with a period")
    return clauses


def _normalise_fixture_id(filename: str | Path) -> str:
    path = Path(filename)
    if path.suffix == ".plt":
        path = path.with_suffix("")
    value = path.as_posix()
    if value in {"", "."} or value.startswith("<"):
        return "fixture"
    return value.lstrip("./")


def load_plunit_cases(path: str | Path) -> list[PlUnitCase]:
    """Discover and load one ``.plt`` file or all ``.plt`` files below a directory."""

    root = Path(path)
    if root.is_file():
        if root.suffix != ".plt":
            raise ValueError(f"PlUnit fixtures must use the .plt extension: {root}")
        fixture_root = root.parent
        fixture_files = [root]
    elif root.is_dir():
        fixture_root = root
        fixture_files = sorted(file for file in root.rglob("*.plt") if file.is_file())
    else:
        raise FileNotFoundError(path)

    cases: list[PlUnitCase] = []
    seen_ids: set[str] = set()
    for fixture_file in fixture_files:
        relative = fixture_file.relative_to(fixture_root)
        fixture_id = relative.with_suffix("").as_posix()
        parsed = parse_plunit(
            fixture_file.read_text(encoding="utf-8"),
            filename=fixture_file.as_posix(),
            fixture_id=fixture_id,
        )
        for case in parsed:
            if case.id in seen_ids:
                raise PlUnitFormatError(
                    fixture_file.as_posix(),
                    1,
                    f"duplicate fixture case ID {case.id!r}",
                )
            seen_ids.add(case.id)
            cases.append(case)
    return cases


__all__ = ["PlUnitCase", "PlUnitFormatError", "load_plunit_cases", "parse_plunit"]
