from pathlib import Path

import pytest

from .plunit_loader import PlUnitCase, PlUnitFormatError, load_plunit_cases, parse_plunit

ANCESTOR_FIXTURE = """\
:- begin_tests(recursion).
parent(john, mary).
parent(mary, anne).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

test(ground, true(ancestor(john, anne) == ancestor(john, anne))) :- ancestor(john, anne).
test(enumeration, all(ancestor(X, anne) == [ancestor(mary, anne), \
ancestor(john, anne)])) :- ancestor(X, anne).
test(failure, [fail]) :- ancestor(anne, X).
:- end_tests(recursion).
"""


def test_parse_plunit_removes_test_clauses_and_preserves_ordered_answers():
    cases = parse_plunit(ANCESTOR_FIXTURE, filename="recursion.plt")

    assert cases == [
        PlUnitCase(
            id="recursion/recursion::ground",
            program=(
                "parent(john, mary).\n"
                "parent(mary, anne).\n"
                "ancestor(X, Y) :- parent(X, Y).\n"
                "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)."
            ),
            query="ancestor(john, anne).",
            mode="true",
            template="ancestor(john, anne)",
            expected=("ancestor(john, anne).",),
        ),
        PlUnitCase(
            id="recursion/recursion::enumeration",
            program=(
                "parent(john, mary).\n"
                "parent(mary, anne).\n"
                "ancestor(X, Y) :- parent(X, Y).\n"
                "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)."
            ),
            query="ancestor(X, anne).",
            mode="all",
            template="ancestor(X, anne)",
            expected=("ancestor(mary, anne).", "ancestor(john, anne)."),
        ),
        PlUnitCase(
            id="recursion/recursion::failure",
            program=(
                "parent(john, mary).\n"
                "parent(mary, anne).\n"
                "ancestor(X, Y) :- parent(X, Y).\n"
                "ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y)."
            ),
            query="ancestor(anne, X).",
            mode="fail",
        ),
    ]


def test_parse_plunit_supports_deterministic_and_list_wrapped_forms():
    source = """\
:- begin_tests(forms).
p(a).
test(det) :- p(a).
test(fail, [fail]) :- p(b).
test(true_list, [true(p(X) == p(a))]) :- p(X).
test(all_list, [all(p(X) == [p(a)])]) :- p(X).
:- end_tests(forms).
"""

    cases = parse_plunit(source, filename="forms.plt")

    assert [(case.id, case.mode, case.expected) for case in cases] == [
        ("forms/forms::det", "det", ()),
        ("forms/forms::fail", "fail", ()),
        ("forms/forms::true_list", "true", ("p(a).",)),
        ("forms/forms::all_list", "all", ("p(a).",)),
    ]


def test_parse_plunit_loads_supported_expected_runtime_errors():
    source = """\
:- begin_tests(errors).
test(undefined, [throws(error(existence_error(procedure, missing/0), _))]) :- missing.
:- end_tests(errors).
"""

    [case] = parse_plunit(source, filename="runtime_errors.plt")

    assert case.mode == "throws"
    assert case.query == "missing."
    assert case.expected_error == "WAM error: UNDEFINED_PREDICATE"


def test_parse_plunit_derives_a_wildcard_procedure_error_from_the_query():
    source = """\
:- begin_tests(errors).
test(arity, [throws(error(existence_error(procedure, _), _))]) :- p(a, b).
:- end_tests(errors).
"""

    [case] = parse_plunit(source, filename="runtime_errors.plt")

    assert case.expected_error == "WAM error: INVALID_PC"


def test_load_plunit_cases_discovers_sorted_plt_files_with_relative_ids(tmp_path: Path):
    fixture_dir = tmp_path / "prolog"
    (fixture_dir / "nested").mkdir(parents=True)
    (fixture_dir / "z.plt").write_text(
        ":- begin_tests(z).\np.\ntest(ok) :- p.\n:- end_tests(z).\n",
        encoding="utf-8",
    )
    (fixture_dir / "nested" / "a.plt").write_text(
        ":- begin_tests(a).\nq.\ntest(ok) :- q.\n:- end_tests(a).\n",
        encoding="utf-8",
    )
    (fixture_dir / "ignored.pl").write_text("not native PlUnit", encoding="utf-8")

    cases = load_plunit_cases(fixture_dir)

    assert [case.id for case in cases] == ["nested/a/a::ok", "z/z::ok"]


def test_pytest_collects_plunit_fixtures_below_test_query(pytester):
    package_root = Path(__file__).resolve().parents[2]

    pytester.makeconftest(f"""
        import sys
        from pathlib import Path

        import pytest

        sys.path.insert(0, {str(package_root)!r})

        from tests.plunit.plunit_collector import PlUnitTestModule


        CASES_ROOT = Path(__file__).parent / "cases"


        def pytest_pycollect_makemodule(
            module_path: Path,
            parent: pytest.Collector,
        ):
            if module_path.name != "test_query.py":
                return None

            return PlUnitTestModule.from_parent(
                parent,
                path=module_path,
                cases_root=CASES_ROOT,
            )
    """)

    # The logical parent shown in Test Explorer.
    pytester.makepyfile(test_query="")

    cases_root = pytester.path / "cases"
    cases_root.mkdir()

    (cases_root / "backtracking.plt").write_text(
        """\
:- begin_tests(two_facts).
two_facts.
test(two_facts) :- two_facts.
:- end_tests(two_facts).

:- begin_tests(across_goals_single).
p(a).
test(across_goals_single) :- p(a).
test(restores_bindings, [fail]) :- p(b).
:- end_tests(across_goals_single).
""",
        encoding="utf-8",
    )

    result = pytester.runpytest(
        "--collect-only",
        "-q",
        "test_query.py",
    )

    assert result.ret == 0

    collected = [
        line.strip()
        for line in result.stdout.str().splitlines()
        if "test_query.py::backtracking.plt::" in line
    ]

    assert collected == [
        "test_query.py::backtracking.plt::two_facts::two_facts",
        ("test_query.py::backtracking.plt::across_goals_single::across_goals_single"),
        ("test_query.py::backtracking.plt::across_goals_single::restores_bindings"),
    ]

    output = result.stdout.str()

    assert "cases/backtracking.plt::" not in output
    assert "test_query.py::backtracking::" not in output
    assert "3 tests collected" in output


@pytest.mark.parametrize(
    ("source", "line", "message"),
    [
        (
            ":- begin_tests(broken).\ntest(missing) :- p.\n",
            1,
            "unterminated begin_tests",
        ),
        (
            ":- begin_tests(broken).\ntest(bad, [nondet]) :- p.\n:- end_tests(broken).\n",
            2,
            "unsupported PlUnit option",
        ),
        (
            ":- begin_tests(broken).\ntest(bad, true(X = a)) :- p(X).\n:- end_tests(broken).\n",
            2,
            "only == comparisons",
        ),
        (
            ":- begin_tests(broken).\ntest(bad) :- p, q.\n:- end_tests(broken).\n",
            2,
            "one WAMpy query goal",
        ),
    ],
)
def test_malformed_plunit_reports_filename_line_and_reason(source, line, message):
    with pytest.raises(PlUnitFormatError) as raised:
        parse_plunit(source, filename="broken.plt")

    error = raised.value
    assert error.filename == "broken.plt"
    assert error.line == line
    assert f"broken.plt:{line}:" in str(error)
    assert message in str(error)


def test_unsupported_directives_and_unit_options_are_rejected():
    sources = [
        ":- begin_tests(unit, [condition(true)]).\ntest(ok) :- p.\n:- end_tests(unit).\n",
        ":- module(example, []).\n",
        ":- begin_tests(unit).\n:- dynamic p/1.\ntest(ok) :- p(a).\n:- end_tests(unit).\n",
    ]

    for source in sources:
        with pytest.raises(PlUnitFormatError, match="unsupported directive|unit options"):
            parse_plunit(source, filename="unsupported.plt")


def test_duplicate_units_and_test_names_are_rejected():
    duplicate_units = """\
:- begin_tests(unit).
test(one) :- p.
:- end_tests(unit).
:- begin_tests(unit).
test(two) :- p.
:- end_tests(unit).
"""
    duplicate_tests = """\
:- begin_tests(unit).
test(one) :- p.
test(one) :- p.
:- end_tests(unit).
"""

    with pytest.raises(PlUnitFormatError, match="duplicate test unit"):
        parse_plunit(duplicate_units, filename="duplicates.plt")
    with pytest.raises(PlUnitFormatError, match="duplicate test name"):
        parse_plunit(duplicate_tests, filename="duplicates.plt")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            ":- begin_tests(unit).\n:- end_tests(other).\n",
            "does not close",
        ),
        (
            "test(outside) :- p.\n",
            "must be inside begin_tests/1",
        ),
        (
            ":- begin_tests(outer).\n"
            ":- begin_tests(inner).\n"
            "test(ok) :- p.\n"
            ":- end_tests(inner).\n"
            ":- end_tests(outer).\n",
            "nested begin_tests",
        ),
        (
            ":- begin_tests(unit).\ntest(malformed, true(p == p)).\n:- end_tests(unit).\n",
            "one :- body",
        ),
    ],
)
def test_plunit_structure_and_test_clause_errors_are_source_located(source, message):
    with pytest.raises(PlUnitFormatError, match=message):
        parse_plunit(source, filename="structure.plt")


def test_comments_are_ignored_without_becoming_program_clauses():
    source = """\
% A fixture comment.
:- begin_tests(comments).
/* A block comment with a period. */
p.
test(ok) :- p. % test comment
:- end_tests(comments).
"""

    [case] = parse_plunit(source, filename="comments.plt")

    assert case.program == "p."
