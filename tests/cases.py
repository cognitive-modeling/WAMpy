from dataclasses import dataclass
from typing import List


# ----------------------------
# Core WAM runtime case types
# ----------------------------


@dataclass(frozen=True)
class Case:
    id: str
    program: str
    query: str
    expected: List[str]
    n: int


@dataclass(frozen=True)
class ErrorCase:
    id: str
    program: str
    query: str
    msg_substr: str


@dataclass(frozen=True)
class InvalidQueryCase:
    id: str
    program: str
    query: str


# ----------------------------
# Parser / compile-time cases
# ----------------------------


@dataclass(frozen=True)
class CompileCase:
    id: str
    program: str


@dataclass(frozen=True)
class CompileErrorCase:
    id: str
    program: str
    exc_type: type[Exception]


# ----------------------------
# Prefix helpers
# ----------------------------


def prefix_cases(prefix: str, cases: List[Case]) -> List[Case]:
    prefix = prefix.strip().strip("/") + "/"
    return [Case(prefix + c.id.lstrip("/"), c.program, c.query, c.expected, c.n) for c in cases]


def prefix_error_cases(prefix: str, cases: List[ErrorCase]) -> List[ErrorCase]:
    prefix = prefix.strip().strip("/") + "/"
    return [ErrorCase(prefix + c.id.lstrip("/"), c.program, c.query, c.msg_substr) for c in cases]


def prefix_invalid_query_cases(prefix: str, cases: List[InvalidQueryCase]) -> List[InvalidQueryCase]:
    prefix = prefix.strip().strip("/") + "/"
    return [InvalidQueryCase(prefix + c.id.lstrip("/"), c.program, c.query) for c in cases]


def prefix_compile_cases(prefix: str, cases: List[CompileCase]) -> List[CompileCase]:
    prefix = prefix.strip().strip("/") + "/"
    return [CompileCase(prefix + c.id.lstrip("/"), c.program) for c in cases]


def prefix_compile_error_cases(prefix: str, cases: List[CompileErrorCase]) -> List[CompileErrorCase]:
    prefix = prefix.strip().strip("/") + "/"
    return [CompileErrorCase(prefix + c.id.lstrip("/"), c.program, c.exc_type) for c in cases]


# ------------------------------------------------------------
# Main semantic cases
# ------------------------------------------------------------

CASES_FACTS: List[Case] = [
    Case("simple_fact", "p.", "p.", ["p."], 1),
    Case("fact_with_constant_argument", "p(a).", "p(X).", ["p(a)."], 1),
    Case("fact_wrong_constant", "p(a).", "p(b).", [], 0),
]

CASES_UNIFY: List[Case] = [
    Case("same_var_success", "p(X, X).", "p(a, a).", ["p(a, a)."], 1),
    Case("same_var_failure", "p(X, X).", "p(a, b).", [], 0),
    Case("distinct_vars_ab", "p(X, Y).", "p(a, b).", ["p(a, b)."], 1),
    Case("distinct_vars_aa", "p(X, Y).", "p(a, a).", ["p(a, a)."], 1),
]

CASES_RULES: List[Case] = [
    Case("single_goal_rule", "q. p :- q.", "p.", ["p."], 1),
    Case("multiple_goals_rule", "q. r. p :- q, r.", "p.", ["p."], 1),
    Case("goal_failure_backtracks", "q. p :- q, r.", "p.", [], 0),
    # Current engine behavior: undefined call in body fails the clause
    # instead of raising a runtime exception.
    Case("undefined_predicate_in_rule_body_fails", "p :- q.", "p.", [], 0),
]

CASES_BACKTRACKING: List[Case] = [
    Case("two_facts", "p(a). p(b).", "p(X).", ["p(a).", "p(b)."], 2),
    Case("three_facts", "p(a). p(b). p(c).", "p(X).", ["p(a).", "p(b).", "p(c)."], 3),
    Case(
        "across_goals_single",
        "q(a). q(b). r(a). p(X) :- q(X), r(X).",
        "p(X).",
        ["p(a)."],
        1,
    ),
    Case(
        "across_goals_multi",
        "q(a). q(b). r(a). r(b). p(X) :- q(X), r(X).",
        "p(X).",
        ["p(a).", "p(b)."],
        2,
    ),
]

CASES_CALLS: List[Case] = [
    Case("nested_calls", "r. q :- r. p :- q.", "p.", ["p."], 1),
    Case("nested_calls_two_levels", "r. q :- r. p :- q. s :- p.", "s.", ["s."], 1),
]

CASES_TRAIL: List[Case] = [
    Case("unwinding_between_answers", "q(a). q(b). p(X) :- q(X).", "p(X).", ["p(a).", "p(b)."], 2),
    Case("restores_bindings", "q(a). q(b). r(a). p(X) :- q(X), r(X).", "p(X).", ["p(a)."], 1),
]

CASES_ARGPASS: List[Case] = [
    Case("two_args_passing", "p(a, b). q(X, Y) :- p(X, Y).", "q(X, Y).", ["q(a, b)."], 1),
    Case("aliasing_success", "q(a, a). p(X) :- q(X, X).", "p(X).", ["p(a)."], 1),
    Case("aliasing_failure", "q(a, b). p(X) :- q(X, X).", "p(X).", [], 0),
    Case("three_args_passing", "t(a, b, c). p(X, Y, Z) :- t(X, Y, Z).", "p(X, Y, Z).", ["p(a, b, c)."], 1),
]

CASES_DETERMINISTIC: List[Case] = [
    Case("single_clause_ground_query", "p(a).", "p(a).", ["p(a)."], 1),
    Case("deterministic_rule", "q. p :- q.", "p.", ["p."], 1),
]

CASES_EDGE: List[Case] = [
    Case("quoted_atom_zero", "p('zero').", "p(X).", ["p(zero)."], 1),
    Case("unquoted_atom_zero", "p(zero).", "p(X).", ["p(zero)."], 1),
    Case("quoted_atom_with_space", "p('has space').", "p('has space').", ["p('has space')."], 1),
    Case("anonymous_var_single", "p(a).", "p(_).", ["p(a)."], 1),
    Case("anonymous_var_multiple", "p(a). p(b).", "p(_).", ["p(a).", "p(b)."], 2),
    Case("true_builtin_in_rule", "p :- true.", "p.", ["p."], 1),
]


# ------------------------------------------------------------
# SWI-style pure Prolog cases (facts, clauses, unification)
# ------------------------------------------------------------

CASES_SWI_FACTS: List[Case] = [
    Case(
        "likes_subject_lookup",
        "likes(mary, food). likes(john, mary). likes(john, wine).",
        "likes(john, X).",
        ["likes(john, mary).", "likes(john, wine)."],
        2,
    ),
    Case(
        "likes_reverse_lookup",
        "likes(mary, food). likes(john, mary). likes(john, wine).",
        "likes(X, food).",
        ["likes(mary, food)."],
        1,
    ),
    Case(
        "likes_missing_fact",
        "likes(mary, food). likes(john, mary). likes(john, wine).",
        "likes(john, beer).",
        [],
        0,
    ),
]

CASES_RECURSION: List[Case] = [
    Case(
        "ancestor_ground_success",
        """
            parent(john, mary).
            parent(mary, anne).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """,
        "ancestor(john, anne).",
        ["ancestor(john, anne)."],
        1,
    ),
    Case(
        "ancestor_enumeration",
        """
            parent(john, mary).
            parent(mary, anne).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """,
        "ancestor(X, anne).",
        ["ancestor(mary, anne).", "ancestor(john, anne)."],
        2,
    ),
    Case(
        "ancestor_failure",
        """
            parent(john, mary).
            parent(mary, anne).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """,
        "ancestor(anne, X).",
        [],
        0,
    ),
]

CASES_RECURSIVE_SHARED_VARS: List[Case] = [
    Case(
        "exact_answer_from_a",
        """
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
        """,
        "p(a, Y).",
        ["p(a, d)."],
        1,
    ),
    Case(
        "exact_answer_from_b",
        """
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
        """,
        "p(b, Y).",
        ["p(b, d)."],
        1,
    ),
    Case(
        "exact_answer_from_c",
        """
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
        """,
        "p(c, Y).",
        ["p(c, d)."],
        1,
    ),
    Case(
        "reject_wrong_output",
        """
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
        """,
        "p(a, c).",
        [],
        0,
    ),
    Case(
        "reject_non_reachable_output",
        """
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
        """,
        "p(a, foo).",
        [],
        0,
    ),
    Case(
        "negation_accepts_wrong_output_failure",
        r"""
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
            ok :- \+ p(a, c).
        """,
        "ok.",
        ["ok."],
        1,
    ),
    Case(
        "negation_rejects_true_output",
        r"""
            q(a, b).
            q(b, c).
            q(c, d).
            done(d).
            p(X, X) :- done(X).
            p(X, Y) :- q(X, Z), p(Z, Y).
            bad :- \+ p(a, d).
        """,
        "bad.",
        [],
        0,
    ),
]

CASES_CLAUSE_ORDER: List[Case] = [
    Case(
        "fact_then_rule_order",
        "p(a). p(X) :- q(X). q(b). q(c).",
        "p(X).",
        ["p(a).", "p(b).", "p(c)."],
        3,
    ),
    Case(
        "rule_then_fact_order",
        "p(X) :- q(X). p(a). q(b). q(c).",
        "p(X).",
        ["p(b).", "p(c).", "p(a)."],
        3,
    ),
]

CASES_STRUCTURAL_UNIFICATION: List[Case] = [
    Case("structure_shared_var_success", "p(f(X), X).", "p(f(a), a).", ["p(f(a), a)."], 1),
    Case("structure_shared_var_failure", "p(f(X), X).", "p(f(a), b).", [], 0),
    Case("cross_structure_success", "p(f(X, Y), f(Y, X)).", "p(f(a, b), f(b, a)).", ["p(f(a, b), f(b, a))."], 1),
    Case("cross_structure_failure", "p(f(X, Y), f(Y, X)).", "p(f(a, b), f(a, b)).", [], 0),
    Case("nested_structure_success", "p(X, g(X)).", "p(a, g(a)).", ["p(a, g(a))."], 1),
    Case("nested_structure_failure", "p(X, g(X)).", "p(a, g(b)).", [], 0),
]

CASES_EXISTENTIAL: List[Case] = [
    Case("body_local_var_single_goal", "q(a). p :- q(X).", "p.", ["p."], 1),
    Case("body_local_var_join_success", "q(a). q(b). r(b). p :- q(X), r(X).", "p.", ["p."], 1),
    Case("body_local_var_join_failure", "q(a). q(b). r(c). p :- q(X), r(X).", "p.", [], 0),
]

CASES_WAM_CLASSIC: List[Case] = [
    Case(
        "peano_plus_success",
        """
            plus(zero, Y, Y).
            plus(s(X), Y, s(Z)) :- plus(X, Y, Z).
        """,
        "plus(s(zero), s(zero), X).",
        ["plus(s(zero), s(zero), s(s(zero)))."],
        1,
    ),
    Case(
        "peano_plus_failure",
        """
            plus(zero, Y, Y).
            plus(s(X), Y, s(Z)) :- plus(X, Y, Z).
        """,
        "plus(s(zero), s(zero), s(zero)).",
        [],
        0,
    ),
]


# ------------------------------------------------------------
# Additional edge cases (missing coverage from TODO/review)
# ------------------------------------------------------------

CASES_MISSING_EDGE: List[Case] = [
    # anonymous variables must be independent placeholders
    Case("anonymous_vars_are_distinct", "p(a, b).", "p(_, _).", ["p(a, b)."], 1),
    Case("anonymous_var_and_named_var", "p(a, b).", "p(X, _).", ["p(a, b)."], 1),
    # structural mismatch failure modes
    Case("structure_functor_mismatch", "p(f(a)).", "p(g(a)).", [], 0),
    Case("structure_arity_mismatch_inner", "p(f(a)).", "p(f(a, b)).", [], 0),
    Case("structure_arity_mismatch_inner_reverse", "p(f(a, b)).", "p(f(a)).", [], 0),
    Case("structure_with_anonymous_success", "p(f(a, b)).", "p(f(_, b)).", ["p(f(a, b))."], 1),
    Case("structure_with_anonymous_failure", "p(f(a, b)).", "p(f(_, c)).", [], 0),
    # same predicate defined by multiple rules
    Case("multi_rule_same_functor", "p(X) :- q(X). p(X) :- r(X). q(a). r(b).", "p(X).", ["p(a).", "p(b)."], 2),
    # negation with unbound variable in goal: should fail if inner goal has any solution
    Case("negation_unbound_variable_goal", r"p(a). q(X) :- \+ p(X).", "q(X).", [], 0),
]

# ------------------------------------------------------------
# Negation: parser + runtime cases (from unittest suite)
# ------------------------------------------------------------

NEGATION_COMPILE_CASES: List[CompileCase] = [
    CompileCase("operator_form_compiles", r"q. p :- \+ q."),
    CompileCase("parenthesized_form_compiles", r"q. p :- \+(q)."),
    CompileCase("not_alias_compiles", "q. p :- not(q)."),
    CompileCase("double_nested_negation_compiles", r"q. p :- \+ (\+ (\+ q))."),
]

NEGATION_COMPILE_ERROR_CASES: List[CompileErrorCase] = [
    CompileErrorCase("empty_negation_operand_raises", r"p :- \+.", ValueError),
    CompileErrorCase("grouped_negation_raises", r"p :- \+(a,b).", ValueError),
    CompileErrorCase("variable_negation_operand_raises", r"p(X) :- \+ X.", ValueError),
]

NEGATION_RUNTIME_CASES: List[Case] = [
    # core negation as failure
    Case("core_negation_ok", r"p(a). ok :- \+ p(b).", "ok.", ["ok."], 1),
    Case("core_negation_bad", r"p(a). bad :- \+ p(a).", "bad.", [], 0),
    # not/1 alias runtime
    Case("not_alias_ok", "p(a). ok :- not(p(b)).", "ok.", ["ok."], 1),
    Case("not_alias_bad", "p(a). bad :- not(p(a)).", "bad.", [], 0),
    # negation rolls back bindings
    Case(
        "rolls_back_bindings",
        r"""
            same(X, X).
            q(X) :- same(X, a), nope.
            p(X) :- \+ q(X), same(X, b).
        """,
        "p(X).",
        ["p(b)."],
        1,
    ),
    # negation keeps outer choicepoints
    Case(
        "keeps_outer_choicepoints",
        r"""
            q(a).
            q(b).
            r(a).
            p(X) :- q(X), \+ r(X).
        """,
        "p(X).",
        ["p(b)."],
        1,
    ),
    # nested negation parity
    Case(
        "nested_parity_p",
        r"""
            a.
            p :- \+ (\+ a).
            q :- \+ (\+ missing).
        """,
        "p.",
        ["p."],
        1,
    ),
    Case(
        "nested_parity_q",
        r"""
            a.
            p :- \+ (\+ a).
            q :- \+ (\+ missing).
        """,
        "q.",
        [],
        0,
    ),
    Case("negation_bound_arg_success", r"p(a). ok(X) :- \+ p(X).", "ok(b).", ["ok(b)."], 1),
    Case("negation_bound_arg_failure", r"p(a). ok(X) :- \+ p(X).", "ok(a).", [], 0),
    Case("not_bound_arg_success", "p(a). ok(X) :- not(p(X)).", "ok(b).", ["ok(b)."], 1),
    Case("not_bound_arg_failure", "p(a). ok(X) :- not(p(X)).", "ok(a).", [], 0),
    # non-negation regression
    Case("non_negation_regression", "p(a). q(X) :- p(X).", "q(X).", ["q(a)."], 1),
]


# ------------------------------------------------------------
# Aggregate exported lists
# ------------------------------------------------------------

CASES: List[Case] = [
    *prefix_cases("facts", CASES_FACTS),
    *prefix_cases("unify", CASES_UNIFY),
    *prefix_cases("rules", CASES_RULES),
    *prefix_cases("backtracking", CASES_BACKTRACKING),
    *prefix_cases("calls", CASES_CALLS),
    *prefix_cases("trail", CASES_TRAIL),
    *prefix_cases("args", CASES_ARGPASS),
    *prefix_cases("deterministic", CASES_DETERMINISTIC),
    *prefix_cases("edge", CASES_EDGE),
    *prefix_cases("swi/facts", CASES_SWI_FACTS),
    *prefix_cases("swi/recursion", CASES_RECURSION),
    *prefix_cases("wam/recursive_shared_vars", CASES_RECURSIVE_SHARED_VARS),
    *prefix_cases("swi/clause_order", CASES_CLAUSE_ORDER),
    *prefix_cases("swi/structural_unification", CASES_STRUCTURAL_UNIFICATION),
    *prefix_cases("swi/existential", CASES_EXISTENTIAL),
    *prefix_cases("wam/classic", CASES_WAM_CLASSIC),
    *prefix_cases("edge/missing", CASES_MISSING_EDGE),
    *prefix_cases("negation/runtime", NEGATION_RUNTIME_CASES),
]

NEGATION_COMPILE: List[CompileCase] = [
    *prefix_compile_cases("negation/parser", NEGATION_COMPILE_CASES),
]

NEGATION_COMPILE_ERRORS: List[CompileErrorCase] = [
    *prefix_compile_error_cases("negation/parser", NEGATION_COMPILE_ERROR_CASES),
]


# ------------------------------------------------------------
# Runtime error cases
# ------------------------------------------------------------

RUNTIME_ERROR_CASES: List[ErrorCase] = [
    ErrorCase("undefined_predicate", "p.", "q.", "procedure `q/0` does not exist"),
    # Current engine behavior for arity mismatch raises INVALID_PC before
    # surfacing an undefined procedure diagnostic for p/2.
    ErrorCase("arity_mismatch_as_undefined_procedure", "p(a).", "p(a, b).", "WAM error: INVALID_PC"),
]

RUNTIME_ERRORS: List[ErrorCase] = [
    *prefix_error_cases("runtime_errors", RUNTIME_ERROR_CASES),
]


# ------------------------------------------------------------
# Invalid query cases
# ------------------------------------------------------------

INVALID_QUERY_CASES: List[InvalidQueryCase] = [
    InvalidQueryCase("empty_query", "p.", ""),
    InvalidQueryCase("whitespace_query", "p.", "   "),
    InvalidQueryCase("malformed_query_missing_paren", "p(a).", "p(a"),
]

INVALID_QUERIES: List[InvalidQueryCase] = [
    *prefix_invalid_query_cases("invalid_query", INVALID_QUERY_CASES),
]
