% Negation query fixtures.
:- begin_tests(rolls_back_bindings).

same(X, X).
nope(a).
q(X) :- same(X, a), nope(b).
p(X) :- \+ q(X), same(X, b).

test(
    rolls_back_bindings,
    true(p(X) == p(b))
) :-
    p(X).

:- end_tests(rolls_back_bindings).


:- begin_tests(keeps_outer_choicepoints).

q(a).
q(b).
r(a).
p(X) :- q(X), \+ r(X).

test(
    keeps_outer_choicepoints,
    true(p(X) == p(b))
) :-
    p(X).

:- end_tests(keeps_outer_choicepoints).


:- begin_tests(nested_parity_p).

a.
p :- \+ (\+ a).
missing_guard(a).
missing :- missing_guard(b).
q :- \+ (\+ missing).

test(
    nested_parity_p,
    true(p == p)
) :-
    p.

test(
    nested_parity_q,
    [fail]
) :-
    q.

:- end_tests(nested_parity_p).


:- begin_tests(negation_bound_arg_success).

p(a).
ok(X) :- \+ p(X).

test(
    negation_bound_arg_success,
    true(ok(b) == ok(b))
) :-
    ok(b).

test(
    negation_bound_arg_failure,
    [fail]
) :-
    ok(a).

:- end_tests(negation_bound_arg_success).


:- begin_tests(not_bound_arg_success).

p(a).
ok(X) :- not(p(X)).

test(
    not_bound_arg_success,
    true(ok(b) == ok(b))
) :-
    ok(b).

test(
    not_bound_arg_failure,
    [fail]
) :-
    ok(a).

:- end_tests(not_bound_arg_success).


:- begin_tests(non_negation_regression).

p(a).
q(X) :- p(X).

test(
    non_negation_regression,
    true(q(X) == q(a))
) :-
    q(X).

:- end_tests(non_negation_regression).


:- begin_tests(negation_unbound_variable_goal).

p(a).
q(X) :- \+ p(X).

test(
    negation_unbound_variable_goal,
    [fail]
) :-
    q(_).

:- end_tests(negation_unbound_variable_goal).


:- begin_tests(negation_syntax).

q.
operator_form :- \+ q.
parenthesized_form :- \+(q).
alias_form :- not(q).
nested_form :- \+ (\+ (\+ q)).

test(
    operator_form_compiles,
    [fail]
) :-
    operator_form.

test(
    parenthesized_form_compiles,
    [fail]
) :-
    parenthesized_form.

test(
    not_alias_compiles,
    [fail]
) :-
    alias_form.

test(
    double_nested_negation_compiles,
    [fail]
) :-
    nested_form.

:- end_tests(negation_syntax).


:- begin_tests(naf_alias_ok).

p(a).
ok :- naf(p(b)).

test(
    naf_alias_ok,
    true(ok == ok)
) :-
    ok.

:- end_tests(naf_alias_ok).


:- begin_tests(nfa_alias_ok).

p(a).
ok :- nfa(p(b)).

test(
    nfa_alias_ok,
    true(ok == ok)
) :-
    ok.

:- end_tests(nfa_alias_ok).


:- begin_tests(undefined_predicate_negation_succeeds).

ok :- not(missing).

test(
    undefined_predicate_negation_succeeds,
    true(ok == ok)
) :-
    ok.

:- end_tests(undefined_predicate_negation_succeeds).


:- begin_tests(checks_all_outer_bindings).

value(a).
value(b).
p(a).
p(b).
bad :- value(X), \+ p(X).

test(
    checks_all_outer_bindings,
    [fail]
) :-
    bad.

:- end_tests(checks_all_outer_bindings).


:- begin_tests(nested_compound_argument).

q(((X, Y), X)).
p((X, Y)) :- q((X, Z)), q((Z, Y)).

test(
    nested_compound_argument,
    true(not(p((((a, b), c), b))) == not(p((((a, b), c), b))))
) :-
    not(p((((a, b), c), b))).

:- end_tests(nested_compound_argument).