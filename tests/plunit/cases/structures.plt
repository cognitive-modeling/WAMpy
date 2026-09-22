% Structure query fixtures.
:- begin_tests(structure_shared_var_success).

p(f(X), X).

test(
    structure_shared_var_success,
    true(p(f(a), a) == p(f(a), a))
) :-
    p(f(a), a).

test(
    structure_shared_var_failure,
    [fail]
) :-
    p(f(a), b).

:- end_tests(structure_shared_var_success).


:- begin_tests(cross_structure_success).

p(f(X, Y), f(Y, X)).

test(
    cross_structure_success,
    true(p(f(a, b), f(b, a)) == p(f(a, b), f(b, a)))
) :-
    p(f(a, b), f(b, a)).

test(
    cross_structure_failure,
    [fail]
) :-
    p(f(a, b), f(a, b)).

:- end_tests(cross_structure_success).


:- begin_tests(nested_structure_success).

p(X, g(X)).

test(
    nested_structure_success,
    true(p(a, g(a)) == p(a, g(a)))
) :-
    p(a, g(a)).

test(
    nested_structure_failure,
    [fail]
) :-
    p(a, g(b)).

:- end_tests(nested_structure_success).


:- begin_tests(anonymous_var_single).

p(a).

test(
    anonymous_var_single,
    true(p(X) == p(a))
) :-
    p(X).

:- end_tests(anonymous_var_single).


:- begin_tests(anonymous_var_multiple).

p(a).
p(b).

test(
    anonymous_var_multiple,
    all(p(X) == [p(a), p(b)])
) :-
    p(X).

:- end_tests(anonymous_var_multiple).


:- begin_tests(anonymous_vars_are_distinct).

p(a, b).

test(
    anonymous_vars_are_distinct,
    true(p(X, Y) == p(a, b))
) :-
    p(X, Y).

test(
    anonymous_var_and_named_var,
    true(p(X, Y) == p(a, b))
) :-
    p(X, Y).

:- end_tests(anonymous_vars_are_distinct).


:- begin_tests(structure_functor_mismatch).

p(f(a)).

test(
    structure_functor_mismatch,
    [fail]
) :-
    p(g(a)).

test(
    structure_arity_mismatch_inner,
    [fail]
) :-
    p(f(a, b)).

:- end_tests(structure_functor_mismatch).


:- begin_tests(structure_arity_mismatch_inner_reverse).

p(f(a, b)).

test(
    structure_arity_mismatch_inner_reverse,
    [fail]
) :-
    p(f(a)).

test(
    structure_with_anonymous_success,
    true(p(f(X, b)) == p(f(a, b)))
) :-
    p(f(X, b)).

test(
    structure_with_anonymous_failure,
    [fail]
) :-
    p(f(_, c)).

:- end_tests(structure_arity_mismatch_inner_reverse).


:- begin_tests(permanent_nested_variable_across_calls).

first_part(((V2, V1), V2)).

p11((V1, V2)) :-
    first_part((V3, V2)),
    first_part((V1, V3)).

p12((V1, V2)) :-
    p11((V3, V1)),
    p11((V1, V3)).


% Sanity check for the relation used by the regression below.
test(
    nested_relation_success
) :-
    p11((((circle, triangle), square), circle)).


% Regression: V3 is a permanent variable embedded in the compound
% argument of the first call. It must remain constrained when the
% second call is executed.
test(
    permanent_nested_variable_preserved,
    [fail]
) :-
    p12((((circle, triangle), square), triangle)).

:- end_tests(permanent_nested_variable_across_calls).