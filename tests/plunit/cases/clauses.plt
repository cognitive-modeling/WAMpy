% Clause query fixtures.
:- begin_tests(simple_fact).
p.
test(simple_fact) :- p.
:- end_tests(simple_fact).


:- begin_tests(fact_with_constant_argument).
p(a).
test(
    fact_with_constant_argument,
    true(p(X) == p(a))
) :- p(X).
test(
    fact_wrong_constant,
    [fail]
) :- p(b).
test(
    single_clause_ground_query,
    true(p(a) == p(a))
) :- p(a).
:- end_tests(fact_with_constant_argument).


:- begin_tests(same_var_success).
p(X, X).
test(
    same_var_success,
    true(p(a, a) == p(a, a))
) :- p(a, a).
test(
    same_var_failure,
    [fail]
) :- p(a, b).
:- end_tests(same_var_success).


:- begin_tests(distinct_vars_ab).
p(_X, _Y).
test(
    distinct_vars_ab,
    true(p(a, b) == p(a, b))
) :- p(a, b).
test(
    distinct_vars_aa,
    true(p(a, a) == p(a, a))
) :- p(a, a).
:- end_tests(distinct_vars_ab).


:- begin_tests(single_goal_rule).
q.
p :- q.
test(
    single_goal_rule,
    true(p == p)
) :-
    p.
test(
    deterministic_rule,
    true(p == p)
) :-
    p.
:- end_tests(single_goal_rule).


:- begin_tests(multiple_goals_rule).
q.
r.
p :- q, r.
test(
    multiple_goals_rule,
    true(p == p)
) :-
    p.
:- end_tests(multiple_goals_rule).


:- begin_tests(goal_failure_backtracks).

r_false(a).
r :- r_false(b).
q.
p :- q, r.

test(
    goal_failure_backtracks,
    [fail]
) :-
    p.

:- end_tests(goal_failure_backtracks).


:- begin_tests(undefined_predicate_in_rule_body_fails).

q_false(a).
q :- q_false(b).
p :- q.

test(
    undefined_predicate_in_rule_body_fails,
    [fail]
) :-
    p.

:- end_tests(undefined_predicate_in_rule_body_fails).


:- begin_tests(nested_calls).

r.
q :- r.
p :- q.

test(
    nested_calls,
    true(p == p)
) :-
    p.

:- end_tests(nested_calls).


:- begin_tests(nested_calls_two_levels).

r.
q :- r.
p :- q.
s :- p.

test(
    nested_calls_two_levels,
    true(s == s)
) :-
    s.

:- end_tests(nested_calls_two_levels).


:- begin_tests(two_args_passing).
p(a, b).
q(X, Y) :- p(X, Y).
test(
    two_args_passing,
    true(q(X, Y) == q(a, b))
) :-
    q(X, Y).
:- end_tests(two_args_passing).


:- begin_tests(aliasing_success).
q(a, a).
p(X) :- q(X, X).
test(
    aliasing_success,
    true(p(X) == p(a))
) :-
    p(X).
:- end_tests(aliasing_success).


:- begin_tests(aliasing_failure).
q(a, b).
p(X) :- q(X, X).
test(
    aliasing_failure,
    [fail]
) :-
    p(_).
:- end_tests(aliasing_failure).


:- begin_tests(three_args_passing).
t(a, b, c).
p(X, Y, Z) :- t(X, Y, Z).

test(
    three_args_passing,
    true(p(X, Y, Z) == p(a, b, c))
) :-
    p(X, Y, Z).

:- end_tests(three_args_passing).

:- begin_tests(likes_subject_lookup).
likes(mary, food).
likes(john, mary).
likes(john, wine).
test(
    likes_subject_lookup,
    all(likes(john, X) == [likes(john, mary), likes(john, wine)])
) :-
    likes(john, X).
test(
    likes_reverse_lookup,
    true(likes(X, food) == likes(mary, food))
) :-
    likes(X, food).
test(
    likes_missing_fact,
    [fail]
) :-
    likes(john, beer).
:- end_tests(likes_subject_lookup).


:- begin_tests(body_local_var_single_goal).
q(a).
p :- q(_).
test(
    body_local_var_single_goal,
    true(p == p)
) :-
    p.
:- end_tests(body_local_var_single_goal).


:- begin_tests(body_local_var_join_success).
q(a).
q(b).
r(b).
p :- q(X), r(X).
test(
    body_local_var_join_success,
    true(p == p)
) :-
    p.
:- end_tests(body_local_var_join_success).


:- begin_tests(body_local_var_join_failure).
q(a).
q(b).
r(c).
p :- q(X), r(X).
test(
    body_local_var_join_failure,
    [fail]
) :-
    p.
:- end_tests(body_local_var_join_failure).


:- begin_tests(fixture_smoke).
fact.
item(first).
item(second).
item(third).

test(
    fact_success,
    true(fact == fact)
) :-
    fact.

test(
    fact_failure,
    [fail]
) :-
    item(missing).

test(
    ordered_answers,
    all(item(X) == [item(first), item(second), item(third)])
) :-
    item(X).

:- end_tests(fixture_smoke).


:- begin_tests(several_queries).
color(red).
color(blue).
shape(circle).
test(
    colors,
    all(color(X) == [color(red), color(blue)])
) :-
    color(X).
test(
    shape,
    true(shape(circle) == shape(circle))
) :-
    shape(circle).
:- end_tests(several_queries).


:- begin_tests(quoted_atom_zero).
p('zero').
test(
    quoted_atom_zero,
    true(p(X) == p(zero))
) :-
    p(X).
:- end_tests(quoted_atom_zero).


:- begin_tests(unquoted_atom_zero).
p(zero).
test(
    unquoted_atom_zero,
    true(p(X) == p(zero))
) :-
    p(X).
:- end_tests(unquoted_atom_zero).


:- begin_tests(quoted_atom_with_space).
p('has space').
test(
    quoted_atom_with_space,
    true(p('has space') == p('has space'))
) :-
    p('has space').
:- end_tests(quoted_atom_with_space).


:- begin_tests(true_builtin_in_rule).
p :- true.
test(
    true_builtin_in_rule,
    true(p == p)
) :-
    p.
:- end_tests(true_builtin_in_rule).
