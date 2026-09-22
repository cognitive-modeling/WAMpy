
% Recursion query fixtures.
:- begin_tests(ancestor_ground_success).

parent(john, mary).
parent(mary, anne).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

test(
    ancestor_ground_success,
    all(ancestor(john, anne) == [ancestor(john, anne)])
) :-
    ancestor(john, anne).

test(
    ancestor_enumeration,
    all(ancestor(X, anne) == [ancestor(mary, anne), ancestor(john, anne)])
) :-
    ancestor(X, anne).

test(
    ancestor_failure,
    [fail]
) :-
    ancestor(anne, _).

:- end_tests(ancestor_ground_success).


:- begin_tests(exact_answer_from_a).

q(a, b).
q(b, c).
q(c, d).
done(d).
p(X, X) :- done(X).
p(X, Y) :- q(X, Z), p(Z, Y).

test(
    exact_answer_from_a,
    all(p(a, Y) == [p(a, d)])
) :-
    p(a, Y).

test(
    exact_answer_from_b,
    all(p(b, Y) == [p(b, d)])
) :-
    p(b, Y).

test(
    exact_answer_from_c,
    all(p(c, Y) == [p(c, d)])
) :-
    p(c, Y).

test(
    reject_wrong_output,
    [fail]
) :-
    p(a, c).

test(
    reject_non_reachable_output,
    [fail]
) :-
    p(a, foo).

:- end_tests(exact_answer_from_a).


:- begin_tests(negation_accepts_wrong_output_failure).

q(a, b).
q(b, c).
q(c, d).
done(d).
p(X, X) :- done(X).
p(X, Y) :- q(X, Z), p(Z, Y).
ok :- \+ p(a, c).

test(
    negation_accepts_wrong_output_failure,
    true(ok == ok)
) :-
    ok.

:- end_tests(negation_accepts_wrong_output_failure).


:- begin_tests(negation_rejects_true_output).

q(a, b).
q(b, c).
q(c, d).
done(d).
p(X, X) :- done(X).
p(X, Y) :- q(X, Z), p(Z, Y).
bad :- \+ p(a, d).

test(
    negation_rejects_true_output,
    [fail]
) :-
    bad.

:- end_tests(negation_rejects_true_output).


:- begin_tests(peano_plus_success).

plus(zero, Y, Y).
plus(s(X), Y, s(Z)) :- plus(X, Y, Z).

test(
    peano_plus_success,
    true(plus(s(zero), s(zero), X) == plus(s(zero), s(zero), s(s(zero))))
) :-
    plus(s(zero), s(zero), X).

test(
    peano_plus_failure,
    [fail]
) :-
    plus(s(zero), s(zero), s(zero)).

:- end_tests(peano_plus_success).
