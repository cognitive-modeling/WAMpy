% Choicepoint query fixtures.
:- begin_tests(two_facts).
p(a).
p(b).

test(
    two_facts,
    all(p(X) == [p(a), p(b)])
) :- p(X).
:- end_tests(two_facts).


:- begin_tests(three_facts).
p(a).
p(b).
p(c).

test(
    three_facts,
    all(p(X) == [p(a), p(b), p(c)])
) :- p(X).
:- end_tests(three_facts).


:- begin_tests(across_goals_single).
q(a).
q(b).
r(a).
p(X) :- q(X), r(X).

test(
    across_goals_single,
    all(p(X) == [p(a)])
) :- p(X).
test(
    restores_bindings,
    all(p(X) == [p(a)])
) :- p(X).
:- end_tests(across_goals_single).


:- begin_tests(across_goals_multi).
q(a).
q(b).
r(a).
r(b).
p(X) :- q(X), r(X).

test(
    across_goals_multi,
    all(p(X) == [p(a), p(b)])
) :- p(X).
:- end_tests(across_goals_multi).


:- begin_tests(unwinding_between_answers).
q(a).
q(b).
p(X) :- q(X).

test(
    unwinding_between_answers,
    all(p(X) == [p(a), p(b)])
) :- p(X).
:- end_tests(unwinding_between_answers).


:- begin_tests(fact_then_rule_order).
p(a).
p(X) :- q(X).
q(b).
q(c).

test(
    fact_then_rule_order,
    all(p(X) == [p(a), p(b), p(c)])
) :- p(X).
:- end_tests(fact_then_rule_order).


:- begin_tests(rule_then_fact_order).
p(X) :- q(X).
p(a).
q(b).
q(c).
test(
    rule_then_fact_order,
    all(p(X) == [p(b), p(c), p(a)])
) :- p(X).
:- end_tests(rule_then_fact_order).


:- begin_tests(multi_rule_same_functor).
p(X) :- q(X).
p(X) :- r(X).
q(a).
r(b).

test(
    multi_rule_same_functor,
    all(p(X) == [p(a), p(b)])
) :-p(X).

:- end_tests(multi_rule_same_functor).
