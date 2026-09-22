
% Runtime error query fixtures.
:- begin_tests(undefined_predicate).

p.

test(
    undefined_predicate,
    [throws(error(existence_error(procedure, _), _))]
) :-
    q.

:- end_tests(undefined_predicate).


:- begin_tests(arity_mismatch_as_undefined_procedure).

p(a).

test(
    arity_mismatch_as_undefined_procedure,
    [throws(error(existence_error(procedure, _), _))]
) :-
    p(a, b).

:- end_tests(arity_mismatch_as_undefined_procedure).


:- begin_tests(fixture_runtime_error).

known.

test(
    undefined_predicate,
    [throws(error(existence_error(procedure, _), _))]
) :-
    missing.

:- end_tests(fixture_runtime_error).
