% Optional SWI-Prolog runner for these fixtures.
:- use_module(library(plunit)).

:- prolog_load_context(directory, Directory),
   directory_file_path(Directory, 'choicepoints.plt', Choicepoints),
   directory_file_path(Directory, 'clauses.plt', Clauses),
   directory_file_path(Directory, 'negation.plt', Negation),
   directory_file_path(Directory, 'recursion.plt', Recursion),
   directory_file_path(Directory, 'runtime_errors.plt', RuntimeErrors),
   directory_file_path(Directory, 'structures.plt', Structures),
   load_files(
       [Choicepoints, Clauses, Negation, Recursion, RuntimeErrors, Structures],
       [silent(true)]
   ).

:- initialization(
    (run_tests -> halt(0); halt(1)),
    main
).
