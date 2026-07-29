from enum import IntEnum, unique


@unique
class OP(IntEnum):
    GET_VAR = 1
    GET_VAL = 2
    GET_CONST = 3

    PUT_VAR = 4
    PUT_VAL = 5
    PUT_CONST = 6

    CALL = 7
    EXECUTE = 8
    PROCEED = 9

    ALLOCATE = 10
    DEALLOCATE = 11

    TRY = 12
    RETRY = 13
    TRUST = 14

    # structure matching / building
    GET_STR = 15  # a=functor_symbol_id, b=arity, c=Xi
    PUT_STR = 16  # a=functor_symbol_id, b=arity, c=Xi

    # unification inside a structure
    UNIFY_VAR = 17  # a=Xj
    UNIFY_VAL = 18  # a=Xj
    UNIFY_CONST = 19  # a=const_symbol_id

    END_STR = 20  # no args; pop back to enclosing structure (nesting)

    MARK_CUT = 21  # save current choicepoint top as the cut boundary
    CUT = 22  # discard choicepoints newer than the saved cut boundary
    DROP_CUT = 23  # pop a saved cut boundary
    FAIL = 24  # force normal backtracking

    JMP_RETRY = 25  # a = clause pc to run now, b = next alternative prefix pc
    JMP_TRUST = 26  # a = clause pc to run now
