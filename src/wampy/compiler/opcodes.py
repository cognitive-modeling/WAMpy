from enum import IntEnum


class OP(IntEnum):
    GET_VAR = 1
    GET_VAL = 2
    GET_CONST = 3

    PUT_VAR = 4
    PUT_VAL = 5
    PUT_CONST = 6

    CALL = 7  # a=predicate_slot, b=arity, c=environment_size
    EXECUTE = 8
    PROCEED = 9

    ALLOCATE = 10
    DEALLOCATE = 11

    TRY_ME_ELSE = 12
    RETRY_ME_ELSE = 13
    TRUST_ME_ELSE_FAIL = 14

    # structure matching / building
    GET_STR = 15  # a=functor_symbol_id, b=arity, c=Xi
    PUT_STR = 16  # a=functor_symbol_id, b=arity, c=Xi

    # unification inside a structure
    UNIFY_VAR = 17  # a=Xj
    UNIFY_VAL = 18  # a=Xj
    UNIFY_CONST = 19  # a=const_symbol_id

    END_STR = 20  # no args; pop back to enclosing structure (nesting)

    FAIL = 21  # force normal backtracking

    JMP_RETRY = 22  # a = clause pc to run now, b = next alternative prefix pc
    JMP_TRUST = 23  # a = clause pc to run now

    # permanent-variable access through the current environment frame
    GET_Y_VAR = 24  # a=Yj, b=Xi
    GET_Y_VAL = 25  # a=Yj, b=Xi
    PUT_Y_VAR = 26  # a=Yj, b=Xi
    PUT_Y_VAL = 27  # a=Yj, b=Xi
    UNIFY_Y_VAR = 28  # a=Yj
    UNIFY_Y_VAL = 29  # a=Yj

    GET_LEVEL = 30  # a=Yj, b=1 saves B0; otherwise saves the current B
    CUT = 31  # a=Yj, discard choicepoints newer than the saved level
    NECK_CUT = 32  # discard choicepoints newer than the current predicate entry

    # construction inside a structure (Ait-Kaci SET instructions)
    SET_VAR = 33  # a=Xj
    SET_VAL = 34  # a=Xj
    SET_CONST = 35  # a=const_symbol_id
    SET_Y_VAR = 36  # a=Yj
    SET_Y_VAL = 37  # a=Yj
