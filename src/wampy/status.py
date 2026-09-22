from enum import IntEnum, unique


@unique
class WAMStatus(IntEnum):
    # No error
    SUCCESS = 0  # a solution exists / current run succeeded
    EXHAUSTED = 1  # normal: backtracking finished (no more solutions)

    # Query / static errors (checked before execution)
    UNDEFINED_PREDICATE = 100  # procedure q/0 does not exist
    ARITY_MISMATCH = 101  # p/1 called, p/2 exists
    INVALID_QUERY = 102  # malformed query
    INVALID_FUNCTOR = 103  # symbol not a predicate

    # Runtime / execution errors
    STACK_OVERFLOW = 200
    TRAIL_OVERFLOW = 201
    HEAP_OVERFLOW = 202
    ENVIRONMENT_OVERFLOW = 203
    STEP_LIMIT = 204
    UNIFY_STEP_LIMIT = 205
    DEPTH_LIMIT = 206

    # Internal / consistency errors (should never happen)
    INVALID_OPCODE = 300
    INVALID_PC = 301
    CORRUPT_CHOICEPOINT = 302
    CORRUPT_ENVIRONMENT = 303
