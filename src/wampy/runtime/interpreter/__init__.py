"""WAM instruction interpreter, organized by instruction family.


run()
    P  = query entry
    CP = top-level sentinel
        ↓
    emulate()


redo()
    fail()
        ↓
    restored choice point
        ↓
    emulate()


emulate()
    does not initialize anything
    does not intentionally backtrack anything
    just executes from the current P

"""

from wampy.runtime.interpreter.choice import (
    exec_JMP_RETRY,
    exec_JMP_TRUST,
    exec_RETRY_ME_ELSE,
    exec_TRUST_ME_ELSE_FAIL,
    exec_TRY_ME_ELSE,
    fail,
    restore_choice,
)
from wampy.runtime.interpreter.control import (
    exec_ALLOCATE,
    exec_CALL,
    exec_CUT,
    exec_DEALLOCATE,
    exec_EXECUTE,
    exec_FAIL,
    exec_GET_LEVEL,
    exec_NECK_CUT,
    exec_PROCEED,
)
from wampy.runtime.interpreter.execution import redo, run
from wampy.runtime.interpreter.loop import emulate
from wampy.runtime.interpreter.term import (
    exec_END_STR,
    exec_GET_CONST,
    exec_GET_STR,
    exec_GET_VAL,
    exec_GET_VAR,
    exec_GET_Y_VAL,
    exec_GET_Y_VAR,
    exec_PUT_CONST,
    exec_PUT_STR,
    exec_PUT_VAL,
    exec_PUT_VAR,
    exec_PUT_Y_VAL,
    exec_PUT_Y_VAR,
    exec_SET_CONST,
    exec_SET_VAL,
    exec_SET_VAR,
    exec_SET_Y_VAL,
    exec_SET_Y_VAR,
    exec_UNIFY_CONST,
    exec_UNIFY_VAL,
    exec_UNIFY_VAR,
    exec_UNIFY_Y_VAL,
    exec_UNIFY_Y_VAR,
)
from wampy.runtime.machine.choice_point import (
    copy_A_to_choice,
    copy_choice_to_A,
)
from wampy.runtime.machine.environment import (
    environment_size_at_continuation,
    environment_y_address,
)

__all__ = [
    "copy_A_to_choice",
    "copy_choice_to_A",
    "emulate",
    "environment_size_at_continuation",
    "environment_y_address",
    "exec_ALLOCATE",
    "exec_CALL",
    "exec_CUT",
    "exec_DEALLOCATE",
    "exec_END_STR",
    "exec_EXECUTE",
    "exec_FAIL",
    "exec_GET_CONST",
    "exec_GET_LEVEL",
    "exec_GET_STR",
    "exec_GET_VAL",
    "exec_GET_VAR",
    "exec_GET_Y_VAL",
    "exec_GET_Y_VAR",
    "exec_JMP_RETRY",
    "exec_JMP_TRUST",
    "exec_NECK_CUT",
    "exec_PROCEED",
    "exec_PUT_CONST",
    "exec_PUT_STR",
    "exec_PUT_VAL",
    "exec_PUT_VAR",
    "exec_PUT_Y_VAL",
    "exec_PUT_Y_VAR",
    "exec_RETRY_ME_ELSE",
    "exec_SET_CONST",
    "exec_SET_VAL",
    "exec_SET_VAR",
    "exec_SET_Y_VAL",
    "exec_SET_Y_VAR",
    "exec_TRUST_ME_ELSE_FAIL",
    "exec_TRY_ME_ELSE",
    "exec_UNIFY_CONST",
    "exec_UNIFY_VAL",
    "exec_UNIFY_VAR",
    "exec_UNIFY_Y_VAL",
    "exec_UNIFY_Y_VAR",
    "fail",
    "redo",
    "restore_choice",
    "run",
]
