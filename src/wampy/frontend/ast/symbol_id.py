"""Stable symbol IDs and the AST symbol allocation layout.

The default layout is ``0`` for no symbol, ``1..8`` for defined language
constructs (including the internal argument-list cell), ``9..15`` for future
reserved constructs, ``16..47`` for the variables in one term, and
``48..MAX`` for user symbols.
"""

from enum import IntEnum, unique


@unique
class CoreSymID(IntEnum):
    """Stable IDs for compiler-recognized AST symbols."""

    # Sentinel
    NO_SYMBOL = 0

    # Fundamental execution constructs
    TRUE = 1  # true/0
    ARGS = 2  # internal binary argument-list cell; never a Prolog functor
    CONJUNCTION = 3  # ,/2

    # Fundamental AST roots
    CLAUSE = 4  # clause root, normally :-/2
    QUERY = 5  # query root, normally ?-/1

    # Core control flow
    # DISJUNCTION = 6  # ;/2
    NEGATION_AS_FAILURE = 7  # \\+/1
    CUT = 8  # !/0
    # IF_THEN = 9  # ->/2

    # Dynamic goal execution
    # META_CALL = 10  # call/1 or variable in goal position

    # Source-file and language extensions
    # DIRECTIVE = 11  # :-/1
    # MODULE_QUALIFICATION = 12  # :/2
    # DCG_RULE = 13  # -->/2
    # SOFT_IF_THEN = 14  # *->/2


# Keep the reserved extension range stable even when a future SymID is not
# enabled in this build. Variable allocation begins after IDs 0..15.
_FIRST_VARIABLE_SYMBOL_ID = 16


def calculate_symbol_id_positions(
    max_variables_per_term: int,
) -> tuple[int, int]:
    """Calculate the starting IDs for variable and user symbols."""
    first_variable_symbol_id = _FIRST_VARIABLE_SYMBOL_ID
    first_user_symbol_id = first_variable_symbol_id + max_variables_per_term
    return first_variable_symbol_id, first_user_symbol_id
