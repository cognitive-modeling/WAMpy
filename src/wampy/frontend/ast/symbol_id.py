"""
Symbol ID layout:

0        EMPTY
1..31    built-in/internal GIDs
32..63   variables
64..254  user symbols/functors

255      ERR sentinel
"""

from enum import IntEnum, unique


@unique
class SymID(IntEnum):
    """Symbol IDs for the Abstract Syntax tree (AST)."""

    SYMBOL_FIRST = 0
    SYMBOL_LAST = 197
    TRUE = 198
    NOT_PROVABLE = 199  # negation as failure
    # ---
    # max functor minimal size is 200
    # ---
    EMPTY = 200
    S = 201
    CLAUSE = 202
    QUERY = 203
    # ---
    BODY = 204
    AND = 205
    TERM = 206
    SYMBOL = 207
    VAR = 208
    #
    # --- Vars ---
    VAR_FIRST = 210
    VAR_LAST = 240
    ERR = 255

    def __str__(self):
        return self.name
