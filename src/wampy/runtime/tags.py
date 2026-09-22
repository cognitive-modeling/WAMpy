"""Tags used by WAM heap cells."""

from enum import IntEnum, unique


@unique
class TAG(IntEnum):
    REF = 0
    CON = 1
    STR = 2
    FUN = 3
