"""Storage layout for WAM environments, choices, cuts, and trace paths."""

from enum import IntEnum, unique
from typing import NamedTuple

import numpy as np


@unique
class ChoiceSlot(IntEnum):
    """Fixed fields stored in each choice-point frame."""

    TR = 0
    BP = 1
    CP = 2
    CE = 3
    PREVIOUS_B = 4
    H = 5
    B0 = 6
    DEPTH = 7
    RETURN_DEPTH = 8
    KIND = 9
    A = 10


@unique
class ChoicePointKind(IntEnum):
    """Internal role/state of a choice point."""

    NORMAL = 0
    NAF = 1
    NAF_TRUNCATED = 2


CHOICE_POINT_FIXED_SIZE = ChoiceSlot.A + 1


class Stack(NamedTuple):
    """Flat WAM stack cells."""

    cells: np.ndarray
    choice_point_frame_size: int
    max_x_registers: int


def is_choice_point_address(stack: Stack, address: int) -> bool:
    """Return whether ``address`` identifies a frame in the choice region."""

    return (
        address >= 0
        and address < stack.cells.shape[0]
        and address + stack.choice_point_frame_size <= stack.cells.shape[0]
    )


def init_stack(config) -> Stack:
    """Allocate one WAM stack address space for environments and choice points."""

    choice_point_size = config.choice_point_size
    choice_point_frame_size = CHOICE_POINT_FIXED_SIZE + config.max_x_registers
    cell_capacity = config.environment_size + choice_point_size * choice_point_frame_size
    return Stack(
        cells=np.empty(
            cell_capacity,
            dtype=np.int32,
        ),
        choice_point_frame_size=choice_point_frame_size,
        max_x_registers=config.max_x_registers,
    )
