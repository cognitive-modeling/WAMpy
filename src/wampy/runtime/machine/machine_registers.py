"""Mutable registers and pointers for one WAM execution."""

from enum import IntEnum, unique
from typing import Final, NamedTuple

import numpy as np

from wampy.config import DEFAULT_CONFIG, RuntimeConfig

NO_CHOICE_POINT: Final = -1
HALT_CONTINUATION: Final = -2
NO_ENVIRONMENT: Final = -1


@unique
class UnificationMode(IntEnum):
    READ = 0
    WRITE = 1


class Registers(NamedTuple):
    """WAM registers followed by execution bookkeeping registers."""

    P: np.ndarray
    CP: np.ndarray
    E: np.ndarray
    B: np.ndarray
    B0: np.ndarray
    H: np.ndarray
    HB: np.ndarray
    TR: np.ndarray
    S: np.ndarray
    mode: np.ndarray
    local_top: np.ndarray
    pdl_top: np.ndarray
    structure_top: np.ndarray
    fuel: np.ndarray
    fail_base: np.ndarray
    trace_enabled: np.ndarray
    path_terms: np.ndarray
    path_depth: np.ndarray
    step_limit: int
    unify_step_limit: int


def init_machine_registers(
    config: RuntimeConfig = DEFAULT_CONFIG.runtime,
) -> Registers:
    """Initialize registers for one WAM machine."""

    return Registers(
        P=np.zeros(1, dtype=np.int32),
        CP=np.full(1, HALT_CONTINUATION, dtype=np.int32),
        E=np.full(1, -1, dtype=np.int32),
        B=np.full(1, NO_CHOICE_POINT, dtype=np.int32),
        B0=np.full(1, NO_CHOICE_POINT, dtype=np.int32),
        H=np.zeros(1, dtype=np.uint16),
        HB=np.zeros(1, dtype=np.uint16),
        TR=np.zeros(1, dtype=np.uint16),
        S=np.zeros(1, dtype=np.uint16),
        mode=np.zeros(1, dtype=np.int32),
        local_top=np.zeros(1, dtype=np.int32),
        pdl_top=np.zeros(1, dtype=np.uint16),
        structure_top=np.zeros(1, dtype=np.int32),
        fuel=np.full(1, config.max_steps, dtype=np.int64),
        fail_base=np.full(1, NO_CHOICE_POINT, dtype=np.int32),
        trace_enabled=np.zeros(1, dtype=np.int32),
        path_terms=np.empty(1, dtype=np.int32),
        path_depth=np.zeros(1, dtype=np.int32),
        step_limit=config.max_steps,
        unify_step_limit=config.max_unify_steps,
    )


def reset_machine_registers(state: Registers) -> None:
    """Reset registers and pointers while retaining allocated memory."""

    state.P[0] = 0
    state.CP[0] = HALT_CONTINUATION
    state.E[0] = -1
    state.B[0] = NO_CHOICE_POINT
    state.B0[0] = NO_CHOICE_POINT
    state.H[0] = 0
    state.HB[0] = 0
    state.TR[0] = 0
    state.S[0] = 0
    state.mode[0] = 0
    state.local_top[0] = 0
    state.pdl_top[0] = 0
    state.structure_top[0] = 0
    state.fuel[0] = state.step_limit
    state.fail_base[0] = NO_CHOICE_POINT
    state.path_depth[0] = 0
