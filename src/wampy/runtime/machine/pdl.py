"""Push-down list storage for WAM unification."""

from typing import NamedTuple

import numpy as np


class PDL(NamedTuple):
    """Work storage for term and nested-structure unification."""

    cells: np.ndarray
    structure_S: np.ndarray
    structure_modes: np.ndarray


def init_pdl(config) -> PDL:
    """Allocate the unification push-down list."""

    return PDL(
        cells=np.empty(config.unify_stack_size * 2, dtype=np.uint16),
        structure_S=np.zeros(config.max_structure_depth, dtype=np.uint16),
        structure_modes=np.zeros(config.max_structure_depth, dtype=np.int32),
    )
