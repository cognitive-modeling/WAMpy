"""Flattened machine ownership and allocation.

Machine
├── registers: Registers
│   ├── P
│   ├── CP
│   ├── S
│   ├── HB
│   ├── H
│   ├── B0
│   ├── B
│   ├── E
│   └── TR
│
├── X
├── heap
├── stack
│   ├── choice point frames
│   └── environment frames
├── trail
└── pdl


             shared / immutable
        ┌─────────────────────────┐
        │ compiled_program        │
        │ compiled_query          │
        └────────────┬────────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
      Machine 1  Machine 2  Machine 3
      mutable    mutable    mutable
"""

from typing import NamedTuple

import numpy as np

from wampy.config import DEFAULT_CONFIG, RuntimeConfig
from wampy.runtime.machine.heap import HeapMemory, init_heap
from wampy.runtime.machine.machine_registers import (
    Registers,
    init_machine_registers,
    reset_machine_registers,
)
from wampy.runtime.machine.pdl import PDL, init_pdl
from wampy.runtime.machine.stack import Stack, init_stack
from wampy.runtime.machine.trail import init_trail


class Machine(NamedTuple):
    """Own WAM registers, argument registers, and memory regions."""

    registers: Registers
    X: np.ndarray
    heap: HeapMemory
    stack: Stack
    trail: np.ndarray
    pdl: PDL
    depth: np.ndarray
    depth_limit: np.ndarray
    return_depth: np.ndarray


def init_machine(config: RuntimeConfig = DEFAULT_CONFIG.runtime) -> Machine:
    """Allocate and initialize a machine."""

    return Machine(
        registers=init_machine_registers(config),
        X=np.zeros(config.max_x_registers, dtype=np.uint16),
        heap=init_heap(config.heap_size),
        stack=init_stack(config),
        trail=init_trail(config.trail_size),
        pdl=init_pdl(config),
        depth=np.zeros(1, dtype=np.int32),
        depth_limit=np.zeros(1, dtype=np.int32),
        return_depth=np.zeros(1, dtype=np.int32),
    )


def reset_machine(machine: Machine) -> None:
    """Reset one machine's mutable execution state for reuse."""

    reset_machine_registers(machine.registers)
    machine.depth[0] = 0
    machine.depth_limit[0] = 0
    machine.return_depth[0] = 0
