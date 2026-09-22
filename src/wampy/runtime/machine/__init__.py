"""WAM machine registers and memory.

Public execution boundaries and instruction handlers take ``Machine``. Low-level
helpers receive the machine registers and flattened machine storage they need.

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

from wampy.runtime.machine.heap import (
    HeapMemory,
    deref,
    init_heap,
    make_const,
    make_functor,
    make_structure,
    make_var,
)
from wampy.runtime.machine.machine import Machine, init_machine, reset_machine
from wampy.runtime.machine.machine_registers import (
    Registers,
    init_machine_registers,
    reset_machine_registers,
)
from wampy.runtime.machine.pdl import PDL, init_pdl
from wampy.runtime.machine.stack import ChoicePointKind, ChoiceSlot, Stack, init_stack
from wampy.runtime.machine.trail import trail_var

__all__ = [
    "PDL",
    "ChoicePointKind",
    "ChoiceSlot",
    "HeapMemory",
    "Machine",
    "Registers",
    "Stack",
    "deref",
    "init_heap",
    "init_machine",
    "init_machine_registers",
    "init_pdl",
    "init_stack",
    "make_const",
    "make_functor",
    "make_structure",
    "make_var",
    "reset_machine",
    "reset_machine_registers",
    "trail_var",
]
