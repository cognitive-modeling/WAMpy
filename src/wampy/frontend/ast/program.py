r"""
# Knowledge Base

This module implements a knowledge layer built on top of structural concepts.
A **concept** is a tree-based structural object, composed of clauses/nodes
with associated symbols.

## Data Layout

We store K knowledge items (concepts) as a dense matrix:

Each node owns its first fixed-size child block, and overflow blocks are taken
from the same per-term block pool:
    child_blocks     : uint16[NUM_TERMS, BLOCK_CAPACITY, BLOCK_SIZE]
    child_count      : uint16[NUM_TERMS, MAX_NODES]
    next_child_block : uint16[NUM_TERMS, BLOCK_CAPACITY]

Blocks ``0 .. MAX_NODES - 1`` are permanently tied to node IDs.  The remaining
blocks are managed with a LIFO free stack and linked from the owning node.
The root is always node 0.

Symbols and node allocation remain separate:
    symbol          : uint16[NUM_TERMS, MAX_NODES]
    free_nodes      : uint16[NUM_TERMS, MAX_NODES - 1]
    free_node_top   : uint16[NUM_TERMS]


Note: All operations mutate preallocated arrays in place.

·               0     1     2     3     4     5     6        CONCEPT
·            ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
·   children ║  2  ║  3  ║  -  ║  -  ║  -  ║  -  ║  -  ║  0  slots
·            ║-----║-----║-----║-----║-----║-----║-----║
·            ║ ... ║ ... ║ ... ║ ... ║ ... ║ ... ║ ... ║  1
·            ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝


## Term (Concept / Clause) Adjacency Storage Using Fixed-Size Block Pools

This module implements a compact adjacency-list structure designed for
compiled tree/graph operations. The structure stores each node's list of
children in a linked sequence of small, fixed-size array blocks. This avoids
Python objects, supports predictable memory layout, and enables efficient
allocation and reclamation of blocks.


### Example for one concept

```
Tree structure for one concept

·                  0:A
·                 /   \
·               2:B    1:D
·               / \
·             4:D  5:E
·
·                0     1     2     3     4     5     6        NODES (blocks)
·             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
·   children  ║  2  ║  -  ║  4  ║  -  ║  -  ║  -  ║  -  ║  0  slots
·             ║-----║-----║-----║-----║-----║-----║-----║
·             ║  1  ║  -  ║  5  ║  -  ║  -  ║  -  ║  -  ║  1
·             ║-----║-----║-----║-----║-----║-----║-----║
·             ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  2
·             ║-----║-----║-----║-----║-----║-----║-----║
·             ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  3
·             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
·                |     |
·                └ children of node 0  (0 → 2, 1)
·                      └ children of node 1  (C → D, E)
·             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
·     symbol  ║  A  ║  D  ║  B  ║  -  ║  D  ║  E  ║  -  ║
·             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
·             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
· free_blocks ║  6  ║  3  ║  -  ║  -  ║  -  ║  -  ║  -  ║   (example ordering)
·             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
·             ╔═════╗
·   free_top  ║  1  ║   → top of stack is index 1 (block 2 is next free)
·             ╚═════╝
·             ╔═════╗
·       ROOT  ║  0  ║   → Allways 0, therefore it is a constant
·             ╚═════╝


## Example for the allocation of blocks using a LIFO stack

Init
    MAX_BLOCKS = 7                      ↓
    free_blocks    = [6, 5, 4, 3, 2, 1, 0]
    free_top       = 6   # next free block is free_blocks[0]

1. Step: Add node                    ↓  ␣
    free_blocks    = [6, 5, 4, 3, 2, 1, 0]
    free_top       =  5

2. Step: Add node                 ↓  ␣  ␣
    free_blocks    = [6, 5, 4, 3, 2, 1, 0]
    free_top       =  4

3. Step: Add node              ↓  ␣  ␣  ␣
    free_blocks    = [6, 5, 4, 3, 2, 1, 0]
    free_top       =  3

4. Step: Delete node 1            ↓  ␣  ␣
    free_blocks    = [6, 5, 4, 3, 1, 1, 0]
    free_top       =  4

5. Step: Add node              ↓  ␣  ␣  ␣
    free_blocks    = [6, 5, 4, 3, 1, 1, 0]
    free_top       =  3


## Canonical table form for evaluation

·                  0:A
·                 /   \
·               2:X    1:B
·               / \
·             4:Y  5:Z
·
·
·                0     1     2     3        NODES (blocks)
·             ╔═════╦═════╦═════╦═════╗
·   children  ║  A  ║  1  ║  4  ║  -  ║  0  evaluation link
·             ║-----║-----║-----║-----║
·             ║  X  ║  4  ║  -  ║  -  ║  1
·             ║-----║-----║-----║-----║
·             ║  Y  ║  -  ║  -  ║  -  ║  2
·             ║-----║-----║-----║-----║
·             ║  Z  ║  -  ║  -  ║  -  ║  3
·             ║-----║-----║-----║-----║
·             ║  B  ║  -  ║  -  ║  -  ║  4
·             ║-----║-----║-----║-----║
·             ║  -  ║  -  ║  -  ║  -  ║  6
·             ╚═════╩═════╩═════╩═════╝


# UPDATED INFORMATIONS:

This module implements the array-backed AST used by the WAMpy frontend.

## Data layout

Every allocated node owns exactly two physical child links::

    child_blocks     : uint16[NUM_TERMS, MAX_NODES, 2]
    node_child_counts: uint16[NUM_TERMS, MAX_NODES]
    node_symbols     : symid [NUM_TERMS, MAX_NODES]

The field name ``child_blocks`` is retained for API compatibility, but the
storage is no longer a variable-size child-block pool.  It is a fixed binary
link table.

Compiler-recognized structural nodes use the two links directly.  General
functors retain arbitrary logical arity through an internal ``ARGS`` spine::

    foo(a, b, c)

    foo --[0]--> ARGS --[1]--> ARGS --[1]--> ARGS
     |             |             |             |
     +--[1] tail   +--[0] a      +--[0] b      +--[0] c

For an argument-list parent, link 0 is the first ``ARGS`` cell and link 1 is a
cached tail pointer.  Each ``ARGS`` cell stores the argument value in link 0
and the next cell in link 1.  ``node_child_counts`` stores the parent's
logical arity, so callers continue to use ``get_child_count`` and
``get_child_at`` without depending on the physical representation.

The root is always node 0.  Nodes 1..MAX_NODES-1 are managed by a per-term
LIFO free stack.  All arrays are preallocated and mutated in place.
"""

from typing import NamedTuple

import numpy as np

from wampy.config import ASTConfig, WAMConfig
from wampy.frontend.ast.symbol_id import calculate_symbol_id_positions

NODE_ROOT_ID = 0
NODE_ID_DTYPE: np.ndarray = np.uint16
NO_NODE = NODE_ID_DTYPE(np.iinfo(NODE_ID_DTYPE).max)

# Compatibility names for callers that still import the former overflow-block
# API.  No overflow blocks are allocated by the binary representation.
BLOCK_ID_DTYPE: np.ndarray = np.uint16
NO_BLOCK = BLOCK_ID_DTYPE(np.iinfo(BLOCK_ID_DTYPE).max)

BINARY_CHILD_SLOTS = 2


class ASTProgram(NamedTuple):
    # Fixed binary links.  The legacy field name is retained to avoid making
    # storage consumers rename the hot array in the same change.
    node_links: np.ndarray
    node_symbols: np.ndarray
    node_arities: np.ndarray
    # Node allocator.
    free_node_stack: np.ndarray
    free_node_count: np.ndarray
    # Number of terms allocated in the contiguous prefix of the AST.
    term_count: np.ndarray
    # Per-term, query-local variable slot -> symbol-table ID for the source
    # spelling.  The maximum value of the configured unsigned symbol dtype is
    # the invalid/anonymous sentinel.
    variable_name_ids: np.ndarray
    # Symbol-ID layout boundaries.
    first_variable_symbol_id: int
    first_user_symbol_id: int


def init_ast_program(
    config: WAMConfig,
    num_terms: int | None = None,
) -> ASTProgram:
    """Allocate an empty AST with two physical links per node."""

    ast_config: ASTConfig = config.frontend.ast
    if num_terms is None:
        num_terms = ast_config.max_terms

    children: np.ndarray = np.full(
        (
            num_terms,
            ast_config.max_nodes_per_term,
            BINARY_CHILD_SLOTS,
        ),
        NO_NODE,
        dtype=NODE_ID_DTYPE,
    )
    child_count: np.ndarray = np.zeros(
        (num_terms, ast_config.max_nodes_per_term),
        dtype=NODE_ID_DTYPE,
    )
    # SymID.NO_SYMBOL == 0.
    symbol: np.ndarray = np.zeros(
        (num_terms, ast_config.max_nodes_per_term),
        dtype=ast_config.symbol_id_dtype,
    )

    free_capacity: int = ast_config.max_nodes_per_term - 1
    free_nodes: np.ndarray = np.empty(
        (num_terms, free_capacity),
        dtype=NODE_ID_DTYPE,
    )

    # Root node 0 is permanently allocated and is not part of the free stack.
    base: np.ndarray = np.arange(free_capacity, 0, -1, dtype=NODE_ID_DTYPE)
    for term_id in range(num_terms):
        free_nodes[term_id, :] = base
    free_top: np.ndarray = np.full(
        num_terms,
        free_capacity,
        dtype=NODE_ID_DTYPE,
    )
    term_count: np.ndarray = np.zeros(1, dtype=np.int32)
    variable_name_ids: np.ndarray = np.full(
        (
            num_terms,
            ast_config.max_variables_per_term,
        ),
        np.iinfo(ast_config.symbol_id_dtype).max,
        dtype=ast_config.symbol_id_dtype,
    )

    first_variable_symbol_id, first_user_symbol_id = calculate_symbol_id_positions(
        ast_config.max_variables_per_term,
    )

    return ASTProgram(
        node_links=children,
        node_symbols=symbol,
        node_arities=child_count,
        free_node_stack=free_nodes,
        free_node_count=free_top,
        term_count=term_count,
        variable_name_ids=variable_name_ids,
        first_variable_symbol_id=first_variable_symbol_id,
        first_user_symbol_id=first_user_symbol_id,
    )
