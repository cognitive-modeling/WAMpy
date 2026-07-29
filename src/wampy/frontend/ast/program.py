r"""
# Knowledge Base

This module implements a knowledge layer built on top of structural concepts.
A **concept** is a tree-based structural object, composed of clauses/nodes
with associated symbols.

## Data Layout

We store K knowledge items (concepts) as a dense matrix:

Each clause/node has an array in the matrix `children`, an array of length N:
Each block `b` holds up to BLOCK_SIZE child IDs in `children[b, :]`.
    children     : int16[NUM_CONCEPTS, MAX_BLOCKS, BLOCK_SIZE]

The root is always 0, therefore we use a const here. Represents the root
    of the tree and indicates the position of the first node.

We track how many slots of each knowledge vector are actually used:
    weight     : int16[NUM_CONCEPTS, ]

The value of each children is stored in:
    symbol      : int16[NUM_CONCEPTS, MAX_BLOCKS, BLOCK_SIZE]


Blocks are taken from a preallocated pool of size MAX_BLOCKS. They are managed
with a simple LIFO free-stack:
    free_blocks : int16[NUM_CONCEPTS, MAX_BLOCKS] array of available block indices
    free_top    : int16[NUM_CONCEPTS,] stack pointer (number of free blocks)

Allocation (`alloc_block`) pops a block from the free stack. Freeing a block
(`free_block`) wipes its contents and pushes it back onto the stack.


Note: All operations are Numba-compatible and side-effect on preallocated arrays.

·               0     1     2     3     4     5     6        CONCEPT
·            ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
·   children ║  2  ║  3  ║  -  ║  -  ║  -  ║  -  ║  -  ║  0  slots
·            ║-----║-----║-----║-----║-----║-----║-----║
·            ║ ... ║ ... ║ ... ║ ... ║ ... ║ ... ║ ... ║  1
·            ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
·            ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
·    weights ║  1  ║  2  ║  4  ║  6  ║  0  ║  0  ║  0  ║
·            ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝


## Term (Concept / Clause) Adjacency Storage Using Fixed-Size Block Pools

This module implements a compact adjacency-list structure designed for
Numba-compiled tree/graph operations. The structure stores each node's list of
children in a linked sequence of small, fixed-size array blocks. This avoids
Python objects, supports predictable memory layout, and enables efficient
allocation and reclamation of blocks inside Numba `jit` functions.


### Example for one concept

```
Tree structure for one concept

                  0:A
                 /   \
               2:B    1:D
               / \
             4:D  5:E

                0     1     2     3     4     5     6        NODES (blocks)
             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
   children  ║  2  ║  -  ║  4  ║  -  ║  -  ║  -  ║  -  ║  0  slots
             ║-----║-----║-----║-----║-----║-----║-----║
             ║  1  ║  -  ║  5  ║  -  ║  -  ║  -  ║  -  ║  1
             ║-----║-----║-----║-----║-----║-----║-----║
             ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  2
             ║-----║-----║-----║-----║-----║-----║-----║
             ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  -  ║  3
             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
                |     |
                └ children of node 0  (0 → 2, 1)
                      └ children of node 1  (C → D, E)
             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
     symbol  ║  A  ║  D  ║  B  ║  -  ║  D  ║  E  ║  -  ║
             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
             ╔═════╦═════╦═════╦═════╦═════╦═════╦═════╗
 free_blocks ║  6  ║  3  ║  -  ║  -  ║  -  ║  -  ║  -  ║   (example ordering)
             ╚═════╩═════╩═════╩═════╩═════╩═════╩═════╝
             ╔═════╗
   free_top  ║  1  ║   → top of stack is index 1 (block 2 is next free)
             ╚═════╝
             ╔═════╗
       ROOT  ║  0  ║   → Allways 0, therefore it is a constant
             ╚═════╝


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

                  0:A
                 /   \
               2:X    1:B
               / \
             4:Y  5:Z


                0     1     2     3        NODES (blocks)
             ╔═════╦═════╦═════╦═════╗
   children  ║  A  ║  1  ║  4  ║  -  ║  0  evaluation link
             ║-----║-----║-----║-----║
             ║  X  ║  4  ║  -  ║  -  ║  1
             ║-----║-----║-----║-----║
             ║  Y  ║  -  ║  -  ║  -  ║  2
             ║-----║-----║-----║-----║
             ║  Z  ║  -  ║  -  ║  -  ║  3
             ║-----║-----║-----║-----║
             ║  B  ║  -  ║  -  ║  -  ║  4
             ║-----║-----║-----║-----║
             ║  -  ║  -  ║  -  ║  -  ║  6
             ╚═════╩═════╩═════╩═════╝


## Notes

- To use more than the fixed amount of block space, a linked-list pointer
  `next_block` can be used from each block to the next. ( If a node's
  child list overflows one block, an additional block is chained via
  `next_block[b] = next_block_index`. A value of const NONE indicates the
  end of the chain.)
- If blocks are ever allocated from a shared pool rather than being tied
  1-to-1 with node IDs, a per-node `head` array becomes necessary to record
  which block stores the first adjacency segment for each node.

"""

from typing import Any, Callable, NamedTuple

import numpy as np
from numpy.typing import NDArray
from numba import jit, uint16

from wampy.frontend.ast.symbol_id import SymID
from wampy.config import WAMConfig

CONCEPT_ROOT = 0
ROOT_ID = 0
NONE = np.uint16(0xFFFF)  # Sentinel for unused slots, blocks, and heads


# --- define namedtuple at module scope (REQUIRED for caching) ---
class Program(NamedTuple):
    children: np.ndarray
    symbol: np.ndarray
    free_blocks: np.ndarray
    free_top: np.ndarray
    stm: NDArray[np.bool_]
    usage: np.ndarray


class Term(NamedTuple):
    children: np.ndarray
    symbol: np.ndarray
    free_blocks: np.ndarray
    free_top: np.ndarray
    stm: NDArray[np.bool_]
    usage: np.ndarray


@jit(cache=True)
def init_ast_structure(
    n_terms: int,
    max_terms_nodes: int,
    max_terms_nodes_childs: int,
) -> Program:

    children = np.full(
        (n_terms, max_terms_nodes, max_terms_nodes_childs),
        NONE,
        dtype=np.uint16,
    )

    symbol = np.zeros((n_terms, max_terms_nodes), dtype=np.uint16)
    symbol[:, 0] = SymID.EMPTY

    free_blocks = np.empty((n_terms, max_terms_nodes), dtype=np.uint16)

    base = np.arange(max_terms_nodes - 1, -1, -1, dtype=np.uint16)

    for i in range(n_terms):
        free_blocks[i, :] = base

    free_top = np.full(n_terms, max_terms_nodes - 1, dtype=np.uint16)

    stm = np.zeros(n_terms, dtype=np.bool_)
    usage = np.zeros(n_terms, dtype=np.int64)

    return Program(children, symbol, free_blocks, free_top, stm, usage)


@jit(cache=True)
def init_program(config: WAMConfig):

    return init_ast_structure(
        config.ast.num_terms,
        config.ast.max_terms_nodes,
        config.ast.max_terms_nodes_childs,
    )


@jit(cache=True)
def update_usage(program: Program, answer_usage: np.ndarray) -> None:
    """
    Overwrite program usage from one answer-usage row.
    """
    n_terms = program.usage.shape[0]
    n_answer_terms = answer_usage.shape[0]

    for term_id in range(n_terms):
        if term_id < n_answer_terms:
            program.usage[term_id] = np.int64(answer_usage[term_id])
        else:
            program.usage[term_id] = np.int64(0)


# ---------------------------------------------------------------
# Stack primitives (in-place)
# ---------------------------------------------------------------


@jit(cache=True)
def allocate_block(term_id, free_blocks, free_top):
    if free_top[term_id] == 0:
        return NONE, False  # failed: Out of adjacency blocks

    free_top[term_id] -= 1
    node_id = free_blocks[term_id, free_top[term_id]]
    return node_id, True


@jit(cache=True)
def free_space_block(term_id, block_id, children, free_blocks, free_top):

    num_child_slots = children.shape[2]  # MAX_CONCEPT_NODES_CHILDS

    for i in range(num_child_slots):
        children[term_id, block_id, i] = NONE

    free_blocks[term_id, free_top[term_id]] = block_id
    free_top[term_id] += 1


# ---------------------------------------------------------------
# Tree mutation
# ---------------------------------------------------------------


@jit(cache=True)
def add_child_node(program: Program, term_id, parent_id, child_symbol):
    """
    Inserts a child into the parent's adjacency list.
    Also sets symbol[child] = child_symbol.

    Note: We could also maintain an auxiliary pointer array that tracks the
        next free child slot for each parent, avoiding the need to scan
        up to BLOCK_SIZE entries. Given that BLOCK_SIZE is small (≤10),
        the current linear scan is already efficient and keeps the structure simpler.

    Returns (child_id, ok).
    """
    children = program.children
    symbol = program.symbol
    free_blocks = program.free_blocks
    free_top = program.free_top

    child_id, ok = allocate_block(term_id, free_blocks, free_top)
    if not ok:
        return NONE, False

    symbol[term_id, child_id] = child_symbol

    num_child_slots = children.shape[2]  # MAX_CONCEPT_NODES_CHILDS
    for i in range(num_child_slots):
        if children[term_id, parent_id, i] == NONE:
            children[term_id, parent_id, i] = child_id
            return child_id, True

    raise RuntimeError("Parent node has no available child slot.")


@jit(cache=True)
def remove_all_descendant(term_id, parent_id, program: Program):
    """
    Removes all descendants of parent_id.
    Keeps parent_id and its symbol.
    """

    # for each direct child of parent_id, delete its subtree
    num_child_slots = program.children.shape[2]  # MAX_CONCEPT_NODES_CHILDS
    for i in range(num_child_slots):

        c_id = program.children[term_id, parent_id, i]
        if c_id != NONE:
            _dfs_delete(term_id, c_id, program)
            program.children[term_id, parent_id, i] = NONE


@jit(cache=True)
def _dfs_delete(term_id, node_id, program: Program):
    """
    Recursively delete subtree rooted at node_id.
    Frees node_id itself and all its descendants.
    Does NOT clear its parent slot; caller is responsible for that.
    """
    # delete children of node_id
    for cid in program.children[term_id, node_id, :]:
        if cid != NONE:
            _dfs_delete(term_id, cid, program)

    # free node_id's block (symbols are not touched)
    free_space_block(term_id, node_id, program.children, program.free_blocks, program.free_top)


@jit(cache=True)
def count_nodes(program: Program) -> np.int64:
    """
    Count used nodes across all terms by inverting free_top.
    Includes root node 0 for each term.
    """
    max_terms_nodes = program.children.shape[1]
    n_terms = program.children.shape[0]

    total = np.int64(0)
    for t in range(n_terms):
        total += np.int64(max_terms_nodes) - np.int64(program.free_top[t])
    return total


@jit(cache=True)
def count_clause_var_nodes(program):
    n_vars = np.int64(0)
    for term_id in range(program.symbol.shape[0]):
        if program.symbol[term_id, 0] != np.uint16(SymID.CLAUSE.value):
            continue
        for node_id in range(program.symbol.shape[1]):
            sym = program.symbol[term_id, node_id]
            if sym >= np.uint16(SymID.VAR_FIRST.value) and sym <= np.uint16(SymID.VAR_LAST.value):
                n_vars += np.int64(1)
    return n_vars
