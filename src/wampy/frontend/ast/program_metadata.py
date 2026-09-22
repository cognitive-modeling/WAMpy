r"""
# Program Metadata

This module stores metadata for each program node in dense arrays that mirror
the layout of the program AST.


## Example

Example AST for one program:

```
                     0:ADD
                    /     \
                1:MUL      4:Y
               /     \
            2:X       3:CONST
```

The AST and metadata are stored at matching node positions:

```
·             0         1         2         3         4         5
·         ╔═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗
·     AST ║   ADD   ║   MUL   ║    X    ║  CONST  ║    Y    ║    -    ║
·         ╚═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝
·
·         ╔═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗
· weights ║  1.00   ║  0.75   ║  0.25   ║  0.50   ║  0.25   ║  0.00   ║
·         ╚═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝
·         ╔═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗
·   usage ║    1    ║    1    ║    1    ║    1    ║    1    ║    0    ║
·         ╚═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝
·         ╔═════════╦═════════╦═════════╦═════════╦═════════╦═════════╗
·    kind ║   ROOT  ║   EXPR  ║  INPUT  ║ LITERAL ║  INPUT  ║  NO_NODE   ║
·         ╚═════════╩═════════╩═════════╩═════════╩═════════╩═════════╝
```



"""

from enum import IntEnum, unique
from typing import NamedTuple

import numpy as np

from wampy.frontend.ast.ops.analysis import (
    get_child_at,
    get_child_count,
    get_node_capacity,
    get_node_symbol,
    get_term_capacity,
    get_term_count,
    is_variable_symbol,
)
from wampy.frontend.ast.ops.mutation import add_child_node, remove_descendants
from wampy.frontend.ast.ops.transforms import copy_term
from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import (
    CoreSymID,
)


@unique
class SymbolKind(IntEnum):
    """Persistent synthesis classification for one AST node."""

    UNSET = 0
    START = 1
    BODY = 2
    TERM = 3
    SYMBOL = 4
    VARIABLE = 5


class ProgramMetadata(NamedTuple):
    """Sidecar arrays aligned with a :class:`~wampy.frontend.ast.Program`.

    ``symbol_kind[term_id, node_id]`` is per-node synthesis state;
    ``usage[term_id]`` and ``weights[term_id]`` are per-term state.
    """

    symbol_kind: np.ndarray
    usage: np.ndarray
    weights: np.ndarray


def init_program_metadata(program: ASTProgram) -> ProgramMetadata:
    """Allocate zeroed synthesis metadata for ``program``."""

    return ProgramMetadata(
        symbol_kind=np.full(
            (get_term_capacity(program), get_node_capacity(program)),
            np.uint8(SymbolKind.UNSET.value),
            dtype=np.uint8,
        ),
        usage=np.zeros(
            get_term_capacity(program),
            dtype=np.int64,
        ),
        weights=np.zeros(
            get_term_capacity(program),
            dtype=np.float64,
        ),
    )


def update_usage(
    metadata: ProgramMetadata,
    answer_usage: np.ndarray,
) -> None:
    """Overwrite term-aligned usage from one answer-usage row."""

    n_terms = metadata.usage.shape[0]
    n_answer_terms = answer_usage.shape[0]

    for term_id in range(n_terms):
        if term_id < n_answer_terms:
            metadata.usage[term_id] = np.int64(answer_usage[term_id])
        else:
            metadata.usage[term_id] = np.int64(0)


def invalidate_term_metadata(metadata: ProgramMetadata, term_id: int) -> None:
    """Invalidate metadata tied to one term's AST structure.

    Node classifications are maintained by the metadata-aware AST mutation
    helpers.  Usage and synthesis weight are term-level values, so every
    structural mutation clears both together here.
    """

    if term_id < 0 or term_id >= metadata.usage.shape[0]:
        return

    metadata.usage[term_id] = np.int64(0)
    metadata.weights[term_id] = np.float64(0.0)


def reset_program_metadata(
    metadata: ProgramMetadata,
    term_id: int = -1,
) -> None:
    """Reset all metadata, or one term when ``term_id`` is supplied.

    Node classifications, execution usage, and synthesis weights are all
    reset.  A negative ``term_id`` selects the whole sidecar.
    """

    if term_id < 0:
        for current_term in range(metadata.symbol_kind.shape[0]):
            for node_id in range(metadata.symbol_kind.shape[1]):
                metadata.symbol_kind[current_term, node_id] = np.uint8(SymbolKind.UNSET.value)
            invalidate_term_metadata(metadata, current_term)
        return

    if term_id >= metadata.symbol_kind.shape[0]:
        return

    for node_id in range(metadata.symbol_kind.shape[1]):
        metadata.symbol_kind[term_id, node_id] = np.uint8(SymbolKind.UNSET.value)
    invalidate_term_metadata(metadata, term_id)


def clear_metadata_node(
    metadata: ProgramMetadata,
    term_id: int,
    node_id: int,
) -> None:
    """Clear the persistent classification of one node."""

    metadata.symbol_kind[term_id, node_id] = np.uint8(SymbolKind.UNSET.value)


def clear_metadata_term(metadata: ProgramMetadata, term_id: int) -> None:
    """Clear one term's node classifications and term-level state."""

    if term_id < 0 or term_id >= metadata.symbol_kind.shape[0]:
        return

    for node_id in range(metadata.symbol_kind.shape[1]):
        metadata.symbol_kind[term_id, node_id] = np.uint8(SymbolKind.UNSET.value)
    invalidate_term_metadata(metadata, term_id)


def copy_metadata_term(
    metadata: ProgramMetadata,
    destination_term: int,
    source_term: int,
) -> None:
    """Copy one term's complete metadata into another term slot."""

    for node_id in range(metadata.symbol_kind.shape[1]):
        metadata.symbol_kind[destination_term, node_id] = metadata.symbol_kind[source_term, node_id]
    metadata.usage[destination_term] = metadata.usage[source_term]
    metadata.weights[destination_term] = metadata.weights[source_term]


def copy_term_with_metadata(
    program: ASTProgram,
    metadata: ProgramMetadata,
    destination_term: int,
    source_term: int,
) -> None:
    """Copy one AST term and its sidecar metadata together."""

    copy_term(program, destination_term, source_term)
    for node_id in range(get_node_capacity(program)):
        metadata.symbol_kind[destination_term, node_id] = metadata.symbol_kind[source_term, node_id]
    metadata.usage[destination_term] = metadata.usage[source_term]
    metadata.weights[destination_term] = metadata.weights[source_term]


def add_child_node_with_kind(
    program: ASTProgram,
    metadata: ProgramMetadata,
    term_id: int,
    parent_id: int,
    symbol_id: int,
    kind: int,
):
    """Allocate a child while writing its AST symbol and metadata kind."""

    if symbol_id == CoreSymID.NO_SYMBOL:
        return NO_NODE, False

    node_id, ok = add_child_node(
        program,
        term_id,
        parent_id,
        symbol_id,
    )
    if not ok:
        return NO_NODE, False

    invalidate_term_metadata(metadata, term_id)
    metadata.symbol_kind[term_id, node_id] = np.uint8(kind)
    return node_id, True


def allocate_child_node_with_kind(
    program: ASTProgram,
    metadata: ProgramMetadata,
    term_id: int,
    parent_id: int,
    symbol_id: int,
    kind: int,
):
    """Allocate a child for a builder that will assign its symbol next.

    The allocated cell receives a valid temporary symbol instead of the
    zero-valued ``NO_SYMBOL`` sentinel.  This keeps partially built ASTs
    distinct from unallocated cells while supporting stack-based builders.
    """

    node_id, ok = add_child_node(
        program,
        term_id,
        parent_id,
        CoreSymID.TRUE,
    )
    if not ok:
        return NO_NODE, False

    invalidate_term_metadata(metadata, term_id)
    metadata.symbol_kind[term_id, node_id] = np.uint8(kind)
    return node_id, True


def remove_descendants_with_metadata(
    term_id: int,
    parent_id: int,
    program: ASTProgram,
    metadata: ProgramMetadata,
) -> None:
    """Delete a subtree and clear both AST and synthesis metadata."""

    child_count = get_child_count(program, term_id, parent_id)
    for child_index in range(child_count):
        child_id = get_child_at(program, term_id, parent_id, child_index)
        if child_id != NO_NODE:
            _delete_metadata_subtree(term_id, child_id, program, metadata)

    remove_descendants(term_id, parent_id, program)

    invalidate_term_metadata(metadata, term_id)


def _delete_metadata_subtree(
    term_id: int,
    node_id: int,
    program: ASTProgram,
    metadata: ProgramMetadata,
) -> None:
    """Clear metadata for a subtree before the AST operation deletes it."""

    max_nodes = get_node_capacity(program)
    node_stack = np.empty(max_nodes, dtype=np.int64)
    top = 0

    node_stack[top] = node_id
    top += 1

    while top > 0:
        top -= 1
        current_id = node_stack[top]
        metadata.symbol_kind[term_id, current_id] = np.uint8(SymbolKind.UNSET.value)
        child_count = get_child_count(program, term_id, current_id)
        for child_index in range(child_count):
            child_id = get_child_at(program, term_id, current_id, child_index)
            if child_id == NO_NODE:
                continue
            if top >= max_nodes:
                raise ValueError("Metadata traversal stack is too small")
            node_stack[top] = child_id
            top += 1


_CONTEXT_TERM = np.uint8(0)
_CONTEXT_BODY = np.uint8(1)


def classify_program(
    program: ASTProgram,
    metadata: ProgramMetadata,
) -> None:
    """Classify the allocated nodes of an existing parsed program.

    Classification is positional: a ``CONJUNCTION`` in the clause body is a
    body-chain node, while a ``CONJUNCTION`` below a head or predicate argument is a
    concrete compound/tuple symbol.
    """

    max_nodes = get_node_capacity(program)

    for term_id in range(get_term_capacity(program)):
        for node_id in range(max_nodes):
            metadata.symbol_kind[term_id, node_id] = np.uint8(SymbolKind.UNSET.value)

    for term_id in range(get_term_count(program)):
        root_symbol = get_node_symbol(program, term_id, 0)
        if root_symbol == CoreSymID.CLAUSE or root_symbol == CoreSymID.QUERY:
            metadata.symbol_kind[term_id, 0] = np.uint8(SymbolKind.START.value)
        else:
            # A non-clause root is still a concrete AST symbol.  This keeps
            # classification useful for manually assembled terms.
            metadata.symbol_kind[term_id, 0] = _term_kind(program, root_symbol)

        node_stack = np.empty(max_nodes, dtype=np.uint16)
        context_stack = np.empty(max_nodes, dtype=np.uint8)
        top = 0

        root_child_count = get_child_count(program, term_id, 0)
        for child_index in range(root_child_count - 1, -1, -1):
            child_id = get_child_at(program, term_id, 0, child_index)
            if child_id == NO_NODE:
                continue

            context = _CONTEXT_TERM
            if root_symbol == CoreSymID.CLAUSE and child_index == 1:
                context = _CONTEXT_BODY

            node_stack[top] = child_id
            context_stack[top] = context
            top += 1

        while top > 0:
            top -= 1
            node_id = node_stack[top]
            context = context_stack[top]
            symbol_id = get_node_symbol(program, term_id, node_id)

            if context == _CONTEXT_BODY:
                metadata.symbol_kind[term_id, node_id] = np.uint8(SymbolKind.BODY.value)

                # A body-chain CONJUNCTION has a goal on the left and the
                # remaining chain on the right. Other conjunction nodes are
                # ordinary terms.
                if symbol_id == CoreSymID.CONJUNCTION:
                    child_count = get_child_count(program, term_id, node_id)
                    for child_index in range(child_count - 1, -1, -1):
                        child_id = get_child_at(
                            program,
                            term_id,
                            node_id,
                            child_index,
                        )
                        if child_id == NO_NODE:
                            continue
                        child_context = _CONTEXT_BODY if child_index == 1 else _CONTEXT_TERM
                        node_stack[top] = child_id
                        context_stack[top] = child_context
                        top += 1
                elif symbol_id == CoreSymID.NEGATION_AS_FAILURE:
                    child_count = get_child_count(program, term_id, node_id)
                    for child_index in range(child_count - 1, -1, -1):
                        child_id = get_child_at(
                            program,
                            term_id,
                            node_id,
                            child_index,
                        )
                        if child_id == NO_NODE:
                            continue
                        node_stack[top] = child_id
                        context_stack[top] = _CONTEXT_TERM
                        top += 1
                else:
                    child_count = get_child_count(program, term_id, node_id)
                    for child_index in range(child_count - 1, -1, -1):
                        child_id = get_child_at(
                            program,
                            term_id,
                            node_id,
                            child_index,
                        )
                        if child_id == NO_NODE:
                            continue
                        node_stack[top] = child_id
                        context_stack[top] = _CONTEXT_TERM
                        top += 1
                continue

            metadata.symbol_kind[term_id, node_id] = _term_kind(program, symbol_id)
            child_count = get_child_count(program, term_id, node_id)
            for child_index in range(child_count - 1, -1, -1):
                child_id = get_child_at(
                    program,
                    term_id,
                    node_id,
                    child_index,
                )
                if child_id == NO_NODE:
                    continue
                node_stack[top] = child_id
                context_stack[top] = _CONTEXT_TERM
                top += 1


def _term_kind(program: ASTProgram, symbol_id) -> np.uint8:
    if symbol_id == CoreSymID.NO_SYMBOL:
        return np.uint8(SymbolKind.TERM.value)
    if is_variable_symbol(program, symbol_id):
        return np.uint8(SymbolKind.VARIABLE.value)
    return np.uint8(SymbolKind.SYMBOL.value)
