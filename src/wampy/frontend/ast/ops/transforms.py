"""Higher-level semantic rewrites for the array-backed AST."""

from wampy.frontend.ast.ops.analysis import get_node_capacity
from wampy.frontend.ast.ops.mutation import clear_term
from wampy.frontend.ast.program import ASTProgram


def copy_program(program: ASTProgram) -> ASTProgram:
    """Return an independent copy of an AST program."""
    return ASTProgram(
        node_links=program.node_links.copy(),
        node_arities=program.node_arities.copy(),
        node_symbols=program.node_symbols.copy(),
        free_node_stack=program.free_node_stack.copy(),
        free_node_count=program.free_node_count.copy(),
        term_count=program.term_count.copy(),
        variable_name_ids=program.variable_name_ids.copy(),
        first_variable_symbol_id=program.first_variable_symbol_id,
        first_user_symbol_id=program.first_user_symbol_id,
    )


def copy_term(
    program: ASTProgram,
    destination_term: int,
    source_term: int,
) -> None:
    """Replace one term slot with the complete contents of another slot."""
    max_nodes = get_node_capacity(program)
    max_free = program.free_node_stack.shape[1]

    for node_id in range(max_nodes):
        program.node_symbols[destination_term, node_id] = program.node_symbols[source_term, node_id]
        program.node_arities[destination_term, node_id] = program.node_arities[source_term, node_id]
        program.node_links[destination_term, node_id, 0] = program.node_links[
            source_term, node_id, 0
        ]
        program.node_links[destination_term, node_id, 1] = program.node_links[
            source_term, node_id, 1
        ]

    for node_id in range(max_free):
        program.free_node_stack[destination_term, node_id] = program.free_node_stack[
            source_term, node_id
        ]

    program.free_node_count[destination_term] = program.free_node_count[source_term]
    for variable_slot in range(program.variable_name_ids.shape[1]):
        program.variable_name_ids[destination_term, variable_slot] = program.variable_name_ids[
            source_term,
            variable_slot,
        ]


def copy_term_between_programs(
    source_program: ASTProgram,
    source_term: int,
    destination_program: ASTProgram,
    destination_term: int,
) -> None:
    """Copy one complete term between compatible AST programs."""
    if source_program.node_symbols.shape[1] != destination_program.node_symbols.shape[1]:
        raise ValueError("AST programs must have the same node capacity")
    if source_program.node_links.shape[2] != destination_program.node_links.shape[2]:
        raise ValueError("AST programs must have the same child-slot capacity")
    if source_program.variable_name_ids.shape[1] != destination_program.variable_name_ids.shape[1]:
        raise ValueError("AST programs must have the same variable capacity")

    clear_term(destination_program, destination_term)

    for node_id in range(source_program.node_symbols.shape[1]):
        destination_program.node_symbols[destination_term, node_id] = source_program.node_symbols[
            source_term,
            node_id,
        ]
        destination_program.node_arities[destination_term, node_id] = source_program.node_arities[
            source_term,
            node_id,
        ]
        for slot in range(source_program.node_links.shape[2]):
            destination_program.node_links[destination_term, node_id, slot] = (
                source_program.node_links[source_term, node_id, slot]
            )

    for node_id in range(source_program.free_node_stack.shape[1]):
        destination_program.free_node_stack[destination_term, node_id] = (
            source_program.free_node_stack[source_term, node_id]
        )

    for variable_slot in range(source_program.variable_name_ids.shape[1]):
        destination_program.variable_name_ids[destination_term, variable_slot] = (
            source_program.variable_name_ids[source_term, variable_slot]
        )

    destination_program.free_node_count[destination_term] = source_program.free_node_count[
        source_term
    ]
