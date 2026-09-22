"""Statically analyzable public facade for the WAMpy AST API.

Implementations live in :mod:`wampy.frontend.ast`; these explicit imports
keep the public namespace flat while preserving the original function
objects, signatures, annotations, and documentation.
"""

from numba.extending import register_jitable

from wampy.frontend.ast.ops.analysis import (
    _count_clause_shape,
    _count_clause_var_nodes,
    collect_head_nodes,
    count_clauses,
    count_nodes,
    get_child_at,
    get_child_count,
    get_clause_head_body,
    get_first_child,
    get_max_symbol_id,
    get_node_capacity,
    get_node_symbol,
    get_query_argument,
    get_query_arity,
    get_query_functor,
    get_query_predicate,
    get_symbol_dtype,
    get_term_capacity,
    get_term_count,
    get_variable_capacity,
    get_variable_symbol_id,
    is_compound_node,
    is_node_allocated,
    is_term_empty,
    is_user_symbol,
    is_variable_symbol,
    terms_have_equal_structure,
    variable_index,
)
from wampy.frontend.ast.ops.mutation import (
    _add_argument_child,
    _add_direct_child,
    _delete_physical_subtree,
    add_child_node,
    allocate_node,
    allocate_term,
    clear_child_blocks,
    clear_term,
    dfs_delete,
    free_node,
    remove_descendants,
    set_node_symbol,
)
from wampy.frontend.ast.ops.transforms import (
    copy_program,
    copy_term,
    copy_term_between_programs,
)
from wampy.frontend.ast.ops.validation import (
    _is_allocated,
    _valid_argument_chain,
    _valid_direct_links,
    is_semantically_valid_term,
    is_structurally_valid_term,
    is_valid_program,
    is_valid_symbol_id,
    is_valid_term,
    validate_program,
    validate_term,
)
from wampy.frontend.ast.program import (
    NO_BLOCK,
    NO_NODE,
    NODE_ROOT_ID,
    ASTProgram,
    init_ast_program,
)
from wampy.frontend.ast.program_metadata import (
    ProgramMetadata,
    SymbolKind,
    _delete_metadata_subtree,
    _term_kind,
    add_child_node_with_kind,
    allocate_child_node_with_kind,
    classify_program,
    clear_metadata_node,
    clear_metadata_term,
    copy_metadata_term,
    copy_term_with_metadata,
    init_program_metadata,
    invalidate_term_metadata,
    remove_descendants_with_metadata,
    reset_program_metadata,
    update_usage,
)
from wampy.frontend.ast.symbol_id import CoreSymID, calculate_symbol_id_positions

__all__ = [
    "NODE_ROOT_ID",
    "NO_BLOCK",
    "NO_NODE",
    "ASTProgram",
    "CoreSymID",
    "ProgramMetadata",
    "SymbolKind",
    "add_child_node",
    "add_child_node_with_kind",
    "allocate_term",
    "classify_program",
    "clear_child_blocks",
    "clear_metadata_node",
    "clear_metadata_term",
    "clear_term",
    "collect_head_nodes",
    "copy_metadata_term",
    "copy_program",
    "copy_term",
    "copy_term_between_programs",
    "copy_term_with_metadata",
    "count_clauses",
    "count_nodes",
    "free_node",
    "get_child_at",
    "get_child_count",
    "get_clause_head_body",
    "get_first_child",
    "get_max_symbol_id",
    "get_node_capacity",
    "get_node_symbol",
    "get_query_argument",
    "get_query_arity",
    "get_query_functor",
    "get_query_predicate",
    "get_symbol_dtype",
    "get_term_capacity",
    "get_term_count",
    "get_variable_capacity",
    "get_variable_symbol_id",
    "init_ast_program",
    "init_program_metadata",
    "invalidate_term_metadata",
    "is_compound_node",
    "is_node_allocated",
    "is_term_empty",
    "is_user_symbol",
    "is_variable_symbol",
    "remove_descendants",
    "remove_descendants_with_metadata",
    "reset_program_metadata",
    "set_node_symbol",
    "terms_have_equal_structure",
    "update_usage",
    "validate_program",
    "validate_term",
    "variable_index",
]

# AST model
register_jitable(init_ast_program)
register_jitable(inline="always")(calculate_symbol_id_positions)

# AST analysis
register_jitable(inline="always")(is_variable_symbol)
register_jitable(inline="always")(variable_index)
register_jitable(inline="always")(get_term_capacity)
register_jitable(inline="always")(get_term_count)
register_jitable(inline="always")(get_node_capacity)
register_jitable(inline="always")(get_node_symbol)
register_jitable(inline="always")(get_variable_capacity)
register_jitable(inline="always")(get_variable_symbol_id)
register_jitable(inline="always")(is_user_symbol)
register_jitable(inline="always")(is_term_empty)
register_jitable(is_compound_node)
register_jitable(get_clause_head_body)
register_jitable(get_child_count)
register_jitable(get_child_at)
register_jitable(get_first_child)
register_jitable(get_query_predicate)
register_jitable(get_query_functor)
register_jitable(get_query_arity)
register_jitable(get_query_argument)
register_jitable(is_node_allocated)
register_jitable(collect_head_nodes)
register_jitable(_count_clause_shape)
register_jitable(count_clauses)
register_jitable(terms_have_equal_structure)
register_jitable(count_nodes)
register_jitable(_count_clause_var_nodes)

# AST mutation
register_jitable(allocate_term)
register_jitable(set_node_symbol)
register_jitable(allocate_node)
register_jitable(free_node)
register_jitable(_add_direct_child)
register_jitable(_add_argument_child)
register_jitable(add_child_node)
register_jitable(_delete_physical_subtree)
register_jitable(dfs_delete)
register_jitable(clear_child_blocks)
register_jitable(clear_term)
register_jitable(remove_descendants)

# AST transforms
register_jitable(copy_program)
register_jitable(copy_term)
register_jitable(copy_term_between_programs)

# AST validation
register_jitable(_valid_direct_links)
register_jitable(_valid_argument_chain)
register_jitable(_is_allocated)
register_jitable(is_valid_symbol_id)
register_jitable(is_structurally_valid_term)
register_jitable(is_semantically_valid_term)
register_jitable(is_valid_term)
register_jitable(is_valid_program)
register_jitable(validate_term)
register_jitable(validate_program)

# Program metadata
register_jitable(init_program_metadata)
register_jitable(update_usage)
register_jitable(invalidate_term_metadata)
register_jitable(reset_program_metadata)
register_jitable(clear_metadata_node)
register_jitable(clear_metadata_term)
register_jitable(copy_metadata_term)
register_jitable(copy_term_with_metadata)
register_jitable(add_child_node_with_kind)
register_jitable(allocate_child_node_with_kind)
register_jitable(_delete_metadata_subtree)
register_jitable(remove_descendants_with_metadata)
register_jitable(_term_kind)
register_jitable(classify_program)
