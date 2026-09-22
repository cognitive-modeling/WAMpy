"""AST data model and metadata API for the WAM compiler frontend.

AST operation implementations are organized under
:mod:`wampy.frontend.ast.ops` and are exposed through :mod:`wampy.ast`:

    import wampy.ast as ast

    head, body = ast.get_clause_head_body(program, term_id)
    arity = ast.get_child_count(program, term_id, head)
    child, ok = ast.add_child_node(program, term_id, head, symbol)

Only explicitly imported data-model and metadata names are public here.
Operation modules and private allocator helpers are implementation details;
the supported flat operation namespace is :mod:`wampy.ast`.


ASTProgram
    private storage representation
        child_blocks
        child_count
        next_child_block
        symbol
        free_nodes
        free_node_top
        free_child_blocks
        free_child_block_top

             ▲
             │ ONLY
             │
frontend/ast/ops/
    analysis.py
    mutation.py
    transforms.py
    validation.py

             ▲
             │ public AST operations
             │
    parser / compiler / renderer / API
"""

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
    add_child_node_with_kind,
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
from wampy.frontend.ast.symbol_id import CoreSymID

__all__ = [
    "NODE_ROOT_ID",
    "NO_BLOCK",
    "NO_NODE",
    "ASTProgram",
    "CoreSymID",
    "ProgramMetadata",
    "SymbolKind",
    "add_child_node_with_kind",
    "classify_program",
    "clear_metadata_node",
    "clear_metadata_term",
    "copy_metadata_term",
    "copy_term_with_metadata",
    "init_ast_program",
    "init_program_metadata",
    "invalidate_term_metadata",
    "remove_descendants_with_metadata",
    "reset_program_metadata",
    "update_usage",
]
