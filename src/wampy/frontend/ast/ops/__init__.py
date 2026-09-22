"""Internal categorized operations on the core AST model.

Import operations from their category-specific module, for example::

    from wampy.frontend.ast.ops.analysis import get_child_count
    from wampy.frontend.ast.ops.mutation import add_child_node
"""

from . import analysis, mutation, transforms, validation

__all__ = [
    "analysis",
    "mutation",
    "transforms",
    "validation",
]
