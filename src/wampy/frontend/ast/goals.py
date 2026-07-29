from collections import namedtuple

from numba import jit

from wampy.config import WAMConfig
from wampy.frontend.ast.program import Program, init_ast_structure


Goals = namedtuple(
    "Goals",
    [
        "program",
        "fun",
        "arity",
        "args",
        "term_id",
    ],
)


@jit(cache=True)
def init_goals(
    num_goals: int,
    config: WAMConfig,
) -> Program:

    return init_ast_structure(
        num_goals,
        max_terms_nodes=config.ast.max_terms_nodes,
        max_terms_nodes_childs=config.ast.max_terms_nodes_childs,
    )


@jit(cache=True)
def init_goal(config: WAMConfig) -> Program:
    return init_goals(1, config)
