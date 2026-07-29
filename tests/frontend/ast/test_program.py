import numpy as np

from wampy.frontend.ast.program import (
    NONE,
    init_ast_structure,
    add_child_node,
    remove_all_descendant,
    update_usage,
)


def test_init_program():
    n_terms = 2
    max_terms_nodes = 10
    max_terms_nodes_childs = 4

    program = init_ast_structure(n_terms, max_terms_nodes, max_terms_nodes_childs)

    assert program.symbol.shape == (2, 10)
    assert program.children.shape[0] == 2
    assert program.children.shape[1] == 10

    for t in range(2):
        assert program.free_top[t] == 9


def test_add_child_node():
    n_terms = 2
    max_terms_nodes = 10
    max_terms_nodes_childs = 4

    program = init_ast_structure(n_terms, max_terms_nodes, max_terms_nodes_childs)
    term_id = 0

    cid_10, ok = add_child_node(program, term_id, 0, 10)
    assert ok
    assert cid_10 != NONE
    assert program.symbol[term_id, cid_10] == 10

    cid_20, ok = add_child_node(program, term_id, 0, 20)
    assert ok
    assert program.symbol[term_id, cid_20] == 20

    root_children = program.children[term_id, 0]
    assert cid_10 in root_children
    assert cid_20 in root_children


def test_remove_all_descendant():
    n_terms = 2
    max_terms_nodes = 10
    max_terms_nodes_childs = 4

    program = init_ast_structure(n_terms, max_terms_nodes, max_terms_nodes_childs)
    term_id = 0

    cid_10, _ = add_child_node(program, term_id, 0, 10)
    cid_20, _ = add_child_node(program, term_id, 0, 20)
    cid_30, _ = add_child_node(program, term_id, cid_10, 30)
    cid_31, _ = add_child_node(program, term_id, cid_30, 31)

    remove_all_descendant(term_id, cid_10, program)

    assert np.all(program.children[term_id, cid_10] == NONE)

    root_children = program.children[term_id, 0]
    assert cid_10 in root_children
    assert cid_20 in root_children


def test_block_reuse_after_deletion():
    n_terms = 2
    max_terms_nodes = 10
    max_terms_nodes_childs = 4

    program = init_ast_structure(n_terms, max_terms_nodes, max_terms_nodes_childs)
    term_id = 0

    cid_10, _ = add_child_node(program, term_id, 0, 10)
    cid_20, _ = add_child_node(program, term_id, 0, 20)
    add_child_node(program, term_id, cid_10, 30)

    free_top_before = int(program.free_top[term_id])

    remove_all_descendant(term_id, cid_10, program)

    free_top_after = int(program.free_top[term_id])
    assert free_top_after > free_top_before

    cid_new, ok = add_child_node(program, term_id, cid_20, 99)
    assert ok
    assert program.symbol[term_id, cid_new] == 99


def test_update_usage_overwrites_usage_rows():
    program = init_ast_structure(4, 10, 4)

    update_usage(program, np.array([1, 0, 3, 2], dtype=np.int64))
    update_usage(program, np.array([2, 2, 0, 1], dtype=np.int64))

    assert np.array_equal(program.usage, np.array([2, 2, 0, 1], dtype=np.int64))


def test_update_usage_handles_short_answer_row():
    program = init_ast_structure(4, 10, 4)
    update_usage(program, np.array([9, 9, 9, 9], dtype=np.int64))

    update_usage(program, np.array([5, 7], dtype=np.int64))

    assert np.array_equal(program.usage, np.array([5, 7, 0, 0], dtype=np.int64))
