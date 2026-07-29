# core/sample.py

import numpy as np
from numba import jit, uint16

from models.minimalist_machine.learn.grammar import GID, sample_rhs

from wampy.frontend.ast.program import (
    CONCEPT_ROOT,
    ROOT_ID,
    NONE,
    Program,
    Term,
    add_child_node,
    remove_all_descendant,
)

# ---------------------------------------------------------------
# Random nonterminal selection
# ---------------------------------------------------------------


@jit
def select_random_nonterminal(program):
    chosen_concept = uint16(CONCEPT_ROOT)
    chosen_node = uint16(ROOT_ID)
    seen = 0

    num_terms = program.symbol.shape[0]
    max_nodes = program.symbol.shape[1]

    for cid in range(num_terms):
        for nid in range(max_nodes):
            node_sym = program.symbol[cid, nid]

            # active if:
            #  - root
            #  - has at least one child (keeps your existing tests valid)
            #  - OR is allocated (enables selecting allocated leaves)
            has_child = program.children[cid, nid, 0] != NONE
            allocated = is_allocated(cid, nid, program.free_blocks, program.free_top)

            if nid == 0 or has_child or allocated:
                seen += 1
                if np.random.randint(seen) == 0:
                    chosen_concept = uint16(cid)
                    chosen_node = uint16(nid)

    return chosen_concept, chosen_node


@jit
def _is_selectable_node(program, term_id, node_id):
    has_child = program.children[term_id, node_id, 0] != NONE
    allocated = is_allocated(term_id, node_id, program.free_blocks, program.free_top)
    return node_id == 0 or has_child or allocated


@jit
def term_has_selectable_nodes(program, term_id):
    max_nodes = program.symbol.shape[1]
    for nid in range(max_nodes):
        if _is_selectable_node(program, term_id, nid):
            return True
    return False


@jit
def select_random_nonterminal_in_term(program, term_id):
    chosen_node = uint16(ROOT_ID)
    seen = 0
    max_nodes = program.symbol.shape[1]

    for nid in range(max_nodes):
        if _is_selectable_node(program, term_id, nid):
            seen += 1
            if np.random.randint(seen) == 0:
                chosen_node = uint16(nid)

    return chosen_node, seen > 0


@jit
def select_weighted_term_for_regeneration(program, term_usage, usage_weighting_enabled, usage_weight_smoothing):
    num_terms = program.symbol.shape[0]

    if not usage_weighting_enabled:
        return uint16(CONCEPT_ROOT), False

    total = 0.0
    weights = np.zeros(num_terms, dtype=np.float64)
    for tid in range(num_terms):
        if not term_has_selectable_nodes(program, tid):
            continue

        # Inverse weighting: prefer less-used terms for regeneration.
        # Larger usage -> smaller weight.
        w = 1.0 / (1.0 + float(term_usage[tid]) + usage_weight_smoothing)
        if w < 0.0:
            w = 0.0
        weights[tid] = w
        total += w

    if total <= 0.0:
        return uint16(CONCEPT_ROOT), False

    r = np.random.rand() * total
    cum = 0.0
    chosen = uint16(CONCEPT_ROOT)
    for tid in range(num_terms):
        if weights[tid] > 0.0:
            chosen = uint16(tid)
            break
    for tid in range(num_terms):
        w = weights[tid]
        if w <= 0.0:
            continue
        cum += w
        if r <= cum:
            chosen = uint16(tid)
            break

    return chosen, True


@jit
def is_allocated(term_id, node_id, free_blocks, free_top):
    # root is always allocated
    if node_id == 0:
        return True

    top = int(free_top[term_id])  # free stack uses indices [0 .. top-1]
    for i in range(top):
        if free_blocks[term_id, i] == node_id:
            return False
    return True


# ---------------------------------------------------------------
# Subtree regeneration
# ---------------------------------------------------------------


@jit
def copy_2d_u16(dst, src):
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            dst[i, j] = src[i, j]


@jit
def copy_1d_u16(dst, src):
    for i in range(dst.shape[0]):
        dst[i] = src[i]


@jit
def snapshot_term(program: Program, term_id):
    children = np.empty_like(program.children[term_id])
    copy_2d_u16(children, program.children[term_id])

    symbol = np.empty_like(program.symbol[term_id])
    copy_1d_u16(symbol, program.symbol[term_id])

    free_blocks = np.empty_like(program.free_blocks[term_id])
    copy_1d_u16(free_blocks, program.free_blocks[term_id])

    # Keep scalar fields as 1-element arrays for stable tuple typing
    free_top = np.array([program.free_top[term_id]], dtype=program.free_top.dtype)
    stm = np.array([program.stm[term_id]], dtype=program.stm.dtype)
    usage = np.array([program.usage[term_id]], dtype=program.usage.dtype)

    return Term(children, symbol, free_blocks, free_top, stm, usage)


@jit
def subtree_regeneration(program: Program, grammar, grammar_lookup, num_var=10):
    rand_cid, rand_nid = select_random_nonterminal(program)
    old_program_entry = snapshot_term(program, rand_cid)

    remove_all_descendant(rand_cid, rand_nid, program)
    ok = expand_subtree(rand_cid, rand_nid, grammar, grammar_lookup, program, num_var)

    return rand_cid, old_program_entry, ok


@jit
def subtree_regeneration_weighted(
    program: Program,
    grammar,
    grammar_lookup,
    term_usage,
    usage_weighting_enabled,
    usage_weight_smoothing,
    num_var=10,
):
    if not usage_weighting_enabled:
        return subtree_regeneration(program, grammar, grammar_lookup, num_var)

    rand_cid, found_term = select_weighted_term_for_regeneration(
        program,
        term_usage,
        usage_weighting_enabled,
        usage_weight_smoothing,
    )
    if not found_term:
        return subtree_regeneration(program, grammar, grammar_lookup, num_var)

    rand_nid, found_node = select_random_nonterminal_in_term(program, rand_cid)
    if not found_node:
        return subtree_regeneration(program, grammar, grammar_lookup, num_var)

    old_program_entry = snapshot_term(program, rand_cid)
    remove_all_descendant(rand_cid, rand_nid, program)
    ok = expand_subtree(rand_cid, rand_nid, grammar, grammar_lookup, program, num_var)
    return rand_cid, old_program_entry, ok


@jit
def _rhs_has_children(rhs):
    for i in range(1, rhs.shape[0]):
        if rhs[i] != 0:
            return True
    return False


@jit
def _node_arity(program, term_id, node_id):
    arity = 0
    n_child_slots = program.children.shape[2]
    for slot in range(n_child_slots):
        if program.children[term_id, node_id, slot] != NONE:
            arity += 1
    return arity


@jit
def _is_clause_head_node(program, term_id, node_id):
    if program.symbol[term_id, 0] != uint16(GID.CLAUSE.value):
        return False

    n_child_slots = program.children.shape[2]
    for slot in range(n_child_slots):
        cid = program.children[term_id, 0, slot]
        if cid != NONE:
            return cid == node_id
    return False


@jit
def _ensure_clause_head_arity(program, term_id, node_id, target_arity, num_var):
    if target_arity < 0:
        return True

    current_arity = _node_arity(program, term_id, node_id)
    if current_arity > target_arity:
        return False

    var_first = int(GID.VarFirst.value)
    var_last = var_first + num_var - 1
    if var_last < var_first:
        return False

    missing = target_arity - current_arity
    for _ in range(missing):
        c_id, ok = add_child_node(program, term_id, node_id, uint16(GID.VAR.value))
        if not ok:
            return False
        program.symbol[term_id, c_id] = uint16(np.random.randint(var_first, var_last + 1))

    return True


@jit
def expand_subtree(
    term_id, node_id, grammar, grammar_lookup, program, num_var, num_embeddings=0, task_fun=-1, task_arity=-1
):

    # 1) Determine LHS class for current node
    lhs_symbol = program.symbol[term_id, node_id]
    if lhs_symbol in grammar_lookup:
        lhs = grammar_lookup[lhs_symbol]
    else:
        # Unknown concrete symbols should be regenerated as Symbol, not Body.
        lhs = uint16(GID.SYMBOL.value)

    # 2) Sample production
    rhs = sample_rhs(grammar, lhs)

    # 3) Apply: rewrite node to RHS functor/nonterminal
    program.symbol[term_id, node_id] = rhs[0]

    # ---- NEW: follow unit productions like Base -> Symbol ----
    # If we rewrote into SYM/VAR without creating children, expand again under that class.
    if not _rhs_has_children(rhs):
        new_sym = program.symbol[term_id, node_id]
        if new_sym in grammar_lookup:
            new_lhs = grammar_lookup[new_sym]

            # Expand again only if we *changed* class (prevents looping on Symbol -> SYM)
            if new_lhs != lhs and new_lhs in (
                uint16(GID.S.value),
                uint16(GID.BODY.value),
                uint16(GID.SYMBOL.value),
            ):
                return expand_subtree(
                    term_id, node_id, grammar, grammar_lookup, program, num_var, num_embeddings, task_fun, task_arity
                )

    # 4) Create RHS children (if any)
    for i in range(1, rhs.shape[0]):
        child_symbol = rhs[i]
        if child_symbol == 0:
            continue

        c_id, ok = add_child_node(program, term_id, node_id, child_symbol)
        if not ok:
            return False

        if child_symbol in grammar_lookup:
            child_type = grammar_lookup[child_symbol]
        else:
            child_type = uint16(0)

        if child_type in (GID.S, GID.BODY, GID.SYMBOL):
            ok = expand_subtree(
                term_id, c_id, grammar, grammar_lookup, program, num_var, num_embeddings, task_fun, task_arity
            )
            if not ok:
                return False

    # 5) Terminalize SYM/VAR/EMB at this node
    sym = program.symbol[term_id, node_id]

    if sym == uint16(GID.SYMBOL.value):
        program.symbol[term_id, node_id] = random_binding_symbol(program, num_embeddings, task_fun, task_arity)
        if task_arity >= 0 and _is_clause_head_node(program, term_id, node_id):
            ok = _ensure_clause_head_arity(program, term_id, node_id, task_arity, num_var)
            if not ok:
                return False

    elif sym == uint16(GID.VAR.value):
        var_first = int(GID.VarFirst.value)
        var_last = var_first + num_var - 1
        program.symbol[term_id, node_id] = uint16(np.random.randint(var_first, var_last + 1))

    elif sym == uint16(GID.SymbolFirst.value):
        emb_first = int(GID.SymbolFirst.value)
        emb_last = emb_first + num_embeddings - 1
        max_last = int(GID.SymbolLast.value)
        if emb_last > max_last:
            emb_last = max_last
        if emb_last >= emb_first:
            program.symbol[term_id, node_id] = uint16(np.random.randint(emb_first, emb_last + 1))
        else:
            program.symbol[term_id, node_id] = uint16(GID.ERR.value)

    return True


@jit
def random_binding_symbol(program, num_embeddings, task_fun=-1, task_arity=-1):
    chosen = uint16(0)
    seen = 0
    n_terms = program.symbol.shape[0]
    n_child_slots = program.children.shape[2]
    task_fun_in_heads = 0

    for tid in range(n_terms):
        if program.symbol[tid, 0] != uint16(GID.CLAUSE.value):
            continue

        head_id = NONE
        for slot in range(n_child_slots):
            cid = program.children[tid, 0, slot]
            if cid != NONE:
                head_id = cid
                break
        if head_id == NONE:
            continue

        fun = program.symbol[tid, head_id]
        if fun == uint16(GID.TRUE.value):
            continue
        # Skip grammar/control IDs (TERM..VarLast), e.g. SYM=60014.
        if fun >= uint16(GID.S.value) and fun <= uint16(GID.VarLast.value):
            continue
        if fun == uint16(GID.ERR.value):
            continue

        if task_arity >= 0:
            if _node_arity(program, tid, head_id) != task_arity:
                continue

        seen += 1
        if np.random.randint(seen) == 0:
            chosen = uint16(fun)
        if task_fun >= 0 and fun == uint16(task_fun):
            task_fun_in_heads = 1

    if task_fun >= 0 and task_fun_in_heads == 0:
        query_fun = uint16(task_fun)
        if query_fun != uint16(GID.TRUE.value):
            if not (query_fun >= uint16(GID.S.value) and query_fun <= uint16(GID.VarLast.value)):
                if query_fun != uint16(GID.ERR.value):
                    seen += 1
                    if np.random.randint(seen) == 0:
                        chosen = query_fun

    if num_embeddings > 0:
        emb_first = int(GID.SymbolFirst.value)
        emb_last = emb_first + num_embeddings - 1
        max_last = int(GID.SymbolLast.value)
        if emb_last > max_last:
            emb_last = max_last
        for emb in range(emb_first, emb_last + 1):
            seen += 1
            if np.random.randint(seen) == 0:
                chosen = uint16(emb)

    if seen == 0:
        return uint16(GID.ERR.value)
    return chosen


@jit()
def restore_old_term(program: Program, term_id: int, old_term: Term):
    p_children = program.children
    p_symbol = program.symbol
    p_free_blocks = program.free_blocks
    p_free_top = program.free_top
    p_stm = program.stm
    p_usage = program.usage

    o_children = old_term.children
    o_symbol = old_term.symbol
    o_free_blocks = old_term.free_blocks
    o_free_top = old_term.free_top
    o_stm = old_term.stm
    o_usage = old_term.usage

    n_nodes = p_children.shape[1]
    n_slots = p_children.shape[2]
    for u in range(n_nodes):
        for k in range(n_slots):
            p_children[term_id, u, k] = o_children[u, k]

    for u in range(p_symbol.shape[1]):
        p_symbol[term_id, u] = o_symbol[u]

    for u in range(p_free_blocks.shape[1]):
        p_free_blocks[term_id, u] = o_free_blocks[u]

    p_free_top[term_id] = o_free_top[0]
    p_stm[term_id] = o_stm[0]
    p_usage[term_id] = o_usage[0]
