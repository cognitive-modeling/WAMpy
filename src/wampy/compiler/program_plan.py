"""Sparse head-only lowering from AST clauses to predicate compile groups."""

from typing import NamedTuple

import numpy as np

from wampy.compiler.compiler_state import NO_PREDICATE_SLOT, CompilerState
from wampy.compiler.predicates import find_predicate_slot, resolve_predicate_slot
from wampy.config import CompilerConfig
from wampy.frontend.ast.ops import analysis as ast
from wampy.frontend.ast.program import NO_NODE, ASTProgram
from wampy.frontend.ast.symbol_id import CoreSymID

PLAN_INVALID_INDEX = -1

PLAN_CLAUSE_TID = 0
PLAN_CLAUSE_HEAD = 1
PLAN_CLAUSE_BODY = 2
PLAN_CLAUSE_NEXT = 3
PLAN_CLAUSE_WIDTH = 4

PLAN_SIGNATURE_SLOT = 0
PLAN_SIGNATURE_ARITY = 1
PLAN_SIGNATURE_FIRST_CLAUSE = 2
PLAN_SIGNATURE_LAST_CLAUSE = 3
PLAN_SIGNATURE_WIDTH = 4


class ProgramCompilePlan(NamedTuple):
    """Compact predicate-signature groups linked through source-order clauses."""

    clauses: np.ndarray
    signatures: np.ndarray
    clause_count: int
    signature_count: int


def _count_and_register_plan_clauses(
    program: ASTProgram,
    compiler_state: CompilerState,
    config: CompilerConfig,
    max_x_registers,
    head_stack,
):
    """Return concrete head count after validating and registering head slots."""

    clause_count = 0

    for tid in range(ast.get_term_count(program)):
        head_root, _ = ast.get_clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        top = 0
        head_stack[top] = head_root
        top += 1

        while top > 0:
            top -= 1
            node = head_stack[top]
            symbol = ast.get_node_symbol(program, tid, node)

            if symbol == CoreSymID.CONJUNCTION:
                child_count = ast.get_child_count(program, tid, node)
                for child_index in range(child_count - 1, -1, -1):
                    child = ast.get_child_at(program, tid, node, child_index)
                    if child == NO_NODE:
                        continue
                    if top >= head_stack.shape[0]:
                        raise ValueError("Clause head nesting is too deep")
                    head_stack[top] = child
                    top += 1
                continue

            if symbol == CoreSymID.TRUE:
                continue

            arity = ast.get_child_count(program, tid, node)
            if arity > max_x_registers:
                raise ValueError("Predicate arity exceeds X register capacity")
            if arity > config.max_arity:
                raise ValueError("ENTRY_MAX_ARITY too small for program")

            resolve_predicate_slot(compiler_state, int(symbol))
            clause_count += 1

    return clause_count


def build_program_compile_plan(
    program: ASTProgram,
    compiler_state: CompilerState,
    config: CompilerConfig,
    max_x_registers,
) -> ProgramCompilePlan:
    """Lower clause heads into sparse signature groups and clause links.

    The first pass validates and registers concrete heads.  The second pass
    stores each concrete head exactly once, joining source-order clauses for
    the same predicate slot and arity with a linked chain.  Clause bodies are
    intentionally not traversed here.
    """

    head_stack = np.empty(ast.get_node_capacity(program), dtype=np.int32)
    clause_count = _count_and_register_plan_clauses(
        program,
        compiler_state,
        config,
        max_x_registers,
        head_stack,
    )
    clauses = np.empty((clause_count, PLAN_CLAUSE_WIDTH), dtype=np.int32)
    signatures = np.empty((clause_count, PLAN_SIGNATURE_WIDTH), dtype=np.int32)

    used_clause_count = 0
    signature_count = 0

    for tid in range(ast.get_term_count(program)):
        head_root, body_root = ast.get_clause_head_body(program, tid)
        if head_root == NO_NODE:
            continue

        top = 0
        head_stack[top] = head_root
        top += 1

        while top > 0:
            top -= 1
            node = head_stack[top]
            symbol = ast.get_node_symbol(program, tid, node)

            if symbol == CoreSymID.CONJUNCTION:
                child_count = ast.get_child_count(program, tid, node)
                for child_index in range(child_count - 1, -1, -1):
                    child = ast.get_child_at(program, tid, node, child_index)
                    if child == NO_NODE:
                        continue
                    if top >= head_stack.shape[0]:
                        raise ValueError("Clause head nesting is too deep")
                    head_stack[top] = child
                    top += 1
                continue

            if symbol == CoreSymID.TRUE:
                continue

            arity = ast.get_child_count(program, tid, node)
            predicate_slot = find_predicate_slot(compiler_state, int(symbol))
            if predicate_slot == NO_PREDICATE_SLOT:
                raise ValueError("Predicate slot was not registered")

            signature_index = PLAN_INVALID_INDEX
            for candidate_index in range(signature_count):
                if (
                    signatures[candidate_index, PLAN_SIGNATURE_SLOT] == predicate_slot
                    and signatures[candidate_index, PLAN_SIGNATURE_ARITY] == arity
                ):
                    signature_index = candidate_index
                    break

            if signature_index == PLAN_INVALID_INDEX:
                signature_index = signature_count
                signatures[signature_index, PLAN_SIGNATURE_SLOT] = predicate_slot
                signatures[signature_index, PLAN_SIGNATURE_ARITY] = arity
                signatures[signature_index, PLAN_SIGNATURE_FIRST_CLAUSE] = PLAN_INVALID_INDEX
                signatures[signature_index, PLAN_SIGNATURE_LAST_CLAUSE] = PLAN_INVALID_INDEX
                signature_count += 1

            clause_index = used_clause_count
            clauses[clause_index, PLAN_CLAUSE_TID] = tid
            clauses[clause_index, PLAN_CLAUSE_HEAD] = node
            clauses[clause_index, PLAN_CLAUSE_BODY] = body_root
            clauses[clause_index, PLAN_CLAUSE_NEXT] = PLAN_INVALID_INDEX

            previous_clause = signatures[signature_index, PLAN_SIGNATURE_LAST_CLAUSE]
            if previous_clause == PLAN_INVALID_INDEX:
                signatures[signature_index, PLAN_SIGNATURE_FIRST_CLAUSE] = clause_index
            else:
                clauses[previous_clause, PLAN_CLAUSE_NEXT] = clause_index
            signatures[signature_index, PLAN_SIGNATURE_LAST_CLAUSE] = clause_index
            used_clause_count += 1

    if used_clause_count != clause_count:
        raise ValueError("Clause plan head count changed during lowering")

    return ProgramCompilePlan(
        clauses=clauses,
        signatures=signatures,
        clause_count=clause_count,
        signature_count=signature_count,
    )
