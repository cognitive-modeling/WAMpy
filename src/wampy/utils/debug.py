# wam/utils/debug_wam.py

# from models.rational_program.wam._config import DEFAULT_WAM_CONFIG
from wampy.runtime.stack import deref, TAG
from wampy.compiler import OP, CodeArea, ENTRY_INVALID_PC

# ---------------------------------------------------------------
# Stack
# ---------------------------------------------------------------


def debug_term_from_stack(stack, addr, atom_symbols):
    """
    Debug printer for WAM terms.
    Returns a string representation of the term.
    """

    p = deref(stack, addr)
    tag = stack.tags[p]

    if tag == TAG.REF and stack.heap[p] == p:
        return "_"

    if tag == TAG.CON:
        return atom_symbols.get(stack.heap[p], str(stack.heap[p]))

    if tag == TAG.STR:
        fun = stack.heap[p]
        symbol = atom_symbols.get(stack.heap[fun], str(stack.heap[fun]))
        arity = stack.heap[fun + 1]
        args = [debug_term_from_stack(stack, p + 1 + i, atom_symbols) for i in range(arity)]
        return f"{symbol}({', '.join(args)})"

    return "<?>"


def print_prolog_clause(program, term_id, binding_mapping):
    """
    Pretty-print a program clause as Prolog source code,
    with proper AND/2 flattening.
    """

    from models.minimalist_machine.learn.grammar import GID
    from wampy.frontend.ast.program import NONE

    children = program.children
    symbol = program.symbol

    # ----------------------------
    # Helpers
    # ----------------------------
    def children_of(node_id):
        return [c for c in children[term_id, node_id] if c != NONE]

    def sym_to_str(sym):
        # Variable
        if GID.VarFirst <= sym <= GID.VarLast:
            return f"X{sym - GID.VarFirst}"

        # Grammar TRUE
        if sym == GID.TRUE:
            return "true"

        # User symbol
        if sym in binding_mapping:
            return binding_mapping[sym]

        # Grammar fallback
        try:
            return GID(sym).name.lower()
        except Exception:
            return str(sym)

    # Term rendering
    def render_term(node_id):
        sym = int(symbol[term_id, node_id])
        args = children_of(node_id)

        symbol_text = sym_to_str(sym)

        if not args:
            return symbol_text

        rendered_args = [render_term(a) for a in args]
        return f"{symbol_text}({', '.join(rendered_args)})"

    # AND/2 flattening
    def flatten_and(node_id, acc):
        """
        Collects conjunction goals into acc list.
        """
        sym = int(symbol[term_id, node_id])

        if sym == GID.AND:
            args = children_of(node_id)
            if len(args) != 2:
                raise ValueError("AND node must have exactly 2 children")

            flatten_and(args[0], acc)
            flatten_and(args[1], acc)
        elif sym == GID.TRUE:
            # TRUE contributes nothing
            return
        else:
            acc.append(node_id)

    # Clause rendering
    root_sym = symbol[term_id, 0]
    if root_sym != GID.CLAUSE or root_sym != GID.QUERY:
        raise ValueError("Term is not a CLAUSE")

    clause_children = children_of(0)
    if not clause_children:
        raise ValueError("Malformed clause")

    head = clause_children[0]
    body_nodes = clause_children[1:]

    head_str = render_term(head)

    # ---- body ----
    goals = []
    for b in body_nodes:
        flatten_and(b, goals)

    if not goals:
        print(f"{head_str}.")
        return

    body_str = ", ".join(render_term(g) for g in goals)
    print(f"{head_str} :- {body_str}.")


def print_solution(stack, solutions, atom_symbols):
    """
    Print a single solution using Prolog-style variable names (_G1, _G2, ...).

    Parameters:
        stack        : WAM stack
        solution     : dict {var_addr: term_addr}
        atom_symbols : dict mapping atom ids to symbols
    """
    from wampy.utils.debug import debug_term_from_stack

    var_ids = {}

    def var_name(var):
        if var not in var_ids:
            var_ids[var] = len(var_ids) + 1
        return f"_G{var_ids[var]}"

    for var, term in sorted(solutions.items()):
        variable = var_name(var)
        val = debug_term_from_stack(stack, term, atom_symbols)
        print(f"  {variable} = {val}")


def debug_compiled_program(compiled_program: CodeArea, binding):
    """Debug the compiler instructions (code, entry)"""

    print("\n=== Predicate entry points ===")
    entry = compiled_program.entry
    for fun in range(entry.shape[0]):
        for arity in range(entry.shape[1]):
            pc = entry[fun, arity]
            if pc != ENTRY_INVALID_PC:
                symbol_text = binding.get(fun, str(fun))
                print(f"{symbol_text}/{arity} -> PC {pc}")

    print("\n=== WAM instructions ===")
    code = compiled_program.code[: compiled_program.pc.value]
    if len(code) == 0:
        print("(no instructions)")
        return

    for pc, instr in enumerate(code):
        op, a, b, c = instr
        print(f"{pc:04d}: {OP(op).name:<10} {a:3d} {b:3d} {c:3d}")


def compiled_global_opcode_trace(compiled_program: CodeArea):
    """
    Return the opcode sequence for the entire compiled program.

    Use for whole-program checks or single-clause programs.
    Not suitable for predicate-local assertions.
    """
    return [OP(instr[0]) for instr in compiled_program.code[: compiled_program.pc.value]]


def compiled_predicate_opcode_trace(compiled_program: CodeArea, fun: int, arity: int):
    """
    Return the opcode sequence starting at the entry point of one predicate.

    Use for predicate- and clause-local assertions.
    """
    pc = compiled_program.entry[fun, arity]
    return [OP(instr[0]) for instr in compiled_program.code[pc : compiled_program.pc.value]]
