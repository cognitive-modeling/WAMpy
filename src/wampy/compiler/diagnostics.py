from wampy.compiler.compiled_program import CompiledProgram
from wampy.compiler.compiler_state import (
    ENTRY_INVALID_PC,
    NO_PREDICATE_SLOT,
    CompilerState,
)
from wampy.compiler.opcodes import OP
from wampy.compiler.predicates import find_predicate_slot
from wampy.frontend.symbol_table import SymbolTable

_OPCODE_NAMES = tuple(opcode.name for opcode in OP)


def opcode_trace(
    compiled_program: CompiledProgram,
    start: int = 0,
    stop: int | None = None,
) -> tuple[OP, ...]:
    """Return opcodes from a selected compiled-program range."""

    if stop is None:
        stop = compiled_program.code_size[0]

    emitted = compiled_program.code_size[0]
    if not 0 <= start <= stop <= emitted:
        raise ValueError(
            f"Invalid opcode trace range: start={start}, stop={stop}, emitted={emitted}"
        )

    return tuple(OP(int(opcode)) for opcode in compiled_program.code[start:stop, 0])


def predicate_opcode_trace(
    compiled_program: CompiledProgram,
    compiler_state: CompilerState,
    symbol_id: int,
    arity: int,
) -> tuple[OP, ...]:
    """Return the opcodes belonging to a raw-symbol predicate."""

    predicate_slot = find_predicate_slot(compiler_state, symbol_id)
    if predicate_slot == NO_PREDICATE_SLOT:
        return ()

    entry_pc = int(compiled_program.predicate_entry[predicate_slot, arity])

    if entry_pc == ENTRY_INVALID_PC:
        return ()

    traces: list[OP] = []
    current_pc = entry_pc
    visited: set[int] = set()

    while True:
        if current_pc in visited:
            raise ValueError(f"Cycle in predicate alternative chain at PC {current_pc}")

        visited.add(current_pc)
        prefix_op = OP(int(compiled_program.code[current_pc, 0]))

        if prefix_op in (OP.TRY_ME_ELSE, OP.RETRY_ME_ELSE):
            traces.append(prefix_op)
            clause_pc = current_pc + 1
            next_pc = int(compiled_program.code[current_pc, 1])

        elif prefix_op == OP.TRUST_ME_ELSE_FAIL:
            traces.append(prefix_op)
            clause_pc = current_pc + 1
            next_pc = None

        elif prefix_op == OP.JMP_RETRY:
            traces.append(prefix_op)
            clause_pc = int(compiled_program.code[current_pc, 1])
            next_pc = int(compiled_program.code[current_pc, 2])

        elif prefix_op == OP.JMP_TRUST:
            traces.append(prefix_op)
            clause_pc = int(compiled_program.code[current_pc, 1])
            next_pc = None

        else:
            clause_pc = current_pc
            next_pc = None

        for pc in range(clause_pc, compiled_program.code_size[0]):
            opcode = OP(int(compiled_program.code[pc, 0]))
            traces.append(opcode)

            if opcode == OP.PROCEED or opcode == OP.EXECUTE:
                break
        else:
            raise ValueError(f"Clause starting at PC {clause_pc} has no terminal instruction")

        if next_pc is None:
            break

        current_pc = next_pc

    return tuple(traces)


def render_compiler_state(
    compiled_program: CompiledProgram,
    compiler_state: CompilerState,
    symbol_table: SymbolTable,
) -> str:
    """Return predicate entry points and emitted WAM instructions as text."""

    lines = ["", "=== Predicate entry points ==="]
    for predicate_slot in range(int(compiler_state.predicate_count[0])):
        functor_id = int(compiler_state.symbol_by_predicate_slot[predicate_slot])
        for arity in range(compiled_program.predicate_entry.shape[1]):
            pc = int(compiled_program.predicate_entry[predicate_slot, arity])

            if pc == ENTRY_INVALID_PC:
                continue

            symbol_by_id = symbol_table.symbol_by_id
            if functor_id in symbol_by_id:
                symbol = symbol_by_id[functor_id]
            else:
                symbol = str(functor_id)
            lines.append(symbol + "/" + str(arity) + " -> PC " + str(pc))

    lines.append("")
    lines.append("=== WAM instructions ===")

    if compiled_program.code_size[0] == 0:
        lines.append("(no instructions)")
        return "\n".join(lines)

    for pc, instruction in enumerate(compiled_program.code[: compiled_program.code_size[0]]):
        opcode, a, b, c = instruction
        pc_text = str(pc)
        while len(pc_text) < 4:
            pc_text = "0" + pc_text

        opcode_text = _OPCODE_NAMES[int(opcode) - 1]
        while len(opcode_text) < 10:
            opcode_text += " "

        a_text = str(int(a))
        while len(a_text) < 3:
            a_text = " " + a_text

        b_text = str(int(b))
        while len(b_text) < 3:
            b_text = " " + b_text

        c_text = str(int(c))
        while len(c_text) < 3:
            c_text = " " + c_text

        lines.append(pc_text + ": " + opcode_text + " " + a_text + " " + b_text + " " + c_text)

    return "\n".join(lines)


def display_compiler_state(
    compiled_program: CompiledProgram,
    compiler_state: CompilerState,
    symbol_table: SymbolTable,
) -> None:
    """Print :func:`render_compiler_state` output."""

    print(render_compiler_state(compiled_program, compiler_state, symbol_table))
