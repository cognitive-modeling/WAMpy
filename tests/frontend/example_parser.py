# core/example_loader.py

# from wampy.frontend.parser import parse


# def example_add_binding():
#     from models.rational_program.core.grammar import initiate_base_grammar, create_lookup_table
#     from models.rational_program.core.program import init_program, debug_ascii_tree
#     from models.rational_program.core.regeneration import subtree_regeneration

#     global NUM_TERMS, MAX_TERMS_NODES

#     NUM_TERMS = 4
#     MAX_TERMS_NODES = 6

#     program: Program = init_program()

#     print("Initial program state:")
#     print(program.symbol)
#     debug_ascii_tree(program.children, program.symbol, term_id=0)
#     print()

#     # ------------------------------------------------------------
#     print("Add first binding")
#     cid1 = add_binding(program)
#     print("Returned concept id:", cid1)
#     print(program.symbol)
#     debug_ascii_tree(program.children, program.symbol, term_id=cid1)
#     print()

#     # ------------------------------------------------------------
#     print("Add second binding")
#     cid2 = add_binding(program)
#     print("Returned concept id:", cid2)
#     print(program.symbol)
#     debug_ascii_tree(program.children, program.symbol, term_id=cid2)
#     print()

#     # ------------------------------------------------------------
#     print("Add third binding")
#     cid3 = add_binding(program)
#     print("Returned concept id:", cid3)
#     print(program.symbol)
#     debug_ascii_tree(program.children, program.symbol, term_id=cid3)

#     # ------------------------------------------------------------
#     print("Resample (Subtree regeneration)")
#     print(program.symbol)

#     grammar = initiate_base_grammar()
#     grammar_lookup = create_lookup_table(grammar)
#     old_term_id, old_term, is_ok = subtree_regeneration(grammar, grammar_lookup, program)
#     print("Returned concept id:", old_term_id)
#     print(program.symbol)
#     debug_ascii_tree(program.children, program.symbol, term_id=old_term_id)


def _example():
    source = """
    parent(john, mary).
    parent(mary, alice).
    add('1', '+', '2', '=', '3').
    add(1).
    ancestor(X, Y) :- parent(X, Y).
    ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
    """

    program, binding = parse(source)

    print("=== Program Loaded ===")
    print(program.symbol.shape)

    print("\n=== Symbol Binding Table ===")
    for addr, sym in binding.items():
        print(f"{addr} -> {sym}")

    # Minimal invariants (not full tests)
    assert isinstance(binding, dict)
    assert len(binding) > 0

    print("\nLoader test completed successfully.")


if __name__ == "__main__":
    _example()
