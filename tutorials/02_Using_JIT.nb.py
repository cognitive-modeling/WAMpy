# %% [markdown]
# # 02 Using JIT
#
# WAMpy runs as normal Python by default.
# For performance-sensitive code, place an explicit Numba boundary around
# JIT-compatible functions from `wampy.api`.


# %%
# %pip install -q wampy-prolog

# %%
import inspect

import wampy.api as wam
from numba import jit


# %% [markdown]
# ## Build a Prolog program
#
# Parsing remains regular Python code.


# %%
source = """
parent(anakin, luke).
parent(padme, luke).
"""

symbols = wam.empty_symbol_table()
program, symbols = wam.parse(source, symbols)


# %% [markdown]
# ## Compile with Numba
#
# `compile_program` is an ordinary Python function registered as JIT-compatible.
# The notebook explicitly chooses the JIT boundary.


# %%
@jit
def compile_fast(program):
    compiler_state = wam.init_compiler_state()
    compiled_program = wam.init_compiled_program()
    wam.compile_program(program, compiled_program, compiler_state)
    return compiled_program


compiled = compile_fast(program)


# %% [markdown]
# The second call is much faster because Numba reuses the compiled specialization.

# %%
compiled = compile_fast(program)


# %% [markdown]
# The same compiler can also be called without JIT:

# %%
compiler_state_python = wam.init_compiler_state()
compiled_program_python = wam.init_compiled_program()

wam.compile_program(
    program,
    compiled_program_python,
    compiler_state_python,
)
compiled_python = compiled_program_python


# %% [markdown]
# ## Verify JIT compilation

# %%
print("compile is Python:", inspect.isfunction(wam.compile_program))
print("JIT signatures:", compile_fast.signatures)
print("Nopython signatures:", compile_fast.nopython_signatures)


# %% [markdown]
# ## Parse and run a query
#
# Query parsing also remains regular Python code. The parsed query can then be
# compiled and executed using the same compiler and runtime API.


# %%
query, symbols = wam.parse("?- parent(Father, luke).", symbols)
compiled_query = wam.init_compiled_query()
wam.compile_query(
    query,
    compiled_query,
    compiled_program_python,
    compiler_state_python,
)

machine = wam.init_machine()
status = wam.run(machine, compiled_program_python, compiled_query)

print("Query status:", wam.WAMStatus(status).name)

# %% [markdown]
# ## Inspect the current solution

# %%
solution = wam.decode_bindings(machine, compiled_query, symbols)
print(solution)

# %%
