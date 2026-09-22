# %% [markdown]
# # Demo WAMpy
#
# This demo uses the high-level `Prolog` API for normal query execution.
# The lower-level parser/compiler API is shown separately for onto-compilation
# and AST inspection.

# %%
# %pip install -q wampy-prolog

# %%
import wampy as wam
from IPython.display import display


# %% [markdown]
# ## High-level Prolog API

# %%

source = r"""
male(anakin).
male(luke).
female(lea)
"""

prolog = wam.Prolog(source)

# Return all solutions as independent binding dictionaries.
display(prolog.query_all("male(X)."))

# %% [markdown]
# ### Lazy query iteration
#
# `Prolog.query(...)` returns a lazy iterator. Each call advances WAM
# backtracking by one solution.

# %%
query = prolog.query("male(X).")

next(query)

# %%
query.next()

# %%
query.next()

# %% [markdown]
# ### Prepared queries
#
# A prepared query can be parsed once and executed repeatedly by the same
# `Prolog` instance.

# %%
prepared = prolog.prepare("male(X).")

first_run = prolog.query_all(prepared)
second_run = prolog.query_all(prepared)

display(
    {
        "first_run": first_run,
        "second_run": second_run,
    }
)

# %% [markdown]
# ### Inspect compiled WAM code

# %%
print(prolog.code)
