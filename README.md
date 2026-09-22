# WAMpy

> [!CAUTION]
> **Experimental research software — ALPHA**
>
> WAMpy is research software under active development.
> It is not a stable implementation or interface.
> Syntax, APIs, semantics, behavior, and data structures may change without notice and without preserving backward compatibility.
>
> **WAMpy comes with absolutely no warranty.** The software is provided "as is", without warranty of any kind, express or implied.
> In particular, there is no guarantee that it is correct, complete, reliable, suitable for any particular purpose, or compatible with future releases.
> If you use WAMpy, you do so entirely at your own risk.


## Installation

```console
pip install wampy-prolog
```


## Tutorials

- 01 Introduction [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cognitive-modeling/WAMpy/blob/tutorials/01_Introduction.ipynb)
- 02 Using JIT [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cognitive-modeling/WAMpy/blob/tutorials/02_using_jit.ipynb)


### Usage.

WAMpy is used as a Python library rather than as a standalone Prolog interpreter.

```
import wampy as wam
prolog = wam.Prolog("""
    parent(anakin, luke).
    parent(padme, luke).
""")

list(prolog.query("parent(X, luke)."))
```

For stateful backtracking, use ``Prolog`` and keep the returned query iterator:

```python
from wampy import Prolog

prolog = Prolog("""
    parent(anakin, luke).
    parent(padme, luke).
""")

query = prolog.query("parent(X, luke).")
next(query)  # {"X": "anakin"}
query.next()  # {"X": "padme"}

print(prolog.ast)
print(prolog.code)
```

Each query owns its execution state, so multiple queries can be interleaved
safely. ``list(prolog.query(...))``, ``query.close()``, ``query_once()``,
``exists()``, and ``query_batch()`` are also available.
Batch execution is
currently serial; passing ``parallel=True`` raises ``NotImplementedError``.

The views also render automatically in IPython.
Lower-level inspection can use
``wampy.frontend.render.render()`` and
``wampy.compiler.diagnostics.render_compiler_state()`` to obtain strings.


## Limitations

WAMpy implements a restricted Prolog and Warren Abstract Machine subset:

- Environment frames hold the previous E, saved CP, and permanent Y variables;
  the compiler emits ALLOCATE/DEALLOCATE around returning calls.
- Lists and native list operations are not supported.
- User-level cut (`!`) is supported, including neck cuts and cuts after calls.
- Disjunction (;) is not supported.
- Clause indexing is not implemented.
- Arithmetic and most standard Prolog built-in predicates are not provided.
- Meta-calls and the general SWI-Prolog built-in library are not supported.
- Execution is bounded by configured stack, answer, unification, and step limits.


## Contributors

- [Dominik Magiera](https://github.com/d-magiera)
- [Lukas Röhrig](https://github.com/LR2244)
