"""
# wam/_config.py
DEFAULT_WAM_CONFIG = WAMConfig(
    STACK_HEAP_SIZE=131.072,
    STACK_TRAIL_SIZE=16384,
    STACK_CP_SIZE=4.036,
    STACK_UNIFY_STACK_SIZE=16384,
    ENTRY_MAX_FUNCTORS=4.036,
    ENTRY_MAX_ARITY=16,
    SOLVER_ENABLE_OCCURS_CHECK=False,
    SOLVER_MAX_STEPS=20_000,
    MAX_X=256,
)
"""

from typing import Any, Mapping, NamedTuple, TypeVar, cast
import tomllib


# --- typed namedtuple configs ---
class ASTreeConfig(NamedTuple):
    num_terms: int = 100
    max_terms_nodes: int = 300  # max amount of concept nodes
    max_terms_nodes_childs: int = 5


class StackConfig(NamedTuple):
    heap_size: int = 65_500
    trail_size: int = 8_192
    cp_size: int = 1_024
    unify_stack_size: int = 8_192


class EntryConfig(NamedTuple):
    max_functors: int = 256
    max_arity: int = 4


class SolverConfig(NamedTuple):
    enable_occurs_check: bool = False
    max_steps: int = 10_000
    max_unify_steps: int = 10_000
    answer_max_answers: int = 10
    answer_max_nodes: int = 512
    trace: bool = False



class LimitsConfig(NamedTuple):
    max_x: int = 16
    max_instr: int = 4_096
    max_struct: int = 64


class WAMConfig(NamedTuple):
    ast: ASTreeConfig = ASTreeConfig()
    stack: StackConfig = StackConfig()
    entry: EntryConfig = EntryConfig()
    solver: SolverConfig = SolverConfig()
    limits: LimitsConfig = LimitsConfig()


def _is_namedtuple_instance(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_replace")


T = TypeVar("T")


def _unwrap_wam_table(data: Mapping[str, Any]) -> Mapping[str, Any]:
    wam = data.get("wam")
    if wam is None:
        return data
    if not isinstance(wam, Mapping):
        raise ValueError("'wam' must be a table/mapping")
    return cast(Mapping[str, Any], wam)


def _merge_namedtuple(cfg: T, overrides: Mapping[str, Any]) -> T:
    if not _is_namedtuple_instance(cfg):
        raise TypeError(f"Expected namedtuple instance, got {type(cfg)}")

    for k, v in overrides.items():
        if k not in cfg._fields:  # type: ignore[attr-defined]
            raise ValueError(f"Unknown config key: {k}")

        cur = getattr(cfg, k)
        if _is_namedtuple_instance(cur):
            if not isinstance(v, Mapping):
                raise TypeError(f"Expected table for '{k}', got {type(v)}")
            cfg = cast(Any, cfg)._replace(**{k: _merge_namedtuple(cur, cast(Mapping[str, Any], v))})
        else:
            cfg = cast(Any, cfg)._replace(**{k: v})

    return cfg


def _validate(cfg: WAMConfig) -> None:
    if cfg.stack.heap_size <= 0:
        raise ValueError("stack.heap_size must be > 0")
    if cfg.stack.trail_size <= 0:
        raise ValueError("stack.trail_size must be > 0")
    if cfg.stack.cp_size <= 0:
        raise ValueError("stack.cp_size must be > 0")
    if cfg.stack.unify_stack_size <= 0:
        raise ValueError("stack.unify_stack_size must be > 0")

    if cfg.entry.max_functors <= 0:
        raise ValueError("entry.max_functors must be > 0")
    if cfg.entry.max_arity <= 0:
        raise ValueError("entry.max_arity must be > 0")

    if cfg.solver.max_steps <= 0:
        raise ValueError("solver.max_steps must be > 0")
    if cfg.solver.max_unify_steps <= 0:
        raise ValueError("solver.max_unify_steps must be > 0")

    if cfg.limits.max_x <= 0:
        raise ValueError("limits.max_x must be > 0")
    if cfg.limits.max_instr <= 0:
        raise ValueError("limits.max_instr must be > 0")
    if cfg.limits.max_struct <= 0:
        raise ValueError("limits.max_struct must be > 0")


def load_config_toml(toml_text: str = "", base: WAMConfig | None = None) -> WAMConfig:
    base_cfg = WAMConfig() if base is None else base

    if not toml_text.strip():
        _validate(base_cfg)
        return base_cfg

    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML: {e}") from e

    data = _unwrap_wam_table(data)
    cfg = _merge_namedtuple(base_cfg, data)
    _validate(cfg)
    return cfg


def apply_overrides(base: WAMConfig | None, overrides: Mapping[str, Any]) -> WAMConfig:
    base_cfg = WAMConfig() if base is None else base
    cfg = _merge_namedtuple(base_cfg, overrides)
    _validate(cfg)
    return cfg


def debug_config(cfg: Any) -> str:
    """
    Dynamically render (nested) NamedTuple config as TOML.
    - NamedTuple fields become TOML tables.
    - Scalars become key/value pairs.
    - Nested NamedTuples become nested tables ([a.b]).
    """

    def toml_value(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            # basic (finite) float support
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError("TOML does not support NaN/Inf")
            return repr(v)
        if isinstance(v, str):
            esc = (
                v.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            return f'"{esc}"'
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(toml_value(x) for x in v) + "]"
        raise TypeError(f"Unsupported TOML value type: {type(v)}")

    def is_table(x: Any) -> bool:
        return _is_namedtuple_instance(x) or isinstance(x, Mapping)

    def iter_items(obj: Any):
        if _is_namedtuple_instance(obj) and hasattr(obj, "_asdict"):
            return obj._asdict().items()
        if isinstance(obj, Mapping):
            return obj.items()
        raise TypeError(f"Expected NamedTuple/Mapping, got {type(obj)}")

    lines: list[str] = []

    def emit(obj: Any, prefix: str = "", header: bool = True) -> None:
        scalars: list[tuple[str, Any]] = []
        children: list[tuple[str, Any]] = []

        for k, v in iter_items(obj):
            if is_table(v):
                children.append((k, v))
            else:
                scalars.append((k, v))

        if header and prefix:
            lines.append(f"[{prefix}]")

        for k, v in scalars:
            lines.append(f"{k} = {toml_value(v)}")

        # blank line after a rendered block (table or root scalars), but not before first table
        if (header and prefix) or scalars:
            lines.append("")

        for k, child in children:
            child_prefix = f"{prefix}.{k}" if prefix else k
            emit(child, child_prefix, header=True)

    emit(cfg, prefix="", header=False)

    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


DEFAULT_CONFIG = load_config_toml("")


if __name__ == "__main__":
    # defaults reflect your old DEFAULT_CONFIG
    cfg0 = load_config_toml()
    print(debug_config(cfg0))

    from numba import jit

    @jit
    def f(cfg):
        return cfg.stack.heap_size + cfg.entry.max_arity

    print(f(cfg0))

    override_toml = """
    [stack]
    heap_size = 131072

    [entry]
    max_arity = 16

    [solver]
    enable_occurs_check = true
    max_steps = 50000
    max_unify_steps = 5000

    [limits]
    max_x = 512
    """
    cfg1 = load_config_toml(override_toml)
    print("-----")
    print(debug_config(cfg1))

    print("-----")
    print(debug_config(DEFAULT_CONFIG))
