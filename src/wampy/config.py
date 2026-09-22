"""Typed, immutable configuration for the WAMpy subsystems."""

import tomllib
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol, TypeVar, cast

import numpy as np


_SYMBOL_ID_DTYPES = {
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
}


class ASTConfig(NamedTuple):
    """Immutable AST storage configuration."""

    max_terms: int = 100
    max_nodes_per_term: int = 300
    child_block_size: int = 4
    max_variables_per_term: int = 32
    max_user_symbols: int = 256
    symbol_id_dtype: np.dtype = np.dtype(np.uint16)


class FrontendConfig(NamedTuple):
    ast: ASTConfig = ASTConfig()


class CompilerConfig(NamedTuple):
    max_instructions: int = 4_096
    max_hypothesis_predicate_changes: int = 16
    max_predicates: int = 256
    max_arity: int = 16
    max_goal_depth: int = 256
    max_structure_depth: int = 64


class RuntimeConfig(NamedTuple):
    trace: bool = False
    heap_size: int = 65_500
    trail_size: int = 8_192
    choice_point_size: int = 1_024
    environment_size: int = 8_192
    unify_stack_size: int = 8_192
    max_x_registers: int = 16
    max_structure_depth: int = 64

    max_steps: int = 10_000_000
    max_unify_steps: int = 10_000
    max_answers: int = 10
    max_answer_nodes: int = 512


class WAMConfig(NamedTuple):
    frontend: FrontendConfig = FrontendConfig()
    compiler: CompilerConfig = CompilerConfig()
    runtime: RuntimeConfig = RuntimeConfig()


def _is_namedtuple_instance(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_replace")


T = TypeVar(name="T")


class _NamedTupleLike(Protocol):
    _fields: tuple[str, ...]

    def _replace(self, **changes: Any) -> Any: ...


def _unwrap_wam_table(data: Mapping[str, Any]) -> Mapping[str, Any]:
    wam = data.get("wam")
    if wam is None:
        return data
    if not isinstance(wam, Mapping):
        raise ValueError("'wam' must be a table/mapping")
    return cast(Mapping[str, Any], wam)


def _merge_namedtuple[T](
    cfg: T,
    overrides: Mapping[str, Any],
) -> T:
    if not _is_namedtuple_instance(cfg):
        raise TypeError(f"Expected namedtuple instance, got {type(cfg)}")

    named_cfg = cast(_NamedTupleLike, cfg)

    for key, value in overrides.items():
        if isinstance(named_cfg, ASTConfig) and key == "max_children_per_node":
            if value <= 0:
                raise ValueError("frontend.ast.max_children_per_node must be > 0")
            cfg = cast(T, named_cfg._replace(max_children_per_node=value))
            named_cfg = cast(_NamedTupleLike, cfg)
            continue

        if key not in named_cfg._fields:
            raise ValueError(f"Unknown config key: {key}")

        current = getattr(named_cfg, key)
        if _is_namedtuple_instance(current):
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected table for '{key}', got {type(value)}")

            value = _merge_namedtuple(
                current,
                cast(Mapping[str, Any], value),
            )

        cfg = cast(T, named_cfg._replace(**{key: value}))
        named_cfg = cast(_NamedTupleLike, cfg)

    return cfg


def _validate_positive(section: str, config: Any, fields: tuple[str, ...]) -> None:
    for field in fields:
        if getattr(config, field) <= 0:
            raise ValueError(f"{section}.{field} must be > 0")


def resolve_symbol_id_dtype(value: Any) -> np.dtype:
    """Resolve and validate an AST symbol storage dtype.

    The resolver intentionally runs at the Python/JIT boundary.  Numba code
    receives an already allocated symbol array and never needs to interpret a
    configuration string as a NumPy dtype.
    """

    if isinstance(value, np.dtype):
        symbol_dtype = value
    else:
        try:
            symbol_dtype = _SYMBOL_ID_DTYPES[str(value)]
        except KeyError as error:
            allowed = ", ".join(_SYMBOL_ID_DTYPES)
            raise ValueError(
                f"Unsupported symbol ID dtype {value!r}; expected one of: {allowed}",
            ) from error

    if symbol_dtype.kind != "u":
        raise ValueError(f"Symbol ID dtype must be unsigned, got {symbol_dtype}")

    if symbol_dtype not in _SYMBOL_ID_DTYPES.values():
        allowed = ", ".join(_SYMBOL_ID_DTYPES)
        raise ValueError(
            f"Unsupported symbol ID dtype {symbol_dtype}; expected one of: {allowed}",
        )

    # Deliberately deferred: importing this module at config module scope
    # enters the eager frontend package initializers and creates an import cycle.
    from wampy.frontend.ast.symbol_id import CoreSymID  # pylint: disable=import-outside-toplevel

    max_symbol_id = int(np.iinfo(int_type=symbol_dtype.name).max)

    for reserved_symbol_id in CoreSymID:
        if int(reserved_symbol_id) > max_symbol_id:
            raise ValueError(
                f"frontend.ast.symbol_id_dtype is too small for SymID.{reserved_symbol_id.name}",
            )

    return symbol_dtype


def normalize_config(config: WAMConfig) -> WAMConfig:
    """Resolve any TOML dtype strings before JIT code sees the config."""

    frontend = config.frontend
    normalized_ast = frontend.ast._replace(
        symbol_id_dtype=resolve_symbol_id_dtype(frontend.ast.symbol_id_dtype),
    )
    return config._replace(
        frontend=frontend._replace(ast=normalized_ast),
    )


def _validate(config: WAMConfig) -> None:
    _validate_positive(
        "frontend.ast",
        config.frontend.ast,
        (
            "max_terms",
            "max_nodes_per_term",
            "child_block_size",
            "max_variables_per_term",
            "max_user_symbols",
        ),
    )
    symbol_dtype = resolve_symbol_id_dtype(config.frontend.ast.symbol_id_dtype)
    from wampy.frontend.ast.symbol_id import (  # pylint: disable=import-outside-toplevel
        calculate_symbol_id_positions,
    )

    _, first_user_symbol_id = calculate_symbol_id_positions(
        config.frontend.ast.max_variables_per_term,
    )
    user_symbol_id_limit = first_user_symbol_id + config.frontend.ast.max_user_symbols
    max_symbol_id = int(np.iinfo(symbol_dtype).max)
    if user_symbol_id_limit - 1 >= max_symbol_id:
        raise ValueError(
            "frontend.ast.symbol_id_dtype is too small",
            "for the configured variable and user symbol ranges",
        )
    _validate_positive(
        "compiler",
        config.compiler,
        (
            "max_instructions",
            "max_predicates",
            "max_arity",
            "max_goal_depth",
            "max_structure_depth",
        ),
    )
    _validate_positive(
        "runtime",
        config.runtime,
        (
            "heap_size",
            "trail_size",
            "choice_point_size",
            "environment_size",
            "unify_stack_size",
            "max_x_registers",
            "max_structure_depth",
            "max_steps",
            "max_unify_steps",
            "max_answers",
            "max_answer_nodes",
        ),
    )
    if config.compiler.max_arity > config.runtime.max_x_registers:
        raise ValueError("compiler.max_arity must not exceed runtime.max_x_registers")


def load_config_toml(toml_text: str = "", base: WAMConfig | None = None) -> WAMConfig:
    """Load a WAMConfig from TOML, recursively merging it into ``base``."""
    base_config = WAMConfig() if base is None else base

    if not toml_text.strip():
        base_config = normalize_config(base_config)
        _validate(base_config)
        return base_config

    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML: {error}") from error

    config = normalize_config(_merge_namedtuple(base_config, _unwrap_wam_table(data)))
    _validate(config)
    return config


def apply_overrides(base: WAMConfig | None, overrides: Mapping[str, Any]) -> WAMConfig:
    """Apply nested mapping overrides to a WAMConfig."""
    base_config = WAMConfig() if base is None else base
    config = normalize_config(_merge_namedtuple(base_config, overrides))
    _validate(config)
    return config


def debug_config(config: Any) -> str:
    """Render a nested NamedTuple or mapping as TOML."""

    def toml_value(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError("TOML does not support NaN/Inf")
            return repr(value)
        if isinstance(value, np.dtype):
            return toml_value(value.name)
        if isinstance(value, str):
            escaped = (
                value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            return f'"{escaped}"'
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(toml_value(item) for item in value) + "]"
        raise TypeError(f"Unsupported TOML value type: {type(value)}")

    def is_table(value: Any) -> bool:
        return _is_namedtuple_instance(value) or isinstance(value, Mapping)

    def iter_items(value: Any):
        if _is_namedtuple_instance(value) and hasattr(value, "_asdict"):
            return value._asdict().items()
        if isinstance(value, Mapping):
            return value.items()
        raise TypeError(f"Expected NamedTuple/Mapping, got {type(value)}")

    lines: list[str] = []

    def emit(value: Any, prefix: str = "", header: bool = True) -> None:
        scalars: list[tuple[str, Any]] = []
        children: list[tuple[str, Any]] = []

        for key, item in iter_items(value):
            if is_table(item):
                children.append((key, item))
            else:
                scalars.append((key, item))

        if header and prefix:
            lines.append(f"[{prefix}]")

        for key, item in scalars:
            lines.append(f"{key} = {toml_value(value=item)}")

        if (header and prefix) or scalars:
            lines.append("")

        for key, child in children:
            child_prefix = f"{prefix}.{key}" if prefix else key
            emit(child, child_prefix, header=True)

    emit(config, prefix="", header=False)
    if lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


DEFAULT_CONFIG: WAMConfig = load_config_toml(toml_text="")
