"""Lexical and syntactic helpers for the supported Prolog subset."""

import re

SYMBOL_RE: re.Pattern[str] = re.compile(
    pattern=r"""
    '([^'\n\r]*)'                         |  # quoted atom (no newlines inside)
    (?<![A-Za-z0-9_])([a-z][A-Za-z0-9_]*)  # unquoted atom, not part of a larger token
    (?![A-Za-z0-9_])
    """,
    flags=re.VERBOSE,
)


def _split_clauses(source: str) -> list[str]:
    source = _strip_comments(source)
    clauses = []
    buf = ""
    depth = 0
    in_quote = False

    for char in source:
        if char == "'":
            in_quote = not in_quote

        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

        if char == "." and depth == 0 and not in_quote:
            clause = buf.strip()
            if clause:
                clauses.append(clause + ".")
            buf = ""
        else:
            buf += char

    # Accept a final clause without a trailing dot.
    tail = buf.strip()
    if tail:
        clauses.append(tail if tail.endswith(".") else tail + ".")

    return clauses


def _strip_comments(source: str) -> str:
    """Strip Prolog comments outside quoted atoms while preserving newlines."""
    out = ""
    i = 0
    in_quote = False

    while i < len(source):
        char = source[i]

        if char == "'":
            in_quote = not in_quote
            out += char
            i += 1
            continue

        if not in_quote and char == "%":
            while i < len(source) and source[i] not in "\r\n":
                i += 1
            continue

        if not in_quote and i + 1 < len(source) and source[i : i + 2] == "/*":
            i += 2
            while i + 1 < len(source) and source[i : i + 2] != "*/":
                if source[i] in "\r\n":
                    out += source[i]
                i += 1
            if i + 1 < len(source):
                i += 2
            out += " "
            continue

        out += char
        i += 1

    return out


def _strip_block_comments(source: str) -> str:
    """Strip block comments while preserving quoted comment markers."""
    out = ""
    i = 0
    in_quote = False

    while i < len(source):
        char = source[i]
        if char == "'":
            in_quote = not in_quote
            out += char
            i += 1
            continue

        if not in_quote and i + 1 < len(source) and source[i : i + 2] == "/*":
            i += 2
            while i + 1 < len(source) and source[i : i + 2] != "*/":
                if source[i] == "\n":
                    out += "\n"
                i += 1
            if i + 1 < len(source):
                i += 2
            out += " "
            continue

        out += char
        i += 1

    return out


def _split_clause(source: str) -> tuple[str, str | None]:
    """Split a fact or rule into its head and optional body."""
    source = source.strip()
    if source.endswith("."):
        source = source[:-1]

    depth = 0
    in_quote = False
    i = 0
    while i < len(source) - 1:
        char = source[i]

        if char == "'":
            in_quote = not in_quote

        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and source[i : i + 2] == ":-":
                return source[:i].strip(), source[i + 2 :].strip()

        i += 1

    return source, None


def _split_goals(source: str) -> list[str]:
    """Split a clause body on top-level, unquoted commas."""
    goals = []
    depth = 0
    in_quote = False
    buf = ""

    for char in source:
        if char == "'":
            in_quote = not in_quote

        if char == "," and depth == 0 and not in_quote:
            goals.append(buf.strip())
            buf = ""
            continue

        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

        buf += char

    if buf.strip():
        goals.append(buf.strip())

    return goals


def is_prolog_var(token: str) -> bool:
    if not token:
        return False

    return token[0].isupper() or token[0] == "_"


def _parse_term(source: str) -> tuple[str, list[str]]:
    """Parse an atom, compound term, or comma-group shorthand."""
    source = source.strip()

    if source == "!":
        return "!", []

    not_operand = _parse_not_alias(source)
    if not_operand is not None:
        return "\\+", [not_operand]

    if source.startswith("\\+"):
        operand = source[2:].strip()
        if not operand:
            raise ValueError("Negation requires an operand")

        if operand.startswith("("):
            inner = _strip_outer_parentheses(operand)
            if inner is None:
                raise ValueError("Negation in parenthesized form must be exactly '\\\\+(Goal)'")
            inner = inner.strip()
            if _has_top_level_comma(inner):
                operand = f"({inner})"
            else:
                operand = inner

        if not operand:
            raise ValueError("Negation requires an operand")

        return "\\+", [operand]

    inner = _strip_outer_parentheses(source)
    if inner is not None:
        if _has_top_level_comma(inner):
            args = _split_top_level_terms(inner)
            if len(args) < 2:
                raise ValueError("Conjunction requires at least two goals")
            if len(args) == 2:
                return ",", args

            remainder = f"({', '.join(args[1:])})"
            return ",", [args[0], remainder]
        return _parse_term(inner)

    if "(" not in source:
        return source, []

    functor, rest = source.split("(", 1)
    return functor.strip(), _split_top_level_terms(rest[:-1])


def _split_top_level_terms(source: str) -> list[str]:
    terms = []
    depth = 0
    in_quote = False
    buf = ""

    for char in source:
        if char == "'" and not in_quote:
            in_quote = True
        elif char == "'" and in_quote:
            in_quote = False

        if char == "," and depth == 0 and not in_quote:
            terms.append(buf.strip())
            buf = ""
            continue

        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

        buf += char

    if buf:
        terms.append(buf.strip())

    return terms


def _parse_not_alias(source: str) -> str | None:
    alias = _negation_alias_prefix(source)
    if alias is None:
        return None

    rest = source[len(alias) :].lstrip()
    if not rest.startswith("("):
        return None

    operand = _strip_outer_parentheses(rest)
    if operand is None:
        raise ValueError("Negation alias must be exactly 'nfa(Goal)' or 'naf(Goal)' or 'not(Goal)'")

    operand = operand.strip()
    if not operand:
        raise ValueError("Negation requires an operand")
    if _has_top_level_comma(operand):
        raise ValueError("Grouped negation is not supported yet; use a predicate goal.")

    return operand


def _negation_alias_prefix(source: str) -> str | None:
    for alias in ("nfa", "naf", "not"):
        if not source.startswith(alias):
            continue

        if len(source) == len(alias):
            return alias

        next_char = source[len(alias)]
        if next_char == "(" or next_char.isspace():
            return alias

    return None


def _strip_outer_parentheses(source: str) -> str | None:
    source = source.strip()
    if len(source) < 2 or source[0] != "(" or source[-1] != ")":
        return None

    depth = 0
    in_quote = False
    for i, char in enumerate(source):
        if char == "'":
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and i != len(source) - 1:
                return None

    if in_quote or depth != 0:
        return None

    return source[1:-1]


def _has_top_level_comma(source: str) -> bool:
    depth = 0
    in_quote = False
    for char in source:
        if char == "'":
            in_quote = not in_quote
            continue

        if in_quote:
            continue

        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return True

    return False
