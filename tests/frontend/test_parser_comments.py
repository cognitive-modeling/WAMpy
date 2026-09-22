import wampy.frontend.parser as parser


def test_split_clauses_ignores_percent_line_comments():
    source = r"""
    % --- People ---
    male(anakin).
    % another comment with punctuation . ()
    female(leia).
    """

    assert parser._split_clauses(source) == ["male(anakin).", "female(leia)."]


def test_split_clauses_ignores_inline_percent_comment():
    source = "p(a). % comment with a dot .\nq(b)."

    assert parser._split_clauses(source) == ["p(a).", "q(b)."]


def test_split_clauses_preserves_percent_inside_quoted_atom():
    source = "note('100% real'). % trailing comment\nq(b)."

    assert parser._split_clauses(source) == ["note('100% real').", "q(b)."]


def test_split_clauses_handles_line_and_block_comments_together():
    source = "% heading\np(a). /* block % comment */ q(b)."

    assert parser._split_clauses(source) == ["p(a).", "q(b)."]
