import numpy as np

from wampy.frontend.answers import prefix_equal, any_correct_match

MAX = 8  # small "heap" for tests
DT = np.dtype(
    [
        ("len", np.int32),
        ("tags", np.int32, (MAX,)),
        ("symbols", np.int32, (MAX,)),
        ("aritys", np.int32, (MAX,)),
    ]
)


def mk_prefix(L, tags, symbols, aritys):
    """Create one structured record compatible with prefix_equal/any_correct_match."""
    rec = np.zeros(1, dtype=DT)[0]
    rec["len"] = L

    rec["tags"][:] = -999
    rec["symbols"][:] = -999
    rec["aritys"][:] = -999

    rec["tags"][:L] = np.array(tags, dtype=np.int32)
    rec["symbols"][:L] = np.array(symbols, dtype=np.int32)
    rec["aritys"][:L] = np.array(aritys, dtype=np.int32)
    return rec


def mk_arr(*records):
    arr = np.zeros(len(records), dtype=DT)
    for i, r in enumerate(records):
        arr[i] = r
    return arr


def test_exact_match_true():
    found = mk_prefix(3, [1, 2, 3], [10, 20, 30], [2, 1, 0])
    correct = mk_arr(mk_prefix(3, [1, 2, 3], [10, 20, 30], [2, 1, 0]))
    assert prefix_equal(found, correct[0]) is True
    assert any_correct_match(found, correct) is True


def test_no_match_false():
    found = mk_prefix(3, [1, 2, 3], [10, 20, 30], [2, 1, 0])
    correct = mk_arr(mk_prefix(3, [1, 9, 3], [10, 20, 30], [2, 1, 0]))  # tag differs
    assert prefix_equal(found, correct[0]) is False
    assert any_correct_match(found, correct) is False


def test_length_mismatch_false():
    found = mk_prefix(2, [1, 2], [10, 20], [0, 0])
    correct = mk_arr(mk_prefix(3, [1, 2, 0], [10, 20, 0], [0, 0, 0]))
    assert any_correct_match(found, correct) is False


def test_symbol_mismatch_false():
    found = mk_prefix(3, [1, 2, 3], [10, 20, 30], [0, 0, 0])
    correct = mk_arr(mk_prefix(3, [1, 2, 3], [10, 99, 30], [0, 0, 0]))
    assert any_correct_match(found, correct) is False


def test_arity_mismatch_false():
    found = mk_prefix(3, [1, 2, 3], [10, 20, 30], [0, 0, 0])
    correct = mk_arr(mk_prefix(3, [1, 2, 3], [10, 20, 30], [0, 7, 0]))
    assert any_correct_match(found, correct) is False


def test_multiple_correct_answers_match_second_true():
    found = mk_prefix(3, [1, 2, 3], [10, 20, 30], [0, 0, 0])
    correct = mk_arr(
        mk_prefix(3, [9, 9, 9], [10, 20, 30], [0, 0, 0]),
        mk_prefix(3, [1, 2, 3], [10, 20, 30], [0, 0, 0]),
    )
    assert any_correct_match(found, correct) is True


def test_empty_correct_answers_false():
    found = mk_prefix(1, [1], [1], [1])
    correct = np.zeros(0, dtype=DT)
    assert any_correct_match(found, correct) is False


def test_tail_ignored_beyond_len_true():
    # Important: prefix_equal only compares indices < len.
    a = mk_prefix(2, [1, 2], [10, 20], [0, 0])
    b = mk_prefix(2, [1, 2], [10, 20], [0, 0])

    # Make them differ *after* len; should still be equal.
    a["tags"][3] = 111
    b["tags"][3] = 222
    assert prefix_equal(a, b) is True


def test_build_correct_answers_zero_arity_constant_is_str_root():
    from wampy.frontend.answers import build_correct_prefix_answers
    from wampy.frontend.symbol_table import symbol_table_from_mapping
    from wampy.runtime.stack import TAG

    symbol_table = symbol_table_from_mapping({0: "1"})
    answers = build_correct_prefix_answers([["'1'."]], symbol_table, max_answer_heap=8)

    rec = answers[0][0]
    assert int(rec["len"]) == 1
    assert int(rec["tags"][0]) == TAG.STR
    assert int(rec["aritys"][0]) == 0
    assert int(rec["symbols"][0]) == int(symbol_table.id_by_symbol["1"])
