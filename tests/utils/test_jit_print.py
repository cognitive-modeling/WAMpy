import numpy as np
import pytest
from numba import jit

from wampy.utils.jit_print import (
    flush_stdout_jit,
    print_dot_jit,
    print_inline_jit,
    print_newline_jit,
    print_progress_complete_jit,
    update_progress_bar_jit,
)

pytestmark = pytest.mark.requires_numba_jit


@jit
def _emit_inline_progress():
    print_dot_jit()
    print_dot_jit()
    print_inline_jit("x")
    print_newline_jit()
    flush_stdout_jit()


def test_jit_print_inline_without_newline(capsys):
    _emit_inline_progress()
    out = capsys.readouterr().out
    assert out == "..x\n"
    assert _emit_inline_progress.nopython_signatures


@jit
def _emit_progress_bar_with_elapsed():
    printed_percent = np.int64(-1)
    total_iterations = np.int64(10)
    bar_width = np.int64(10)
    started_s = np.float64(0.0)
    printed_percent = update_progress_bar_jit(np.int64(0), total_iterations, printed_percent, bar_width, started_s)
    printed_percent = update_progress_bar_jit(np.int64(5), total_iterations, printed_percent, bar_width, started_s)
    print_progress_complete_jit(total_iterations, np.int64(5 * 3600 + 10 * 60 + 10), bar_width)


def test_jit_progress_bar_and_elapsed(capsys):
    _emit_progress_bar_with_elapsed()
    out = capsys.readouterr().out
    assert "\rlearn   0%|          |  0/10 [" in out
    assert "\rlearn  50%|█████     |  5/10 [" in out
    assert "it/s]" in out
    assert "\rlearn 100%|██████████| 10/10 [5:10:10<00:00, 0.00it/s]" in out
    assert out.endswith("\n")
    assert _emit_progress_bar_with_elapsed.nopython_signatures
