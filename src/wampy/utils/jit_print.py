import numpy as np
from numba import float64, int64, jit, types, void
from numba.core import cgutils, typing
from numba.core.imputils import impl_ret_untracked, lower_builtin
from numba.core.typing.templates import AbstractTemplate, infer_global, signature

__all__ = [
    "print_inline_jit",
    "print_dot_jit",
    "print_newline_jit",
    "flush_stdout_jit",
    "update_progress_bar_jit",
    "print_progress_complete_jit",
]


def print_inline_jit(text) -> None:
    """
    Print text without appending a newline from JIT-compiled code.
    """
    raise RuntimeError("print_inline_jit is only supported inside @jit-compiled functions")


@infer_global(print_inline_jit)
class _PrintInlineTyper(AbstractTemplate):
    def generic(self, args, kws):
        if kws:
            return
        if len(args) != 1:
            return
        return signature(types.none, args[0])


@lower_builtin(print_inline_jit, types.Any)
def _lower_print_inline_jit(context, builder, sig, args):
    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    print_sig = typing.signature(types.none, sig.args[0])
    imp = context.get_function("print_item", print_sig)
    imp(builder, [args[0]])

    pyapi.gil_release(gil)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


def flush_stdout_jit() -> None:
    """
    Flush sys.stdout from JIT-compiled code.
    """
    raise RuntimeError("flush_stdout_jit is only supported inside @jit-compiled functions")


def perf_counter_seconds_jit() -> np.float64:
    """
    Return time.perf_counter() from JIT-compiled code.
    """
    raise RuntimeError("perf_counter_seconds_jit is only supported inside @jit-compiled functions")


@infer_global(flush_stdout_jit)
class _FlushStdoutTyper(AbstractTemplate):
    def generic(self, args, kws):
        if kws:
            return
        if len(args) != 0:
            return
        return signature(types.none)


@infer_global(perf_counter_seconds_jit)
class _PerfCounterTyper(AbstractTemplate):
    def generic(self, args, kws):
        if kws:
            return
        if len(args) != 0:
            return
        return signature(types.float64)


@lower_builtin(flush_stdout_jit)
def _lower_flush_stdout_jit(context, builder, sig, args):
    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    sys_mod_name = context.insert_const_string(builder.module, "sys")
    sys_mod = pyapi.import_module(sys_mod_name)
    with cgutils.if_likely(builder, cgutils.is_not_null(builder, sys_mod)):
        stdout_obj = pyapi.object_getattr_string(sys_mod, "stdout")
        with cgutils.if_likely(builder, cgutils.is_not_null(builder, stdout_obj)):
            flush_res = pyapi.call_method(stdout_obj, "flush")
            with builder.if_then(cgutils.is_not_null(builder, flush_res)):
                pyapi.decref(flush_res)
            with builder.if_then(cgutils.is_null(builder, flush_res)):
                pyapi.err_clear()
            pyapi.decref(stdout_obj)
        with builder.if_then(cgutils.is_null(builder, stdout_obj)):
            pyapi.err_clear()
        pyapi.decref(sys_mod)

    pyapi.gil_release(gil)
    res = context.get_dummy_value()
    return impl_ret_untracked(context, builder, sig.return_type, res)


@lower_builtin(perf_counter_seconds_jit)
def _lower_perf_counter_seconds_jit(context, builder, sig, args):
    pyapi = context.get_python_api(builder)
    gil = pyapi.gil_ensure()

    out = cgutils.alloca_once_value(builder, context.get_constant(types.float64, 0.0))

    time_mod_name = context.insert_const_string(builder.module, "time")
    time_mod = pyapi.import_module(time_mod_name)
    with cgutils.if_likely(builder, cgutils.is_not_null(builder, time_mod)):
        perf_fn = pyapi.object_getattr_string(time_mod, "perf_counter")
        with cgutils.if_likely(builder, cgutils.is_not_null(builder, perf_fn)):
            perf_res = pyapi.call_function_objargs(perf_fn, ())
            with cgutils.if_likely(builder, cgutils.is_not_null(builder, perf_res)):
                perf_float = pyapi.number_float(perf_res)
                with cgutils.if_likely(builder, cgutils.is_not_null(builder, perf_float)):
                    perf_value = pyapi.float_as_double(perf_float)
                    builder.store(perf_value, out)
                    pyapi.decref(perf_float)
                with builder.if_then(cgutils.is_null(builder, perf_float)):
                    pyapi.err_clear()
                pyapi.decref(perf_res)
            with builder.if_then(cgutils.is_null(builder, perf_res)):
                pyapi.err_clear()
            pyapi.decref(perf_fn)
        with builder.if_then(cgutils.is_null(builder, perf_fn)):
            pyapi.err_clear()
        pyapi.decref(time_mod)
    with builder.if_then(cgutils.is_null(builder, time_mod)):
        pyapi.err_clear()

    pyapi.gil_release(gil)
    return impl_ret_untracked(context, builder, sig.return_type, builder.load(out))


@jit(void(), cache=True)
def print_dot_jit() -> None:
    print_inline_jit(".")


@jit(void(), cache=True)
def print_newline_jit() -> None:
    print_inline_jit("\n")
    flush_stdout_jit()


@jit(int64(int64, int64), cache=True)
def _percent_from_progress(done_iterations: np.int64, total_iterations: np.int64) -> np.int64:
    if total_iterations <= 0:
        return np.int64(100)
    if done_iterations <= 0:
        return np.int64(0)

    percent = (done_iterations * np.int64(100)) // total_iterations
    if percent > 100:
        return np.int64(100)
    return percent


@jit(int64(int64), cache=True)
def _digits_i64(value: np.int64) -> np.int64:
    if value <= 0:
        return np.int64(1)
    digits = np.int64(0)
    current = value
    while current > 0:
        digits += np.int64(1)
        current //= np.int64(10)
    return digits


@jit(void(int64, int64), cache=True)
def _print_i64_padded(value: np.int64, width: np.int64) -> None:
    digits = _digits_i64(value)
    spaces = width - digits
    if spaces < 0:
        spaces = np.int64(0)
    for _ in range(spaces):
        print_inline_jit(" ")
    print_inline_jit(value)


@jit(void(int64), cache=True)
def _print_2digit_jit(value: np.int64) -> None:
    if value < 10:
        print_inline_jit("0")
    print_inline_jit(value)


@jit(void(int64), cache=True)
def _print_duration_tqdm_jit(total_seconds: np.int64) -> None:
    if total_seconds < 0:
        total_seconds = np.int64(0)

    if total_seconds < np.int64(3600):
        minutes = total_seconds // np.int64(60)
        seconds = total_seconds % np.int64(60)
        _print_2digit_jit(minutes)
        print_inline_jit(":")
        _print_2digit_jit(seconds)
        return

    hours = total_seconds // np.int64(3600)
    minutes = (total_seconds % np.int64(3600)) // np.int64(60)
    seconds = total_seconds % np.int64(60)
    print_inline_jit(hours)
    print_inline_jit(":")
    _print_2digit_jit(minutes)
    print_inline_jit(":")
    _print_2digit_jit(seconds)


@jit(int64(int64, int64, int64), cache=True)
def _estimate_eta_seconds_jit(
    done_iterations: np.int64,
    total_iterations: np.int64,
    elapsed_seconds: np.int64,
) -> np.int64:
    if done_iterations <= 0:
        return np.int64(0)
    if total_iterations <= done_iterations:
        return np.int64(0)
    if elapsed_seconds <= 0:
        return np.int64(0)
    remaining = total_iterations - done_iterations
    return (elapsed_seconds * remaining) // done_iterations


@jit(float64(int64, float64), cache=True)
def _rate_it_per_s_jit(done_iterations: np.int64, elapsed_seconds_f: np.float64) -> np.float64:
    if elapsed_seconds_f <= 1e-12:
        return np.float64(0.0)
    if done_iterations <= 0:
        return np.float64(0.0)
    return np.float64(done_iterations) / elapsed_seconds_f


@jit(void(float64), cache=True)
def _print_float_2dp_jit(value: np.float64) -> None:
    if value < 0:
        value = np.float64(0.0)
    scaled = np.int64(value * np.float64(100.0) + np.float64(0.5))
    whole = scaled // np.int64(100)
    frac = scaled % np.int64(100)
    print_inline_jit(whole)
    print_inline_jit(".")
    if frac < 10:
        print_inline_jit("0")
    print_inline_jit(frac)


@jit(int64(int64, int64, int64, int64, int64, int64, float64), cache=True)
def _print_progress_line_jit(
    done_iterations: np.int64,
    total_iterations: np.int64,
    bar_width: np.int64,
    elapsed_seconds: np.int64,
    eta_seconds: np.int64,
    eta_known: np.int64,
    rate_it_s: np.float64,
) -> np.int64:
    if done_iterations < 0:
        done_iterations = np.int64(0)
    if total_iterations < 0:
        total_iterations = np.int64(0)

    percent = _percent_from_progress(done_iterations, total_iterations)
    filled = (percent * bar_width) // np.int64(100)
    if filled > bar_width:
        filled = bar_width

    print_inline_jit("\rlearn ")
    _print_i64_padded(percent, np.int64(3))
    print_inline_jit("%|")
    for i in range(bar_width):
        if i < filled:
            print_inline_jit("█")
        else:
            print_inline_jit(" ")
    print_inline_jit("| ")

    total_digits = _digits_i64(total_iterations)
    _print_i64_padded(done_iterations, total_digits)
    print_inline_jit("/")
    print_inline_jit(total_iterations)

    print_inline_jit(" [")
    _print_duration_tqdm_jit(elapsed_seconds)
    print_inline_jit("<")
    if eta_known == 1:
        _print_duration_tqdm_jit(eta_seconds)
    else:
        print_inline_jit("--:--")
    print_inline_jit(", ")
    _print_float_2dp_jit(rate_it_s)
    print_inline_jit("it/s]")
    # Clear stale trailing chars if current line is shorter than previous line.
    print_inline_jit("   ")

    return percent


@jit(int64(int64, int64, int64, int64, float64), cache=True)
def update_progress_bar_jit(
    done_iterations: np.int64,
    total_iterations: np.int64,
    last_percent: np.int64,
    bar_width: np.int64,
    start_seconds: np.float64,
) -> np.int64:
    next_percent = _percent_from_progress(done_iterations, total_iterations)
    if next_percent <= last_percent:
        return last_percent

    elapsed_seconds_f = perf_counter_seconds_jit() - start_seconds
    if elapsed_seconds_f < 0.0:
        elapsed_seconds_f = np.float64(0.0)
    elapsed_seconds = np.int64(elapsed_seconds_f)

    rate_it_s = _rate_it_per_s_jit(done_iterations, elapsed_seconds_f)

    eta_known = np.int64(0)
    eta_seconds = np.int64(0)
    if done_iterations > 0 and total_iterations > done_iterations and rate_it_s > 0.0:
        eta_seconds = _estimate_eta_seconds_jit(done_iterations, total_iterations, elapsed_seconds)
        eta_known = np.int64(1)
    elif done_iterations >= total_iterations:
        eta_known = np.int64(1)

    shown_percent = _print_progress_line_jit(
        done_iterations,
        total_iterations,
        bar_width,
        elapsed_seconds,
        eta_seconds,
        eta_known,
        rate_it_s,
    )
    flush_stdout_jit()
    return shown_percent


@jit(void(int64, float64, int64), cache=True)
def print_progress_complete_jit(
    total_iterations: np.int64,
    elapsed_seconds_f: np.float64,
    bar_width: np.int64,
) -> None:
    if elapsed_seconds_f < 0.0:
        elapsed_seconds_f = np.float64(0.0)
    elapsed_seconds = np.int64(elapsed_seconds_f)
    rate_it_s = _rate_it_per_s_jit(total_iterations, elapsed_seconds_f)
    _print_progress_line_jit(
        total_iterations,
        total_iterations,
        bar_width,
        elapsed_seconds,
        np.int64(0),
        np.int64(1),
        rate_it_s,
    )
    print_newline_jit()
