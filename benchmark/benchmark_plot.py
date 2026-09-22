import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

# use default publication style if available
style_path = Path(__file__).resolve().parents[3] / "publications/base.mplstyle"
if style_path.is_file():
    plt.style.use([style_path])
else:
    plt.style.use(["default"])

with Path(__file__).with_name("benchmark_results.json").open() as f:
    raw_results = json.load(f)


METRICS = [
    "wampy_static_compile_avg_us",
    "wampy_static_query_avg_us",
    "wampy_dynamic_compile_avg_us",
    "wampy_dynamic_query_avg_us",
    "janus_compile_avg_us",
    "janus_query_avg_us",
]


# --- Colors from mplstyle ---


def lighten(color, amount=0.45):
    r, g, b = to_rgb(color)
    return (
        r + (1.0 - r) * amount,
        g + (1.0 - g) * amount,
        b + (1.0 - b) * amount,
    )


style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

static_color = style_colors[4]
dynamic_color = style_colors[2]
janus_color = style_colors[3]


# --- Group samples by iteration count ---

grouped = defaultdict(lambda: {metric: [] for metric in METRICS})

for row in raw_results:
    n = row["n_iterations"]

    for metric in METRICS:
        value = row.get(metric)
        if value is not None:
            grouped[n][metric].append(value)


# --- Aggregate samples ---

results = []

for n in sorted(grouped.keys()):
    row = {"n_iterations": n}

    for metric in METRICS:
        samples = np.array(grouped[n][metric], dtype=float)

        if len(samples) == 0:
            row[metric] = np.nan
            row[metric.replace("_avg_us", "_std_us")] = np.nan
            row[metric.replace("_avg_us", "_min_us")] = np.nan
            row[metric.replace("_avg_us", "_max_us")] = np.nan
            continue

        row[metric] = float(np.mean(samples))
        row[metric.replace("_avg_us", "_std_us")] = float(np.std(samples))
        row[metric.replace("_avg_us", "_min_us")] = float(np.min(samples))
        row[metric.replace("_avg_us", "_max_us")] = float(np.max(samples))

    row["n_samples"] = max(len(grouped[n][metric]) for metric in METRICS)
    results.append(row)


# --- Helpers ---


def values(metric):
    return np.array([row.get(metric, np.nan) for row in results], dtype=float)


def speedup(baseline, candidate):
    """
    Speedup of candidate over baseline.

    Example:
        baseline = 800 us
        candidate = 28 us
        speedup = 800 / 28 = 28.6x
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return baseline / candidate


def speedup_std(baseline, baseline_std, candidate, candidate_std):
    """
    Approximate propagated standard deviation for baseline / candidate.

    For y = a / b:
        sigma_y / y ~= sqrt((sigma_a / a)^2 + (sigma_b / b)^2)
    """
    s = speedup(baseline, candidate)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(np.square(baseline_std / baseline) + np.square(candidate_std / candidate))

    return s * rel


def add_horizontal_bar_labels(ax, bars, vals, errs=None):
    for i, (bar, value) in enumerate(zip(bars, vals, strict=True)):
        if np.isnan(value):
            continue

        err = 0.0
        if errs is not None and not np.isnan(errs[i]):
            err = errs[i]

        ax.annotate(
            f"{value:.1f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err,
            ),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )


# --- Data ---

labels = [str(row["n_iterations"]) for row in results]
x = np.arange(len(labels))

static_compile = values("wampy_static_compile_avg_us")
static_query = values("wampy_static_query_avg_us")
static_total = static_compile + static_query

dynamic_compile = values("wampy_dynamic_compile_avg_us")

# Prefer measured dynamic query values if present.
# If the dynamic query metric is absent, fall back to static query values.
dynamic_query_raw = values("wampy_dynamic_query_avg_us")
dynamic_query = np.where(np.isnan(dynamic_query_raw), static_query, dynamic_query_raw)
dynamic_total = dynamic_compile + dynamic_query

janus_compile = values("janus_compile_avg_us")
janus_query = values("janus_query_avg_us")
janus_total = janus_compile + janus_query

static_compile_std = values("wampy_static_compile_std_us")
static_query_std = values("wampy_static_query_std_us")

dynamic_compile_std = values("wampy_dynamic_compile_std_us")

dynamic_query_raw_std = values("wampy_dynamic_query_std_us")
dynamic_query_std = np.where(
    np.isnan(dynamic_query_raw_std),
    static_query_std,
    dynamic_query_raw_std,
)

janus_compile_std = values("janus_compile_std_us")
janus_query_std = values("janus_query_std_us")


# --- Query times ---

print("\nQuery times [µs]")
print("--------------------------------")
print(f"{'iterations':>10} | {'janus_swi':>18} | {'WAMpy static':>18} | {'WAMpy dynamic':>18}")
print("-" * 79)

for label, j, j_std, s, s_std, d, d_std in zip(
    labels,
    janus_query,
    janus_query_std,
    static_query,
    static_query_std,
    dynamic_query,
    dynamic_query_std,
    strict=True,
):
    print(
        f"{label:>10} | "
        f"{j:>7.2f} ± {j_std:<7.2f} | "
        f"{s:>7.2f} ± {s_std:<7.2f} | "
        f"{d:>7.2f} ± {d_std:<7.2f}"
    )

print("-" * 79)


# --- Compilation times ---

print("\nCompilation times [µs]")
print("--------------------------------")
print(f"{'iterations':>10} | {'janus_swi':>18} | {'WAMpy static':>18} | {'WAMpy dynamic':>18}")
print("-" * 79)

for label, j, j_std, s, s_std, d, d_std in zip(
    labels,
    janus_compile,
    janus_compile_std,
    static_compile,
    static_compile_std,
    dynamic_compile,
    dynamic_compile_std,
    strict=True,
):
    print(
        f"{label:>10} | "
        f"{j:>7.2f} ± {j_std:<7.2f} | "
        f"{s:>7.2f} ± {s_std:<7.2f} | "
        f"{d:>7.2f} ± {d_std:<7.2f}"
    )

print("-" * 79)


# --- Total-time uncertainty ---

static_total_std = np.sqrt(np.square(static_compile_std) + np.square(static_query_std))

dynamic_total_std = np.sqrt(np.square(dynamic_compile_std) + np.square(dynamic_query_std))

janus_total_std = np.sqrt(np.square(janus_compile_std) + np.square(janus_query_std))


# --- End-to-end times ---

print("\nEnd-to-end times [µs]")
print("--------------------------------")
print(f"{'iterations':>10} | {'janus_swi':>18} | {'WAMpy static':>18} | {'WAMpy dynamic':>18}")
print("-" * 79)

for label, j, j_std, s, s_std, d, d_std in zip(
    labels,
    janus_total,
    janus_total_std,
    static_total,
    static_total_std,
    dynamic_total,
    dynamic_total_std,
    strict=True,
):
    print(
        f"{label:>10} | "
        f"{j:>7.2f} ± {j_std:<7.2f} | "
        f"{s:>7.2f} ± {s_std:<7.2f} | "
        f"{d:>7.2f} ± {d_std:<7.2f}"
    )

print("-" * 79)


# --- Speedups over janus_swi ---

static_compile_speedup = speedup(janus_compile, static_compile)
dynamic_compile_speedup = speedup(janus_compile, dynamic_compile)

static_compile_speedup_std = speedup_std(
    janus_compile,
    janus_compile_std,
    static_compile,
    static_compile_std,
)

dynamic_compile_speedup_std = speedup_std(
    janus_compile,
    janus_compile_std,
    dynamic_compile,
    dynamic_compile_std,
)


# ======================================================================
# Combined figure:
#   Left  = compilation speedup bar plot
#   Right = total time line plot in µs on log scale
# ======================================================================

error_kw = {
    "elinewidth": 0.6,
    "capthick": 0.6,
}

fig, (ax_total, ax_compile) = plt.subplots(
    1,
    2,
    figsize=(4.75, 2.1),
    gridspec_kw={"width_ratios": [1.1, 1.0]},
)

# ----------------------------------------------------------------------
# Left subfigure: compilation speedup bar plot
# ----------------------------------------------------------------------

bar_width = 0.46

static_compile_x = x - bar_width / 2
dynamic_compile_x = x + bar_width / 2

static_compile_bars = ax_compile.bar(
    static_compile_x,
    static_compile,
    bar_width,
    yerr=static_compile_std,
    capsize=2,
    error_kw=error_kw,
    label="WAMpy static",
    color=static_color,
)

dynamic_compile_bars = ax_compile.bar(
    dynamic_compile_x,
    dynamic_compile,
    bar_width,
    yerr=dynamic_compile_std,
    capsize=2,
    error_kw=error_kw,
    label="WAMpy dynamic",
    color=dynamic_color,
)

add_horizontal_bar_labels(
    ax_compile,
    static_compile_bars[-1:],
    static_compile[-1:],
    static_compile_std[-1:],
)

add_horizontal_bar_labels(
    ax_compile,
    dynamic_compile_bars[-1:],
    dynamic_compile[-1:],
    dynamic_compile_std[-1:],
)

ax_compile.set_title("b. Compilation time")
ax_compile.set_xlabel("Iterations")
ax_compile.set_ylabel("Time [µs]")

ax_compile.set_xticks(x)
ax_compile.set_xticklabels(labels)

ax_compile.grid(axis="y", linewidth=0.4, alpha=0.4)

# ----------------------------------------------------------------------
# Right subfigure: total time line plot in µs on log scale
# ----------------------------------------------------------------------

ax_total.fill_between(
    x,
    np.maximum(static_total - static_total_std, np.finfo(float).tiny),
    static_total + static_total_std,
    color=lighten(static_color, 0.7),
    alpha=1,
    linewidth=0,
)

ax_total.fill_between(
    x,
    np.maximum(dynamic_total - dynamic_total_std, np.finfo(float).tiny),
    dynamic_total + dynamic_total_std,
    color=lighten(dynamic_color, 0.7),
    alpha=1,
    linewidth=0,
)

ax_total.fill_between(
    x,
    np.maximum(janus_total - janus_total_std, np.finfo(float).tiny),
    janus_total + janus_total_std,
    color=lighten(janus_color, 0.7),
    alpha=1,
    linewidth=0,
)

ax_total.plot(
    x,
    dynamic_total,
    marker="s",
    linewidth=1.0,
    markersize=3,
    label="WAMpy partial recompilation",
    color=dynamic_color,
)

ax_total.plot(
    x,
    static_total,
    marker="o",
    linewidth=1.0,
    markersize=3,
    label="WAMpy full compilation",
    color=static_color,
)

ax_total.plot(
    x,
    janus_total,
    marker="^",
    linewidth=1.0,
    markersize=3,
    label="Janus SWI-Prolog",
    color=janus_color,
)


ax_total.set_yscale("log")

ax_total.set_title("a. End-to-end time")
ax_total.set_ylabel("Time [µs, log]")
ax_total.set_xlabel("Iterations")

ax_total.set_xticks(x)
ax_total.set_xticklabels(labels)

ax_total.grid(axis="y", which="both", linewidth=0.4, alpha=0.4)

# ----------------------------------------------------------------------
# Shared styling
# ----------------------------------------------------------------------

for ax in [ax_compile, ax_total]:
    ax.tick_params(axis="both", labelsize=7)
    ax.title.set_fontsize(8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

# Shared legend
handles_total, labels_total = ax_total.get_legend_handles_labels()

fig.legend(
    handles_total,
    labels_total,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    frameon=False,
    fontsize=7,
    handlelength=1.2,
    columnspacing=0.8,
)

fig.subplots_adjust(
    left=0.10,
    right=0.98,
    bottom=0.28,
    top=0.82,
    wspace=0.32,
)

fig.savefig(
    "publications/2026/IJCLR/demo/fig/benchmark@4_75in.pdf",
    format="pdf",
    bbox_inches=None,
)

plt.show()
