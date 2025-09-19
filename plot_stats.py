import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# File naming: <prefix>-<variant>-<key>.npy
#   prefix  := arbitrary string (often dt<dt>-tol<tol>-opt<maxOptIter>)
#   variant := iisph | 2014Bender | (pcg|nopcg)-ours
#   key     := elapsed_time_ms | opt_iter | opt_error | pcg_iter | pcg_error
FNAME_RE_PREFIX = re.compile(
    r"^(.+)-((?:iisph)|(?:2014Bender)|(?:(?:pcg|nopcg)-ours))-(elapsed_time_ms|opt_iter|opt_error|pcg_iter|pcg_error)\.npy$"
)

KNOWN_KEYS = [
    "elapsed_time_ms",
    "opt_iter",
    "opt_error",
    "pcg_iter",
    "pcg_error",
]


#############################################
# Style and utilities
#############################################

def _style_for_label(label: str):
    """Return (color, linestyle, zorder) for a given variant label.

    Policy:
    - pcg-ours: royalblue (topmost)
    - nopcg-ours: seagreen
    - nopcg-2014Bender: orange
    - fallback: gray
    """
    if label.startswith("pcg-ours"):
        return ("royalblue", "-", 10)
    if label.startswith("nopcg-ours"):
        return ("seagreen", "-", 5)
    if label == "2014Bender":
        return ("orange", "-", 1)
    return ("gray", "-", 1)


def _sorted_pairs(groups: List[Dict[str, str]], labels: List[str]):
    """Return (group, label) pairs with pcg-ours variants rendered last (on top)."""
    pairs = list(zip(groups, labels))
    pairs.sort(key=lambda gl: (gl[1].startswith("pcg-ours")))
    return pairs


#############################################
# Scan and selection
#############################################

def scan_groups(stats_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Scan .npy files and build map {prefix: {variant: {key: path}}}.

    Note: This scans non-recursively (only the given directory).
    For different comparisons, run the script with other parent directories accordingly.
    """
    groups: Dict[str, Dict[str, Dict[str, str]]] = {}
    if not os.path.isdir(stats_dir):
        return groups

    for fname in os.listdir(stats_dir):
        if not fname.endswith(".npy"):
            continue
        m = FNAME_RE_PREFIX.match(fname)
        if not m:
            continue
        prefix, variant, key = m.group(1), m.group(2), m.group(3)
        path = os.path.join(stats_dir, fname)
        groups.setdefault(prefix, {}).setdefault(variant, {})[key] = path

    return groups


def parse_compare_item(spec: str) -> Tuple[str, str]:
    """Parse 'prefix/variant' string into (prefix, variant)."""
    if "/" not in spec:
        raise ValueError("--compare item must be in 'prefix/variant' form.")
    prefix, variant = spec.split("/", 1)
    if not prefix or not variant:
        raise ValueError("--compare parse failed: prefix or variant is empty.")
    return prefix, variant


def auto_labels(compare_pairs: List[Tuple[str, str]], user_labels: List[str] = None) -> List[str]:
    """Auto-generate legend labels.

    If all prefixes are equal, use the variant only; otherwise use 'variant@prefix'.
    If user_labels are provided with matching length, return them as-is.
    """
    if user_labels is not None and len(user_labels) == len(compare_pairs):
        return user_labels

    prefixes = [p for p, _ in compare_pairs]
    all_same_prefix = all(p == prefixes[0] for p in prefixes)
    labels: List[str] = []
    for prefix, variant in compare_pairs:
        labels.append(variant if all_same_prefix else f"{variant}@{prefix}")
    return labels


#############################################
# Axis and slicing
#############################################

def _xlabel_for_key(key: str) -> str:
    if key in ("elapsed_time_ms", "opt_iter"):
        return "timestep"
    if ("error" in key) or (key == "pcg_iter"):
        return "iteration"
    return "index"


def _title_for_key(key: str, *, use_log: bool = False) -> str:
    """Return a human-friendly plot title for a given key."""
    mapping = {
        "elapsed_time_ms": "Elapsed Time (ms)",
        "opt_iter": "Iteration",
        "opt_error": "Error",
        "pcg_iter": "PCG Iteration",
        "pcg_error": "PCG Error",
    }
    base = mapping.get(key, key)
    return base + (" (log)" if use_log else "")


def _slice_for_key(
    arr: np.ndarray,
    key: str,
    *,
    start: int = None,
    end: int = None,
    opt_iter: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) with frame slicing and error-series iteration mapping.

    For error-series (keys containing 'error') and pcg_iter, when start/end are provided
    and opt_iter is given, map the frame range [start, end) into an iteration subrange.
    Otherwise, x is the natural index (timestep or iteration).
    """
    base = arr

    if start is not None or end is not None:
        s = 0 if start is None else max(0, int(start))
        e = len(base) if (end is None or int(end) < 0) else int(end)
        if "error" in key:
            if opt_iter is not None:
                s_it = int(np.sum(opt_iter[:s]))
                e_it = int(np.sum(opt_iter[:e]))
                s_it = max(0, min(s_it, len(base)))
                e_it = max(s_it, min(e_it, len(base)))
                sub = base[s_it:e_it]
                x = np.arange(len(sub))
                return x, sub
            # no mapping info; fall through to natural slicing
        elif key == "pcg_iter":
            # pcg_iter is per-timestep counts; natural slicing by timestep
            sub = base[s:e]
            x = np.arange(s, s + len(sub))
            return x, sub
        # non-error/non-pcg_iter or missing mapping
        sub = base[s:e]
        x = np.arange(s, s + len(sub))
        return x, sub

    # no explicit range
    if "error" in key:
        x = np.arange(len(base))
    else:
        x = np.arange(len(base))
    return x, base


#############################################
# Plotting
#############################################

def _draw_mean_line(ax, key: str, x_y: Tuple[np.ndarray, np.ndarray], color: str):
    """Draw a horizontal mean line for given series using provided color (no legend entry)."""
    _, y_series = x_y
    if y_series.size == 0:
        return
    mean_v = float(np.mean(y_series))
    ax.axhline(mean_v, color=color, lw=1.0, ls=":", alpha=0.9)


def plot_groups_overlay(
    groups: List[Dict[str, str]],
    labels: List[str],
    *,
    keys: List[str] = None,
    out_path: str = None,
    show: bool = False,
    logy_errors: bool = True,
    separate_figs: bool = False,
    start: int = None,
    end: int = None,
):
    if len(groups) == 0:
        print("No groups to plot.")
        return
    if len(groups) != len(labels):
        raise ValueError("labels length must match groups length")

    # Determine which keys to plot
    if (keys is None) or (len(keys) == 0) or ("all" in keys):
        selected_keys = [k for k in KNOWN_KEYS if any(k in g for g in groups)]
    else:
        selected_keys = [k for k in keys if k in KNOWN_KEYS and any(k in g for g in groups)]
    if len(selected_keys) == 0:
        print("No known keys present in provided groups.")
        return

    pairs = _sorted_pairs(groups, labels)

    def _plot_one_key(ax, key: str):
        use_log = bool(logy_errors and ("error" in key))
        if use_log:
            ax.set_yscale("log")
        for group, label in pairs:
            if key not in group:
                continue
            arr_full = np.load(group[key])
            opt_iter_arr = None
            if ("error" in key) and ("opt_iter" in group):
                opt_iter_arr = np.load(group["opt_iter"])  # per-timestep counts
            arr_plot = np.clip(arr_full, 1e-16, None) if use_log else arr_full
            color, _, z = _style_for_label(label)
            x_plot, y_plot = _slice_for_key(arr_plot, key, start=start, end=end, opt_iter=opt_iter_arr)
            # Series lines are always solid; only mean lines are dotted
            line = ax.plot(x_plot, y_plot, lw=1.2, label=label, color=color, ls="-", zorder=z)[0]
            if key in ("elapsed_time_ms", "opt_iter"):
                _draw_mean_line(ax, key, (x_plot, y_plot), color=line.get_color())
        ax.set_title(_title_for_key(key, use_log=use_log))
        ax.set_xlabel(_xlabel_for_key(key))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)
        ax.legend(loc="best")

    if separate_figs:
        # Create one figure per key
        base_out = out_path
        if base_out is None:
            safe = [f"{lbl}" for lbl in labels]
            base_out = os.path.join(
                os.getcwd(),
                "compare-" + "__".join(safe) + ".png",
            )
        base_no_ext, ext = os.path.splitext(base_out)
        for key in selected_keys:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 3.0), constrained_layout=True)
            _plot_one_key(ax, key)
            out_key = f"{base_no_ext}-{key}{ext or '.png'}"
            os.makedirs(os.path.dirname(out_key), exist_ok=True)
            fig.savefig(out_key, dpi=150)
            print(f"Saved figure to {out_key}")
            if show:
                plt.show()
            else:
                plt.close(fig)
        return

    # Combined in a single image
    nrows = len(selected_keys)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 1 + 2.8 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]
    for ax, key in zip(axes, selected_keys):
        _plot_one_key(ax, key)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


#############################################
# Main
#############################################

def main():
    parser = argparse.ArgumentParser(description="Flexible stats plotter: compare arbitrary prefix/variant pairs")
    parser.add_argument("--dir", default=os.path.join("data", "stats"), help="Directory containing stats .npy files")
    parser.add_argument("--compare", "-c", nargs="+", required=True, help="Items to compare: 'prefix/variant'. Multiple allowed")
    parser.add_argument("--labels", "-l", nargs="*", default=None, help="Legend labels for compared items; auto when omitted")
    parser.add_argument("--keys", "-k", nargs="+", default=["all"], choices=KNOWN_KEYS + ["all"], help="Keys to plot. Default 'all'")
    parser.add_argument("--separate-figs", action="store_true", help="Save one image per key instead of a combined image")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    parser.add_argument("--save", default=None, help="Output image path. Auto-picked when omitted")
    parser.add_argument("--no-logy-errors", action="store_true", help="Do not use log scale for *error series")
    parser.add_argument("--start", type=int, default=None, help="Start frame index (inclusive) for timestep-level series and error slicing")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive) for timestep-level series and error slicing; negative or omitted means till end")
    args = parser.parse_args()

    groups_by_prefix = scan_groups(args.dir)
    if not groups_by_prefix:
        print("No recognizable .npy stats files in the given directory.")
        return

    # Parse compare items and pick groups
    try:
        compare_pairs = [parse_compare_item(s) for s in args.compare]
    except ValueError as e:
        print(f"--compare parse error: {e}")
        return

    missing: List[str] = []
    group_list: List[Dict[str, str]] = []
    for prefix, variant in compare_pairs:
        if prefix not in groups_by_prefix:
            missing.append(f"{prefix}/{variant} (missing prefix)")
            continue
        variants = groups_by_prefix[prefix]
        if variant not in variants:
            missing.append(f"{prefix}/{variant} (missing variant)")
            continue
        group_list.append(variants[variant])

    if missing:
        print("Missing compare items:")
        for m in missing:
            print("  - ", m)
        if not group_list:
            return

    labels = auto_labels(compare_pairs, args.labels)
    if len(labels) != len(group_list):
        kept_pairs: List[Tuple[str, str]] = []
        for (prefix, variant), g in zip(compare_pairs, group_list):
            kept_pairs.append((prefix, variant))
        labels = auto_labels(kept_pairs, None)

    # Build output path
    out_path = args.save
    if out_path is None:
        safe = [f"{p.replace('/', '_')}-{v}" for p, v in compare_pairs]
        base = "compare-" + "__".join(safe)
        out_path = os.path.join(args.dir, base + ".png")

    # Determine selected keys
    selected_keys = None if (args.keys is None or ("all" in args.keys)) else args.keys

    # Always use overlay logic (works for single or multiple groups)
    plot_groups_overlay(
        group_list,
        labels,
        keys=selected_keys,
        out_path=out_path,
        show=bool(args.show),
        logy_errors=not args.no_logy_errors,
        separate_figs=bool(args.separate_figs),
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()


