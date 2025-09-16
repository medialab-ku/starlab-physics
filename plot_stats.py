import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# New naming: <prefix>-<variant>-<key>.npy
#   prefix  := dt<dt>-tol<tol>
#   variant := ours-pcg | ours-nopcg | 2014Bender-nopcg
# Also keep backward compatibility with older patterns.
FNAME_RE_PREFIX  = re.compile(r"^(.+)-(ours-pcg|ours-nopcg|2014Bender-nopcg)-(elapsed_time_ms|opt_iter|opt_error|pcg_iter|pcg_error)\.npy$")
FNAME_RE_LABELED = re.compile(r"^(\d{8}_\d{6})-([A-Za-z0-9]+)-([A-Za-z0-9_]+)\.npy$")
FNAME_RE_LEGACY  = re.compile(r"^(\d{8}_\d{6})-([A-Za-z0-9_]+)\.npy$")
KNOWN_KEYS = [
    "elapsed_time_ms",
    "opt_iter",
    "opt_error",
    "pcg_iter",
    "pcg_error",
]


def scan_groups(stats_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Scan by prefix and variant for the new naming; fall back to legacy patterns.

    Returns: {prefix: {variant: {key: path}}}
    """
    groups: Dict[str, Dict[str, Dict[str, str]]] = {}
    if not os.path.isdir(stats_dir):
        return groups

    for fname in os.listdir(stats_dir):
        m = FNAME_RE_PREFIX.match(fname)
        if m:
            prefix, variant, key = m.group(1), m.group(2), m.group(3)
            path = os.path.join(stats_dir, fname)
            groups.setdefault(prefix, {}).setdefault(variant, {})[key] = path
            continue

        # backward compatibility: map legacy timestamps to their own prefix
        m2 = FNAME_RE_LABELED.match(fname)
        if m2:
            ts, label, key = m2.group(1), m2.group(2), m2.group(3)
            path = os.path.join(stats_dir, fname)
            groups.setdefault(f"{ts}-{label}", {}).setdefault("legacy", {})[key] = path
            continue

        m3 = FNAME_RE_LEGACY.match(fname)
        if m3:
            ts, key = m3.group(1), m3.group(2)
            path = os.path.join(stats_dir, fname)
            groups.setdefault(ts, {}).setdefault("legacy", {})[key] = path

    return groups


def parse_prefix_fields(prefix: str) -> Tuple[str, str]:
    """Parse dt and tol from prefix for display."""
    parts = prefix.split("-")
    dt = tol = ""
    for p in parts:
        if p.startswith("dt"):
            dt = p[2:]
        elif p.startswith("tol"):
            tol = p[3:]
    return dt, tol


def pick_timestamp(groups: Dict[str, Dict[str, str]], ts_opt: str) -> str:
    if not groups:
        return ""
    if ts_opt and ts_opt.lower() != "latest":
        return ts_opt if ts_opt in groups else ""
    # latest by lexicographic order matches chronological for the pattern
    return sorted(groups.keys())[-1]


def plot_group(group: Dict[str, str], out_path: str = None, show: bool = False, logy_errors: bool = True):
    # Determine which keys are present (preserve KNOWN_KEYS order)
    present_keys: List[str] = [k for k in KNOWN_KEYS if k in group]
    if len(present_keys) == 0:
        print("No known stats in selected group.")
        return

    nrows = len(present_keys)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9, 1 + 2.5 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    for ax, key in zip(axes, present_keys):
        arr = np.load(group[key])
        x = np.arange(len(arr))
        ax.plot(x, arr, lw=1.2)
        ax.set_title(key)
        ax.set_xlabel("index")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, alpha=0.3)
        if logy_errors and ("error" in key):
            # avoid non-positive values breaking log-scale
            safe = np.clip(arr, 1e-16, None)
            ax.clear()
            ax.semilogy(x, safe, lw=1.2)
            ax.set_title(key + " (log10)")
            ax.set_xlabel("index")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(True, which="both", alpha=0.3)

        if key == "elapsed_time_ms" and len(arr) > 0:
            mean_ms = float(np.mean(arr))
            ax.axhline(mean_ms, color="orange", lw=1.0, ls="--", alpha=0.7, label=f"mean {mean_ms:.2f} ms")
            ax.legend(loc="best")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_groups_overlay(groups: List[Dict[str, str]], labels: List[str], out_path: str = None, show: bool = False, logy_errors: bool = True):
    """Overlay multiple experiment groups on the same axes per metric.

    groups: list of {key -> npy_path}
    labels: list of legend labels (same length as groups)
    """
    if len(groups) == 0:
        print("No groups to plot.")
        return
    if len(groups) != len(labels):
        raise ValueError("labels length must match groups length")

    # Determine union of present keys in KNOWN_KEYS order
    present_keys: List[str] = [k for k in KNOWN_KEYS if any(k in g for g in groups)]
    if len(present_keys) == 0:
        print("No known stats in provided groups.")
        return

    nrows = len(present_keys)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 1 + 2.8 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    for ax, key in zip(axes, present_keys):
        # For error metrics, use log-scale across all series
        use_log = bool(logy_errors and ("error" in key))
        if use_log:
            ax.set_yscale("log")

        for group, label in zip(groups, labels):
            if key not in group:
                continue
            arr = np.load(group[key])
            if use_log:
                arr = np.clip(arr, 1e-16, None)
            x = np.arange(len(arr))
            ax.plot(x, arr, lw=1.2, label=label)

        ax.set_title(key + (" (log)" if use_log else ""))
        ax.set_xlabel("index")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)

        if key == "elapsed_time_ms":
            # Draw per-series mean lines in matching colors
            handles, _ = ax.get_legend_handles_labels()
            for line, group in zip(handles, groups):
                if key not in group:
                    continue
                arr = np.load(group[key])
                mean_ms = float(np.mean(arr)) if len(arr) > 0 else float("nan")
                ax.axhline(mean_ms, color=line.get_color(), lw=1.0, ls="--", alpha=0.7)

        ax.legend(loc="best")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot stats by prefix: overlay variants (ours/2014Bender) sharing the same prefix")
    parser.add_argument("--dir", default=os.path.join("data", "stats"), help="Directory containing stats .npy files")
    parser.add_argument("--prefix", required=True, help="Prefix to match (dt..-tol..-ours|2014Bender-..)")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--save", default=None, help="Output image path. Auto-picked if not provided")
    parser.add_argument("--no-logy-errors", action="store_true", help="Do not use log-scale for *error series")
    args = parser.parse_args()

    # Collect groups by exact prefix match
    groups_by_prefix = scan_groups(args.dir)
    if args.prefix not in groups_by_prefix:
        print("No matching prefix found.")
        return
    variants = groups_by_prefix[args.prefix]
    # Pick available variants among the three expected ones
    ordered_variants = [v for v in ["ours-pcg", "ours-nopcg", "2014Bender-nopcg"] if v in variants]
    if len(ordered_variants) < 2:
        # If fewer than 2 variants, fall back to a single-variant plot
        single_variant = ordered_variants[0] if ordered_variants else next(iter(variants))
        group = variants[single_variant]
        out_path = args.save
        if out_path is None:
            safe_prefix = args.prefix.replace("/", "_")
            out_path = os.path.join(args.dir, f"{safe_prefix}-{single_variant}-plot.png")
        plot_group(group, out_path=out_path, show=bool(args.show), logy_errors=not args.no_logy_errors)
        return

    # Build overlay groups and labels
    group_list = [variants[v] for v in ordered_variants]
    labels = ordered_variants

    out_path = args.save
    if out_path is None:
        safe_prefix = args.prefix.replace("/", "_")
        out_path = os.path.join(args.dir, f"{safe_prefix}-compare.png")

    plot_groups_overlay(group_list, labels, out_path=out_path, show=bool(args.show), logy_errors=not args.no_logy_errors)


if __name__ == "__main__":
    main()


