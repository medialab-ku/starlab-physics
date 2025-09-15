import os
import re
import argparse
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


FNAME_RE = re.compile(r"^(\d{8}_\d{6})-([A-Za-z0-9_]+)\.npy$")
KNOWN_KEYS = [
    "elapsed_time_ms",
    "opt_iter",
    "opt_error",
    "pcg_iter",
    "pcg_error",
]


def scan_groups(stats_dir: str) -> Dict[str, Dict[str, str]]:
    """Scan directory and group npy files by timestamp prefix.

    Returns: {timestamp: {key: filepath}}
    """
    groups: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(stats_dir):
        return groups

    for fname in os.listdir(stats_dir):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        ts, key = m.group(1), m.group(2)
        path = os.path.join(stats_dir, fname)
        if ts not in groups:
            groups[ts] = {}
        groups[ts][key] = path
    return groups


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
        ax.grid(True, alpha=0.3)
        if logy_errors and ("error" in key):
            # avoid non-positive values breaking log-scale
            safe = np.clip(arr, 1e-16, None)
            ax.clear()
            ax.semilogy(x, safe, lw=1.2)
            ax.set_title(key + " (log10)")
            ax.set_xlabel("index")
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


def main():
    parser = argparse.ArgumentParser(description="Plot stats saved as .npy from data/stats")
    parser.add_argument("--dir", default=os.path.join("data", "stats"), help="Directory containing stats .npy files")
    parser.add_argument("--ts", default="latest", help="Timestamp to plot (e.g., 20250915_142030) or 'latest'")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--save", default=None, help="Output image path. Default: data/stats/<ts>-plot.png")
    parser.add_argument("--no-logy-errors", action="store_true", help="Do not use log-scale for *error series")
    args = parser.parse_args()

    groups = scan_groups(args.dir)
    ts = pick_timestamp(groups, args.ts)
    if not ts:
        print("No stats found.")
        return

    group = groups[ts]
    out_path = args.save
    if out_path is None:
        out_path = os.path.join(args.dir, f"{ts}-plot.png")

    plot_group(group, out_path=out_path, show=bool(args.show), logy_errors=not args.no_logy_errors)


if __name__ == "__main__":
    main()


