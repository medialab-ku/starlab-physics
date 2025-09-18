import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# New naming: <prefix>-<variant>-<key>.npy
#   prefix  := dt<dt>-tol<tol>-opt<maxOptIter>
#   variant := pcg-ours | nopcg-ours | nopcg-2014Bender
# Also keep backward compatibility with older patterns.
FNAME_RE_PREFIX  = re.compile(r"^(.+)-(pcg-ours-[A-Za-z0-9]+|nopcg-ours-[A-Za-z0-9]+|nopcg-2014Bender)-(elapsed_time_ms|opt_iter|opt_error|pcg_iter|pcg_error|ldv_mean|ldv_max|assemble_ms|pcg_precond_ms|pcg_matvec_ms|other_ms|jtj_ms)\.npy$")
FNAME_RE_LABELED = re.compile(r"^(\d{8}_\d{6})-([A-Za-z0-9]+)-([A-Za-z0-9_]+)\.npy$")
FNAME_RE_LEGACY  = re.compile(r"^(\d{8}_\d{6})-([A-Za-z0-9_]+)\.npy$")
KNOWN_KEYS = [
    "elapsed_time_ms",
    "opt_iter",
    "opt_error",
    "pcg_iter",
    "pcg_error",
    "ldv_mean",
    "ldv_max",
    # new timing breakdown
    "assemble_ms",
    "pcg_precond_ms",
    "pcg_matvec_ms",
        "other_ms",
        "jtj_ms",
]

#############################################
# Styling helpers and utilities
#############################################

def _style_for_label(label: str):
    """Return (color, linestyle, zorder) for a given variant label.

    Policy:
    - pcg-ours-Hii: crimson
    - pcg-ours-Mii: royalblue
    - pcg-ours-I: darkorange
    - nopcg-ours: seagreen
    - nopcg-2014Bender: orange
    - fallback: gray
    """
    if label.startswith("pcg-ours-"):
        tag = label.split("-")[-1]
        if tag == "Hii":
            return ("crimson", "-", 12)
        if tag == "Mii":
            return ("royalblue", "-", 11)
        if tag == "I":
            return ("darkorange", "-", 10)
        return ("royalblue", "-", 10)
    if label.startswith("pcg-ours"):
        return ("royalblue", "-", 10)
    if label.startswith("nopcg-ours"):
        return ("seagreen", "-", 5)
    if label == "nopcg-2014Bender":
        return ("orange", "-", 1)
    return ("gray", "-", 1)


def _sorted_pairs(groups: List[Dict[str, str]], labels: List[str]):
    """Return list of (group, label) with pcg-ours last for topmost rendering."""
    pairs = list(zip(groups, labels))
    pairs.sort(key=lambda gl: (gl[1].startswith("pcg-ours")))
    return pairs


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


#############################################
# Sampling and warmup controls (no CLI)
#############################################
# Skip first N points for warmup
WARMUP_SKIP: int = 10
# Downsample rules by key type
STRIDE_ERROR: int = 100           # for iteration-level error and pcg_iter
STRIDE_TIMESTEP: int = 10         # for timestep-level series


def _downsample_for_key(arr: np.ndarray, key: str, *, sample: bool = False, start: int = None, end: int = None,
                        opt_iter: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x, y) sliced by frame range and with proper x-axis labeling.

    - sample=False: use full raw array (no warmup skip). sample=True: apply warmup skip to timestep arrays.
    - For timestep-level arrays: when start/end are provided, slice by [start:end] and set x to the actual frame
      numbers (start..end-1). Otherwise, x starts at 0 (or WARMUP_SKIP if sample=True).
    - For iteration-level arrays (keys containing "error"): when start/end provided and opt_iter is given,
      map the frame range into an iteration subrange. x is iteration index within the slice (0..N-1).
    """
    # sample=True  -> full data (no warmup skip)
    # sample=False -> apply warmup skip for timestep series
    if sample:
        base = arr
    else:
        base = arr[WARMUP_SKIP:] if WARMUP_SKIP > 0 else arr

    # Defaults for x-axis labeling
    x_offset = 0
    if (not sample) and (start is None and end is None) and ("error" not in key):
        # Warmup skip applied → offset x by skipped frames to show global frame ids
        x_offset = WARMUP_SKIP

    arr2 = base
    # Frame range slicing and x-axis mapping
    if start is not None or end is not None:
        s = 0 if start is None else max(0, int(start))
        # end is exclusive in CLI
        # When sample=True and base already skipped warmup, the indices s/e refer to global frames,
        # but base is truncated. Convert to local indices by subtracting the skip for timestep series.
        if (not sample) and ("error" not in key):
            s_local = max(0, s - WARMUP_SKIP)
        else:
            s_local = s
        e_global = None if end is None else int(end)
        if e_global is None or e_global < 0:
            e_local = len(base)
        else:
            e_local = e_global - (WARMUP_SKIP if (sample and ("error" not in key)) else 0)
            e_local = max(s_local, min(e_local, len(base)))

        if "error" in key:
            # iteration-level series: map frame range to iteration index range using opt_iter
            if opt_iter is not None:
                opt_iter2 = opt_iter if sample else (opt_iter[WARMUP_SKIP:] if WARMUP_SKIP > 0 else opt_iter)
                s_it = int(np.sum(opt_iter2[:s]))
                e_it = int(np.sum(opt_iter2[:(e_global if e_global is not None and e_global >= 0 else len(opt_iter2))]))
                s_it = max(0, min(s_it, len(base)))
                e_it = max(s_it, min(e_it, len(base)))
                arr2 = base[s_it:e_it]
            else:
                arr2 = base
            x = np.arange(len(arr2))
            return x, arr2
        else:
            # timestep-level series
            arr2 = base[s_local:e_local]
            x = np.arange(s, s + len(arr2))
            return x, arr2

    # No explicit range
    if "error" in key:
        x = np.arange(len(arr2))
    else:
        x = np.arange(x_offset, x_offset + len(arr2))
    return x, arr2


def pick_timestamp(groups: Dict[str, Dict[str, str]], ts_opt: str) -> str:
    if not groups:
        return ""
    if ts_opt and ts_opt.lower() != "latest":
        return ts_opt if ts_opt in groups else ""
    # latest by lexicographic order matches chronological for the pattern
    return sorted(groups.keys())[-1]


def _xlabel_for_key(key: str) -> str:
    # Timestep series
    if key in ("elapsed_time_ms", "opt_iter", "ldv_mean", "ldv_max"):
        return "timestep"
    # Iteration series (errors and pcg_iter)
    if ("error" in key) or (key == "pcg_iter"):
        return "iteration"
    return "index"


def plot_group(group: Dict[str, str], out_path: str = None, show: bool = False, logy_errors: bool = True,
               *, sample: bool = False, start: int = None, end: int = None, iter_frame: int = None,
               iter_out: str = None):
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
        arr_full = np.load(group[key])

        # Provide opt_iter to slice iteration series by frame range when needed
        opt_iter_arr = None
        if ("error" in key) and ("opt_iter" in group):
            opt_iter_arr = np.load(group["opt_iter"])  # per-timestep counts

        if logy_errors and ("error" in key):
            safe_full = np.clip(arr_full, 1e-16, None)
            x_plot, y_plot = _downsample_for_key(safe_full, key, sample=sample, start=start, end=end, opt_iter=opt_iter_arr)
            ax.semilogy(x_plot, y_plot, lw=1.2)
            ax.set_title(key + " (log10)")
            ax.set_xlabel(_xlabel_for_key(key))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.grid(True, which="both", alpha=0.3)
        else:
            x_plot, y_plot = _downsample_for_key(arr_full, key, sample=sample, start=start, end=end, opt_iter=opt_iter_arr)
            ax.plot(x_plot, y_plot, lw=1.2)
            ax.set_title(key)
            ax.set_xlabel(_xlabel_for_key(key))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=100))
            ax.grid(True, alpha=0.3)

        if key in ("elapsed_time_ms", "opt_iter"):
            # Use the same slicing and x-axis mapping as plotted series for mean computation
            x_tmp, arr_wo = _downsample_for_key(arr_full, key, sample=sample, start=start, end=end)
            if len(arr_wo) > 0:
                mean_ms = float(np.mean(arr_wo))
                # dotted mean line for the series
                label_txt = f"mean {mean_ms:.2f}" if key != "opt_iter" else f"mean {mean_ms:.2f} iters"
                ax.axhline(mean_ms, color="orange", lw=1.0, ls=":", alpha=0.9, label=label_txt)
                ax.legend(loc="best")

    # Optional: per-timestep iteration error decay plot
    if (iter_frame is not None) and ("opt_error" in group) and ("opt_iter" in group):
        try:
            opt_err = np.load(group["opt_error"])  # iteration-level
            iters_per_step = np.load(group["opt_iter"])  # per timestep
            # Determine raw arrays (no warmup) unless raw=False
            if sample and WARMUP_SKIP > 0:
                # For iteration series, warmup skip corresponds to skipping first WARMUP_SKIP timesteps
                skip_iters = int(np.sum(iters_per_step[:WARMUP_SKIP]))
                opt_err_use = opt_err[skip_iters:]
                iters_use = iters_per_step[WARMUP_SKIP:]
            else:
                opt_err_use = opt_err
                iters_use = iters_per_step

            # Frame range slicing does not affect which single frame we show; iter_frame is global in selected arrays
            f_idx = int(iter_frame)
            if f_idx < 0:
                f_idx = 0
            if f_idx >= len(iters_use):
                f_idx = len(iters_use) - 1
            start_it = int(np.sum(iters_use[:f_idx]))
            end_it = start_it + int(iters_use[f_idx])
            sub_err = np.clip(opt_err_use[start_it:end_it], 1e-16, None)

            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.semilogy(np.arange(len(sub_err)), sub_err, lw=1.4, color="crimson")
            ax2.set_title(f"opt_error decay in timestep {f_idx} (iters={len(sub_err)})")
            ax2.set_xlabel("iteration")
            ax2.set_ylabel("opt_error")
            ax2.grid(True, which="both", alpha=0.3)
            if iter_out:
                os.makedirs(os.path.dirname(iter_out), exist_ok=True)
                fig2.savefig(iter_out, dpi=150)
                print(f"Saved iteration decay figure to {iter_out}")
            plt.close(fig2)
        except Exception as e:
            print(f"[warn] iteration decay plot failed: {e}")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_groups_overlay(groups: List[Dict[str, str]], labels: List[str], out_path: str = None, show: bool = False, logy_errors: bool = True,
                        *, sample: bool = False, start: int = None, end: int = None, iter_frame: int = None, iter_out: str = None):
    """Overlay multiple experiment groups on the same axes per metric.

    groups: list of {key -> npy_path}
    labels: list of legend labels (same length as groups)
    """
    if len(groups) == 0:
        print("No groups to plot.")
        return
    if len(groups) != len(labels):
        raise ValueError("labels length must match groups length")

    # Focus only on requested 5 metrics for overlay
    focus = ["elapsed_time_ms", "pcg_precond_ms", "pcg_matvec_ms", "assemble_ms", "other_ms"]
    present_keys: List[str] = [k for k in focus if any(k in g for g in groups)]
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

        # Ensure plotting order: non-pcg first, then pcg-ours last (topmost)
        pairs = _sorted_pairs(groups, labels)
        for group, label in pairs:
            if key not in group:
                continue
            arr_full = np.load(group[key])
            # Provide opt_iter for iteration-level slicing by frame range
            opt_iter_arr = None
            if ("error" in key) and ("opt_iter" in group):
                opt_iter_arr = np.load(group["opt_iter"])  # per timestep
            arr_base = arr_full if sample else (arr_full[WARMUP_SKIP:] if WARMUP_SKIP > 0 else arr_full)
            arr_plot = np.clip(arr_base, 1e-16, None) if use_log else arr_base
            x_plot, y_plot = _downsample_for_key(arr_plot, key, sample=True if sample else False, start=start, end=end, opt_iter=opt_iter_arr if not sample else opt_iter_arr)
            color, ls, z = _style_for_label(label)
            # shorten label to precond tag when comparing preconditions
            short_label = label.split("-")[-1] if label.startswith("pcg-ours-") else label
            ax.plot(x_plot, y_plot, lw=1.2, label=short_label, color=color, ls=ls, zorder=z)

        ax.set_title(key + (" (log)" if use_log else ""))
        ax.set_xlabel(_xlabel_for_key(key))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)

        if key in ("elapsed_time_ms",):
            # Draw per-series mean lines for the same sliced ranges used in plotting
            for group, label in pairs:
                if key not in group:
                    continue
                arr_full = np.load(group[key])
                x_tmp, arr_wo = _downsample_for_key(arr_full, key, sample=sample, start=start, end=end)
                mean_ms = float(np.mean(arr_wo)) if len(arr_wo) > 0 else float("nan")
                color, _, z = _style_for_label(label)
                ax.axhline(mean_ms, lw=1.0, ls=":", alpha=0.9, zorder=z, color=color)

        ax.legend(loc="best")

    # Optional: overlay iteration error decay for a specific timestep across variants
    if iter_frame is not None:
        try:
            f_idx = int(iter_frame)
            curves: List[Tuple[str, np.ndarray]] = []
            pairs_it = _sorted_pairs(groups, labels)
            for group, label in pairs_it:
                if ("opt_error" not in group) or ("opt_iter" not in group):
                    continue
                opt_err = np.load(group["opt_error"])   # iteration-level
                iters_per_step = np.load(group["opt_iter"])  # per timestep counts
                if sample and WARMUP_SKIP > 0:
                    skip_iters = int(np.sum(iters_per_step[:WARMUP_SKIP]))
                    opt_err_use = opt_err[skip_iters:]
                    iters_use = iters_per_step[WARMUP_SKIP:]
                else:
                    opt_err_use = opt_err
                    iters_use = iters_per_step
                if len(iters_use) == 0:
                    continue
                ff = max(0, min(f_idx, len(iters_use) - 1))
                start_it = int(np.sum(iters_use[:ff]))
                end_it = start_it + int(iters_use[ff])
                sub_err = np.clip(opt_err_use[start_it:end_it], 1e-16, None)
                if sub_err.size > 0:
                    curves.append((label, sub_err))

            if len(curves) > 0:
                fig_it, ax_it = plt.subplots(figsize=(7, 3))
                # Plot non-pcg first, pcg last (top)
                curves.sort(key=lambda ls: (ls[0] == "pcg-ours"))
                for label, sub_err in curves:
                    color, ls_, z = _style_for_label(label)
                    ax_it.semilogy(np.arange(len(sub_err)), sub_err, lw=1.4, label=label, zorder=z, color=color, ls=ls_)
                ax_it.set_title(f"opt_error decay in timestep {f_idx}")
                ax_it.set_xlabel("iteration")
                ax_it.set_ylabel("opt_error")
                ax_it.grid(True, which="both", alpha=0.3)
                ax_it.legend(loc="best")
                if iter_out:
                    os.makedirs(os.path.dirname(iter_out), exist_ok=True)
                    fig_it.savefig(iter_out, dpi=150)
                    print(f"Saved iteration decay overlay to {iter_out}")
                if show:
                    plt.show()
                plt.close(fig_it)
        except Exception as e:
            print(f"[warn] overlay iteration decay plot failed: {e}")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot stats by prefix: overlay variants (pcg/nopcg + label) sharing the same prefix")
    parser.add_argument("--dir", default=os.path.join("data", "stats"), help="Directory containing stats .npy files")
    parser.add_argument("--prefix", required=True, help="Prefix to match (dt..-tol..-opt..)")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--save", default=None, help="Output image path. Auto-picked if not provided")
    parser.add_argument("--no-logy-errors", action="store_true", help="Do not use log-scale for *error series")
    parser.add_argument("--sample", action="store_true", help="Disable warmup skip and any downsampling; use full data")
    parser.add_argument("--start", type=int, default=None, help="Start frame index (inclusive) for timestep-level series and error slicing")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive) for timestep-level series and error slicing; negative or omitted means till end")
    parser.add_argument("--iter-frame", type=int, default=None, help="Plot iteration error decay for this timestep index (single-variant mode)")
    parser.add_argument("--iter-out", default=None, help="Output image path for the iteration decay figure")
    args = parser.parse_args()

    # Collect groups by exact prefix match
    groups_by_prefix = scan_groups(args.dir)
    if args.prefix not in groups_by_prefix:
        print("No matching prefix found.")
        return
    variants = groups_by_prefix[args.prefix]
    # Prefer overlaying all precondition variants for our PCG method: pcg-ours-<tag>
    precond_keys = [k for k in variants.keys() if k.startswith("pcg-ours-")]
    if len(precond_keys) >= 2:
        order_map = {"Hii": 0, "Mii": 1, "I": 2}
        ordered_variants = sorted(precond_keys, key=lambda k: order_map.get(k.split("-")[-1], 99))
    else:
        # Fallback: Pick available variants among the expected ones (pcg flag first)
        ordered_variants = [v for v in ["pcg-ours", "nopcg-ours", "nopcg-2014Bender"] if v in variants]
    if len(ordered_variants) < 2:
        # If fewer than 2 variants, fall back to a single-variant plot
        single_variant = ordered_variants[0] if ordered_variants else next(iter(variants))
        group = variants[single_variant]
        out_path = args.save
        if out_path is None:
            safe_prefix = args.prefix.replace("/", "_")
            # build suffix from CLI range and iter-frame
            suffix_parts: List[str] = []
            if (args.start is not None) or (args.end is not None):
                s = 0 if args.start is None else int(args.start)
                if (args.end is None) or (args.end < 0):
                    suffix_parts.append(f"f{s}-end")
                else:
                    suffix_parts.append(f"f{s}-{int(args.end)}")
            if args.iter_frame is not None:
                suffix_parts.append(f"iter{int(args.iter_frame)}")
            suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""
            out_path = os.path.join(args.dir, f"{safe_prefix}-{single_variant}{suffix}-plot.png")
        # iteration decay figure path (auto if not provided)
        iter_out = args.iter_out
        if (args.iter_frame is not None) and (iter_out is None):
            safe_prefix = args.prefix.replace("/", "_")
            suffix_parts: List[str] = []
            if (args.start is not None) or (args.end is not None):
                s = 0 if args.start is None else int(args.start)
                if (args.end is None) or (args.end < 0):
                    suffix_parts.append(f"f{s}-end")
                else:
                    suffix_parts.append(f"f{s}-{int(args.end)}")
            suffix_parts.append(f"iter{int(args.iter_frame)}")
            suffix = "-" + "-".join(suffix_parts)
            iter_out = os.path.join(args.dir, f"{safe_prefix}-{single_variant}{suffix}-iter_decay.png")

        plot_group(group, out_path=out_path, show=bool(args.show), logy_errors=not args.no_logy_errors,
                   sample=bool(args.sample), start=args.start, end=args.end, iter_frame=args.iter_frame, iter_out=args.iter_out)
        return

    # Build overlay groups and labels (use full variant key including precond suffix as label)
    group_list = [variants[v] for v in ordered_variants]
    labels = ordered_variants

    # Additionally, create a compact pie/bar chart for PCG overhead breakdown per variant
    # Save one figure per variant showing mean proportions of pcg_precond_ms (incl. jtj), pcg_matvec_ms, assemble_ms, other_ms
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        for v in ordered_variants:
            g = variants[v]
            # Compute means safely; default to 0 when missing
            def _mean_or_zero(key: str) -> float:
                try:
                    if key in g:
                        return float(np.mean(np.load(g[key])))
                except Exception:
                    pass
                return 0.0

            m_elapsed = _mean_or_zero("elapsed_time_ms")
            m_assm = _mean_or_zero("assemble_ms")
            m_prec = _mean_or_zero("pcg_precond_ms")
            m_mv = _mean_or_zero("pcg_matvec_ms")
            m_other = _mean_or_zero("other_ms")

            # Pie: show share of PCG-related parts vs others
            labels_pie = ["precond", "matvec", "assemble", "other"]
            vals_pie = [m_prec, m_mv, m_assm, m_other]
            if sum(vals_pie) <= 0:
                continue
            figp, axp = plt.subplots(figsize=(4.2, 4.2))
            axp.pie(vals_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90)
            axp.set_title(f"Overhead breakdown: {v}")
            out_pie = os.path.join(args.dir, f"{args.prefix}-{v}-overhead_pie.png")
            figp.savefig(out_pie, dpi=140, bbox_inches="tight")
            plt.close(figp)

            # Bar: absolute ms (means)
            figb, axb = plt.subplots(figsize=(6, 3.2))
            axb.bar(labels_pie, vals_pie, color=[_style_for_label(v)[0]]*4)
            axb.set_ylabel("ms (mean per timestep)")
            axb.set_title(f"Absolute overheads: {v}")
            axb.grid(True, axis='y', alpha=0.3)
            out_bar = os.path.join(args.dir, f"{args.prefix}-{v}-overhead_bar.png")
            figb.savefig(out_bar, dpi=140, bbox_inches="tight")
            plt.close(figb)
    except Exception as e:
        print(f"[warn] overhead breakdown plot failed: {e}")

    out_path = args.save
    if out_path is None:
        safe_prefix = args.prefix.replace("/", "_")
        # build suffix from CLI range and iter-frame
        suffix_parts: List[str] = []
        if (args.start is not None) or (args.end is not None):
            s = 0 if args.start is None else int(args.start)
            if (args.end is None) or (args.end < 0):
                suffix_parts.append(f"f{s}-end")
            else:
                suffix_parts.append(f"f{s}-{int(args.end)}")
        if args.iter_frame is not None:
            suffix_parts.append(f"iter{int(args.iter_frame)}")
        suffix = ("-" + "-".join(suffix_parts)) if suffix_parts else ""
        out_path = os.path.join(args.dir, f"{safe_prefix}{suffix}-compare.png")

    # iteration decay figure path (auto if not provided)
    iter_out = args.iter_out
    if (args.iter_frame is not None) and (iter_out is None):
        safe_prefix = args.prefix.replace("/", "_")
        suffix_parts: List[str] = []
        if (args.start is not None) or (args.end is not None):
            s = 0 if args.start is None else int(args.start)
            if (args.end is None) or (args.end < 0):
                suffix_parts.append(f"f{s}-end")
            else:
                suffix_parts.append(f"f{s}-{int(args.end)}")
        suffix_parts.append(f"iter{int(args.iter_frame)}")
        suffix = "-" + "-".join(suffix_parts)
        iter_out = os.path.join(args.dir, f"{safe_prefix}{suffix}-iter_decay_overlay.png")

    plot_groups_overlay(group_list, labels, out_path=out_path, show=bool(args.show), logy_errors=not args.no_logy_errors,
                        sample=bool(args.sample), start=args.start, end=args.end, iter_frame=args.iter_frame, iter_out=iter_out)


if __name__ == "__main__":
    main()


