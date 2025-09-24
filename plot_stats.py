import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "Noto Serif"]
plt.rcParams["axes.labelsize"] = 14  # slightly larger axis label font size
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# File naming: <prefix>-<variant>-<key>.npy
#   prefix  := arbitrary string (often dt<dt>-tol<tol>-opt<maxOptIter>), may include flags like '-cfl' and '-warmstart'
#   variant := iisph | 2014Bender | ours
#              (backward-compat: also accepts legacy 'pcg-ours' and 'nopcg-ours')
#   key     := elapsed_time_ms | opt_iter | opt_error | pcg_iter | pcg_error
FNAME_RE_PREFIX = re.compile(
    r"^(.+)-((?:iisph)|(?:2014Bender)|(?:ours)|(?:(?:pcg|nopcg)-ours))-(elapsed_time_ms|opt_iter|opt_error|pcg_iter|pcg_error)\.npy$"
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

def _base_variant(label: str) -> str:
    """Extract base variant name from a legend label.

    Labels may be auto-generated as 'variant@prefix' when comparing different prefixes.
    This function strips the '@prefix' suffix for color/style decisions.
    """
    if not label:
        return label
    return label.split("@", 1)[0]


def _style_for_label(label: str):
    """Return (color, linestyle, zorder) for a given variant label.

    Policy:
    - ours (and legacy pcg-ours/nopcg-ours): royalblue (topmost)
    - 2014Bender: orange
    - iisph: seagreen
    - fallback: gray
    """
    base = _base_variant(label)
    if base == "ours" or base.startswith("pcg-ours") or base.startswith("nopcg-ours"):
        return ("royalblue", "-", 10)
    if base == "2014Bender":
        return ("orange", "-", 1)
    if base == "iisph":
        return ("seagreen", "-", 1)
    return ("gray", "-", 1)


def _sorted_pairs(groups: List[Dict[str, str]], labels: List[str]):
    """Return (group, label) pairs with 'ours' variants rendered last (on top)."""
    pairs = list(zip(groups, labels))
    pairs.sort(key=lambda gl: (
        _base_variant(gl[1]) == "ours"
        or _base_variant(gl[1]).startswith("pcg-ours")
        or _base_variant(gl[1]).startswith("nopcg-ours")
    ))
    return pairs


def _smooth_series(y: np.ndarray, ma_window: int = 0, ema_alpha: float = None) -> np.ndarray:
    """Return a smoothed copy of y using MA or EMA.

    - If ema_alpha is provided (0<a<1), use EMA (takes precedence)
    - Else if ma_window > 1, use centered moving average with "same" length
    - Otherwise, return y unchanged
    """
    if y.size == 0:
        return y
    if ema_alpha is not None:
        a = float(ema_alpha)
        if not (0.0 < a < 1.0):
            return y
        out = np.empty_like(y, dtype=float)
        out[0] = float(y[0])
        for i in range(1, y.size):
            out[i] = a * float(y[i]) + (1.0 - a) * out[i - 1]
        return out
    if ma_window and int(ma_window) > 1:
        w = int(ma_window)
        kernel = np.ones(w, dtype=float) / float(w)
        # Edge padding to avoid boundary drop when slicing; keep length
        y_f = y.astype(float)
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        y_pad = np.pad(y_f, (pad_left, pad_right), mode="edge")
        return np.convolve(y_pad, kernel, mode="valid")
    return y


def _extract_prefix_from_fname(fname: str) -> str:
    base = os.path.basename(fname)
    m = FNAME_RE_PREFIX.match(base)
    if not m:
        return ""
    return m.group(1)


def _parse_dt_from_prefix(prefix: str) -> float:
    m = re.search(r"dt([0-9]*\.?[0-9]+)", prefix)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_tol_from_prefix(prefix: str) -> int:
    m = re.search(r"-tol(\d+)", prefix)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _get_dt_for_group(group: Dict[str, str]) -> float:
    # pick any file path from the group
    for _k, p in group.items():
        prefix = _extract_prefix_from_fname(p)
        dt = _parse_dt_from_prefix(prefix)
        if dt is not None:
            return dt
    return None


def _extract_scene_from_prefix(prefix: str) -> str:
    """Heuristic: scene name is substring before '-dt', else before first '-', else the whole prefix."""
    if not prefix:
        return ""
    if "-dt" in prefix:
        return prefix.split("-dt", 1)[0]
    if "-" in prefix:
        return prefix.split("-", 1)[0]
    return prefix


def _legend_text_for_label(label: str) -> str:
    base = _base_variant(label)
    if base == "ours" or base.startswith("pcg-ours") or base.startswith("nopcg-ours"):
        return "Ours"
    if base == "2014Bender":
        return "Bender et al."
    return base or label


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
    if ("error" in key) or key in ("elapsed_time_ms", "opt_iter", "pcg_iter"):
        return "timestep"
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
    return base


def _ylabel_for_key(key: str) -> str:
    mapping = {
        "elapsed_time_ms": "Elapsed time (ms)",
        "opt_iter": "Iteration count",
        "opt_error": r"$\|$Δv$\|^2$",
        "pcg_iter": "PCG iteration counts",
        "pcg_error": "PCG error",
    }
    return mapping.get(key, key)


def _slice_for_key(
    arr: np.ndarray,
    key: str,
    *,
    start: int = None,
    end: int = None,
    opt_iter: np.ndarray = None,
    pcg_iter: np.ndarray = None,
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
            # Map error-series to iteration subrange using the corresponding per-timestep iteration counts
            iter_counts = None
            if key == "pcg_error" and pcg_iter is not None:
                iter_counts = pcg_iter
            elif key != "pcg_error" and opt_iter is not None:
                iter_counts = opt_iter
            if iter_counts is not None:
                s_it = int(np.sum(iter_counts[:s]))
                e_it = int(np.sum(iter_counts[:e]))
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
    ylog: bool = False,
    show_grid: bool = False,
    separate_figs: bool = False,
    start: int = None,
    end: int = None,
    iter_decay_frame: int = None,
    dt_min: float = None,
    dt_max: float = None,
    smooth: int = 0,
    ema: float = None,
    show_raw: bool = True,
    title_text: str = None
):
    if len(groups) == 0:
        print("No groups to plot.")
        return
    if len(groups) != len(labels):
        raise ValueError("labels length must match groups length")

    # Determine which keys to plot
    # Support extra, derived keys (e.g., 'iter_decay', 'avg_iter_vs_dt') in addition to KNOWN_KEYS
    if (keys is None) or (len(keys) == 0) or ("all" in keys):
        selected_keys = [k for k in KNOWN_KEYS if any(k in g for g in groups)]
    else:
        selected_keys = []
        for k in keys:
            if k == "iter_decay":
                selected_keys.append(k)
            elif k == "avg_iter_vs_dt":
                selected_keys.append(k)
            elif (k in KNOWN_KEYS) and any(k in g for g in groups):
                selected_keys.append(k)
    if len(selected_keys) == 0:
        print("No known keys present in provided groups.")
        return

    pairs = _sorted_pairs(groups, labels)

    # When comparing multiple items of the same base variant (e.g., two 'ours'),
    # assign distinct colors per-label so overlays are visually separable.
    labels_in_order = [lbl for _grp, lbl in pairs]

    def _build_label_styles(labels_list: List[str]):
        base_counts = {}
        for lbl in labels_list:
            b = _base_variant(lbl)
            base_counts[b] = base_counts.get(b, 0) + 1
        # Use a dedicated palette for 'ours' duplicates; keep zorder high
        palette_ours = [
            "royalblue",
            "crimson",
            "darkmagenta",
            "darkcyan",
            "goldenrod",
            "forestgreen",
            "slateblue",
            "chocolate",
        ]
        styles = {}
        ours_index = 0
        for lbl in labels_list:
            b = _base_variant(lbl)
            is_ours_family = (b == "ours") or b.startswith("pcg-ours") or b.startswith("nopcg-ours")
            if is_ours_family and base_counts.get(b, 0) > 1:
                color = palette_ours[ours_index % len(palette_ours)]
                ours_index += 1
                styles[lbl] = (color, "-", 10)
            else:
                styles[lbl] = _style_for_label(lbl)
        return styles

    _label_styles = _build_label_styles(labels_in_order)

    def _style_for(lbl: str):
        return _label_styles.get(lbl, _style_for_label(lbl))

    # Figure sizing: single-column width for 2-column papers
    # Approx ~3.4 inches wide per subplot; adjust height per row
    # width_inch to meet 600px at 150dpi
    target_px_w = 800
    dpi = 150
    single_col_w = target_px_w / dpi
    row_h = 1.8

    # Integrated plotting; 'iter_decay' handled as a derived key in _plot_one_key

    # Multi-mode: combined image with multiple keys → do not convert timestep x to seconds
    multi_mode = (not separate_figs) and (len(selected_keys) > 1)

    # Helper: print the frame index where the difference of per-frame MAX(opt_error)
    # across compared variants is maximal within current [start, end). Print only once.
    _printed_error_diff = {"done": False}

    def _maybe_print_max_error_diff():
        if _printed_error_diff["done"]:
            return
        per_frame_max_arrays = []
        min_frames = None
        for group, _label in pairs:
            if ("opt_error" in group) and ("opt_iter" in group):
                try:
                    err = np.load(group["opt_error"]).astype(float)
                    itc = np.load(group["opt_iter"]).astype(int)
                except Exception:
                    continue
                if itc.size == 0:
                    continue
                cum = np.concatenate(([0], np.cumsum(itc)))
                # compute max error per frame
                frame_max = []
                for f in range(itc.size):
                    s_i, e_i = int(cum[f]), int(cum[f + 1])
                    if e_i <= s_i:
                        frame_max.append(0.0)
                    else:
                        frame_max.append(float(np.max(err[s_i:e_i])))
                arr = np.asarray(frame_max, dtype=float)
                per_frame_max_arrays.append(arr)
                min_frames = arr.size if min_frames is None else min(min_frames, arr.size)
        if len(per_frame_max_arrays) < 2 or not min_frames:
            return
        s = int(start) if (start is not None and int(start) >= 0) else 0
        e = int(end) if (end is not None and int(end) >= 0) else min_frames
        s = max(0, min(s, min_frames))
        e = max(s, min(e, min_frames))
        if e <= s:
            return
        best_idx = s
        best_diff = -1.0
        for i in range(s, e):
            vals = [float(a[i]) for a in per_frame_max_arrays]
            d = float(max(vals) - min(vals))
            if d > best_diff:
                best_diff = d
                best_idx = i
        print(best_idx)
        _printed_error_diff["done"] = True

    # If we're in multi-mode, print once here
    if multi_mode:
        _maybe_print_max_error_diff()

    def _plot_one_key(ax, key: str):
        use_log = bool(ylog)
        if use_log:
            ax.set_yscale("log")
        any_line = False
        global_x_min, global_x_max = None, None

        # Special handling for iter_decay: align series to the maximum iteration count within the frame
        if key == "iter_decay":
            collected = []  # list of tuples (label, y_raw)
            max_len = 0
            frame = int(iter_decay_frame) if (iter_decay_frame is not None) else None

            # Auto-select timestep: choose the frame where the difference between
            # max error of Ours and Bender is largest. Fallback to first label if needed.
            if frame is None:
                # Find candidate labels for 'ours' and '2014Bender'
                ours_idx = None
                bender_idx = None
                for idx, (_g, lbl) in enumerate(pairs):
                    base = _base_variant(lbl)
                    if ours_idx is None and (base == "ours" or base.startswith("pcg-ours") or base.startswith("nopcg-ours")):
                        ours_idx = idx
                    if bender_idx is None and base == "2014Bender":
                        bender_idx = idx
                # Compute argmax over frames when both present and data available
                if ours_idx is not None and bender_idx is not None:
                    g_ours = pairs[ours_idx][0]
                    g_bndr = pairs[bender_idx][0]
                    if ("opt_error" in g_ours and "opt_iter" in g_ours and
                        "opt_error" in g_bndr and "opt_iter" in g_bndr):
                        err_o = np.load(g_ours["opt_error"]).astype(float)
                        it_o = np.load(g_ours["opt_iter"]).astype(int)
                        err_b = np.load(g_bndr["opt_error"]).astype(float)
                        it_b = np.load(g_bndr["opt_iter"]).astype(int)
                        num_frames = min(it_o.size, it_b.size)
                        best_frame = None
                        best_diff = -1.0
                        cum_o = np.concatenate(([0], np.cumsum(it_o)))
                        cum_b = np.concatenate(([0], np.cumsum(it_b)))
                        for f in range(num_frames):
                            s_o, e_o = int(cum_o[f]), int(cum_o[f+1])
                            s_b, e_b = int(cum_b[f]), int(cum_b[f+1])
                            if e_o <= s_o or e_b <= s_b:
                                continue
                            max_o = float(np.max(err_o[s_o:e_o]))
                            max_b = float(np.max(err_b[s_b:e_b]))
                            diff = abs(max_o - max_b)
                            if diff > best_diff:
                                best_diff = diff
                                best_frame = f
                        if best_frame is not None:
                            frame = int(best_frame)
                            print(f"iter_decay: auto-selected timestep {frame} (max |Δerror_max| between Ours and Bender)")
                # Fallback: pick frame with largest max error from the first available label
                if frame is None and len(pairs) > 0:
                    g0 = pairs[0][0]
                    if ("opt_error" in g0) and ("opt_iter" in g0):
                        err0 = np.load(g0["opt_error"]).astype(float)
                        it0 = np.load(g0["opt_iter"]).astype(int)
                        if it0.size > 0:
                            cum0 = np.concatenate(([0], np.cumsum(it0)))
                            best_frame = 0
                            best_max = -1.0
                            for f in range(it0.size):
                                s0, e0 = int(cum0[f]), int(cum0[f+1])
                                if e0 <= s0:
                                    continue
                                mx = float(np.max(err0[s0:e0]))
                                if mx > best_max:
                                    best_max = mx
                                    best_frame = f
                            frame = int(best_frame)
                            print(f"iter_decay: auto-selected timestep {frame} (fallback by max error)")

            for group, label in pairs:
                if (frame is None) or ("opt_error" not in group) or ("opt_iter" not in group):
                    continue
                opt_err = np.load(group["opt_error"])  # use raw for slicing; clip later if needed
                opt_it = np.load(group["opt_iter"])  # per-timestep counts
                if frame < 0 or frame >= len(opt_it):
                    continue
                start_idx = int(np.sum(opt_it[:frame]))
                end_idx = start_idx + int(opt_it[frame])
                if end_idx <= start_idx:
                    continue
                y_raw = opt_err[start_idx:end_idx]
                if use_log:
                    y_raw = np.clip(y_raw, 1e-16, None)
                y_raw = np.asarray(y_raw, dtype=float)
                collected.append((label, y_raw))
                if y_raw.size > max_len:
                    max_len = y_raw.size
            if max_len == 0 or len(collected) == 0:
                return
            # Force integer iteration index on x-axis
            x_master = np.arange(max_len, dtype=int)
            # Determine tolerance threshold (10^-tol) from any participating prefix
            tol_mag = None
            for group, _label in pairs:
                if len(group) == 0:
                    continue
                try:
                    any_path = next(iter(group.values()))
                except Exception:
                    any_path = ""
                prefix = _extract_prefix_from_fname(any_path)
                tmag = _parse_tol_from_prefix(prefix)
                if tmag is not None:
                    tol_mag = tmag if tol_mag is None else min(tol_mag, tmag)
            thr = (10.0 ** (-int(tol_mag))) if tol_mag is not None else None
            for label, y_raw in collected:
                real_len = int(y_raw.size)
                y_disp = y_raw.astype(float)
                color, _, z = _style_for(label)
                do_ma = bool(smooth and int(smooth) > 1)
                do_ema = ema is not None
                if do_ma or do_ema:
                    y_s = _smooth_series(y_disp, ma_window=int(smooth) if do_ma else 0, ema_alpha=ema if do_ema else None)
                    if show_raw:
                        # Draw original unsmoothed real segment faintly
                        ax.plot(x_master[:real_len], y_disp[:real_len], lw=0.5, color=color, alpha=0.25, ls="-", zorder=z-1)
                    # Draw smoothed only on real segment
                    ax.plot(x_master[:real_len], y_s[:real_len], lw=0.8, label=label, color=color, ls="-", zorder=z)
                    if thr is not None:
                        arr = y_s[:real_len]
                        idxs = np.nonzero(arr <= thr)[0]
                        if idxs.size > 0:
                            k = int(idxs[0])
                            ax.axvline(k, color=color, ls="--", lw=0.8, alpha=0.7, zorder=z+1)
                else:
                    # Draw only real iterations; no padding
                    ax.plot(x_master[:real_len], y_disp[:real_len], lw=0.9, label=label, color=color, ls="-", zorder=z)
                    if thr is not None:
                        arr = y_disp[:real_len]
                        idxs = np.nonzero(arr <= thr)[0]
                        if idxs.size > 0:
                            k = int(idxs[0])
                            ax.axvline(k, color=color, ls="--", lw=0.8, alpha=0.7, zorder=z+1)
                any_line = True
                last_x = x_master
            ax.set_xlabel("Iteration")

            if any_line:
                ax.margins(x=0)
                if global_x_min is not None and global_x_max is not None:
                    ax.set_xlim(left=0, right=float(global_x_max))
                else:
                    ax.set_xlim(left=0)

            if not use_log:
                ax.set_ylim(bottom=0)
                yticks = ax.get_yticks()
                ax.set_yticks([t for t in yticks if abs(t) > 1e-12])
            # Draw tolerance reference line
            if thr is not None:
                ax.axhline(thr, color="gray", lw=0.8, ls="--", alpha=0.7)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            if show_grid:
                ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)
            # Larger tick labels in multi mode
            if multi_mode:
                ax.tick_params(axis='x', labelsize=12)
                ax.tick_params(axis='y', labelsize=11)
            handles, labels_txt = ax.get_legend_handles_labels()
            labels_txt = [_legend_text_for_label(l) for l in labels_txt]
            leg = ax.legend(handles, labels_txt, loc="best")
            if leg is not None:
                leg.get_frame().set_linewidth(0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            return
        # Special aggregation for avg_iter_vs_dt: group by variant and plot dt vs avg iterations
        if key == "avg_iter_vs_dt":
            variant_to_points: Dict[str, List[Tuple[float, float]]] = {}
            for group, label in pairs:
                if "opt_iter" not in group:
                    continue
                dt_val = _get_dt_for_group(group)
                if dt_val is None:
                    continue
                if (dt_min is not None and dt_val < float(dt_min)) or (dt_max is not None and dt_val > float(dt_max)):
                    continue
                it = np.load(group["opt_iter"])
                s = int(start) if (start is not None and int(start) >= 0) else 0
                e = int(end) if (end is not None and int(end) >= 0) else it.size
                s = max(0, min(s, it.size))
                e = max(s, min(e, it.size))
                it_sub = it[s:e]
                if it_sub.size == 0:
                    continue
                avg_iter = float(np.mean(it_sub))
                base_var = _base_variant(label)
                # Normalize legacy names to 'ours'
                if base_var.startswith("pcg-ours") or base_var.startswith("nopcg-ours"):
                    base_var = "ours"
                variant_to_points.setdefault(base_var, []).append((float(dt_val), avg_iter))
            # Plot aggregated lines
            for variant, pts in variant_to_points.items():
                pts.sort(key=lambda t: t[0])
                xs = np.asarray([p[0] for p in pts], dtype=float)
                ys = np.asarray([p[1] for p in pts], dtype=float)
                if xs.size == 0:
                    continue
                color, _, z = _style_for_label(variant)
                label_txt = _legend_text_for_label(variant)
                ax.plot(xs, ys, lw=1.0, label=label_txt, color=color, ls="-", zorder=z)
                any_line = True
                if xs.size > 0:
                    if global_x_min is None or xs[0] < global_x_min:
                        global_x_min = xs[0]
                    if global_x_max is None or xs[-1] > global_x_max:
                        global_x_max = xs[-1]

            # Axis labels for this special key
            ax.set_xlabel("Δt")
            ax.set_ylabel("avg. iterations")
            # Align x-range
            if any_line and global_x_min is not None and global_x_max is not None:
                ax.set_xlim(left=float(global_x_min), right=float(global_x_max))
            if not use_log:
                ax.set_ylim(bottom=0)
                yticks = ax.get_yticks()
                ax.set_yticks([t for t in yticks if abs(t) > 1e-12])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            if show_grid:
                ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)
            handles, labels_txt = ax.get_legend_handles_labels()
            labels_txt = [_legend_text_for_label(l) for l in labels_txt]
            leg = ax.legend(handles, labels_txt, loc="best")
            if leg is not None:
                leg.get_frame().set_linewidth(0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            return
        for group, label in pairs:
            if key == "iter_decay":
                # This key is handled above as a special case
                continue
            
            x_plot, y_plot = None, None

            if key not in group:
                continue
            
            if key == 'opt_error':
                # Plot the LAST error per frame on a logical time axis.
                if 'opt_iter' not in group:
                    continue
                counts = np.load(group['opt_iter']).astype(int)
                err_flat = np.load(group[key]).astype(float)
                num_frames = int(counts.size)

                y_per_frame = np.full(num_frames, np.nan, dtype=float)
                cum_counts = np.concatenate(([0], np.cumsum(counts)))
                
                for f in range(num_frames):
                    s_it = int(cum_counts[f])
                    e_it = int(cum_counts[f+1])
                    if e_it > s_it and e_it <= len(err_flat):
                        y_per_frame[f] = err_flat[e_it - 1]

                s = int(start) if start is not None and int(start) >= 0 else 0
                e = int(end) if end is not None and int(end) >= 0 else num_frames
                s = max(0, min(s, num_frames))
                e = max(s, min(e, num_frames))
                
                y_plot = y_per_frame[s:e]
                frame_indices = np.arange(s, e)
                
                dt_s = _get_dt_for_group(group)
                if dt_s is None: continue
                x_plot = frame_indices * dt_s
            
            elif "error" in key:
                # For other errors (e.g., pcg_error), expand all iterations to a logical time axis.
                counts_key = "pcg_iter" if key == "pcg_error" else "opt_iter"
                if counts_key not in group:
                    continue
                counts = np.load(group[counts_key]).astype(int)

                dt_s = _get_dt_for_group(group)
                if dt_s is None:
                    print(f"Warning: could not determine dt for '{label}', skipping error plot.")
                    continue
                elapsed_ms_per_frame = dt_s * 1000.0

                num_frames = int(counts.size)
                s = int(start) if (start is not None and int(start) >= 0) else 0
                e = int(end) if (end is not None and int(end) >= 0) else num_frames
                s = max(0, min(s, num_frames))
                e = max(s, min(e, num_frames))
                
                cum_counts = np.concatenate(([0], np.cumsum(counts)))
                s_it = int(cum_counts[s])
                e_it = int(cum_counts[e])
                
                err_flat_full = np.load(group[key]).astype(float)
                err_flat = err_flat_full[s_it:e_it]
                
                times_ms = []
                t0_ms = s * elapsed_ms_per_frame
                t_acc_frame = t0_ms
                for f_idx in range(s, e):
                    iter_count = int(counts[f_idx])
                    if iter_count > 0:
                        inc = elapsed_ms_per_frame / iter_count
                        for j in range(iter_count):
                            times_ms.append(t_acc_frame + (j + 1) * inc)
                    t_acc_frame += elapsed_ms_per_frame
                x_plot = np.asarray(times_ms, dtype=float) / 1000.0
                y_plot = err_flat
            else:
                # Non-error series: use natural series with slicing, then convert to logical time
                arr_full = np.load(group[key])
                x_idx, y_plot = _slice_for_key(
                    arr_full, key, start=start, end=end
                )
                dt = _get_dt_for_group(group)
                if dt is not None:
                    x_plot = x_idx * float(dt)
                else:
                    x_plot = x_idx
            
            if use_log and y_plot is not None:
                y_plot = np.where(np.isnan(y_plot), np.nan, np.clip(y_plot, 1e-16, None))

            y_disp = y_plot
            color, _, z = _style_for(label)
            do_ma = bool(smooth and int(smooth) > 1)
            do_ema = ema is not None
            
            if x_plot is None or y_plot is None: continue

            if do_ma or do_ema:
                y_s = _smooth_series(y_disp, ma_window=int(smooth) if do_ma else 0, ema_alpha=ema if do_ema else None)
                if show_raw:
                    ax.plot(x_plot, y_disp, lw=0.6, color=color, alpha=0.25, ls="-", zorder=z-1)
                line = ax.plot(x_plot, y_s, lw=1.0, label=label, color=color, ls="-", zorder=z)[0]
            else:
                line = ax.plot(x_plot, y_disp, lw=1.0, label=label, color=color, ls="-", zorder=z)[0]

            if key in ("elapsed_time_ms", "opt_iter"):
                _draw_mean_line(ax, key, (x_plot, y_disp), color=line.get_color())
            
            any_line = True
            if x_plot.size > 0:
                if global_x_min is None or x_plot[0] < global_x_min:
                    global_x_min = x_plot[0]
                if global_x_max is None or x_plot[-1] > global_x_max:
                    global_x_max = x_plot[-1]
        
        # Axis labels (move title to y-label)
        if key in ("elapsed_time_ms", "opt_iter", "pcg_iter", "opt_error", "pcg_error"):
            ax.set_xlabel("time (s)")
        elif key == "iter_decay":
            ax.set_xlabel("iteration")
        else:
            ax.set_xlabel(_xlabel_for_key(key))

        ylabel_kwargs = {}
        if key not in ("opt_iter", "pcg_iter", "elapsed_time_ms"):
             ylabel_kwargs = {'rotation': 'horizontal', 'ha': 'right', 'va': 'center', 'x': -0.1}
        
        if key == "iter_decay":
            ax.set_ylabel("$\|$Δv$\|^2$", **ylabel_kwargs)
        else:
            ax.set_ylabel(_ylabel_for_key(key), **ylabel_kwargs)

        # Align x-range to plotted data only
        if any_line and global_x_min is not None and global_x_max is not None:
            ax.set_xlim(left=float(global_x_min), right=float(global_x_max))
        
        if not use_log:
            if "error" in key and any_line:
                ax.margins(y=0.03)
                y_min, _ = ax.get_ylim()
                if y_min >= 0:
                    ax.set_ylim(bottom=0)
                
                ticks = [t for t in ax.get_yticks() if t >= -1e-9]
                if 0.0 not in ticks:
                    ticks.append(0.0)
                ax.set_yticks(sorted(list(set(ticks))))
            else:
                ax.set_ylim(bottom=0)

        # X-axis tick formatting
        try:
            label_lower = ax.get_xlabel().strip().lower()
        except Exception:
            label_lower = ""
        if label_lower == "time (s)":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            # Fallback to float ticks if integer ticks are too sparse
            x0, x1 = ax.get_xlim()
            int_ticks = [t for t in ax.get_xticks() if x0 <= t <= x1 and float(t).is_integer()]
            if len(int_ticks) <= 2:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))

        if show_grid:
            ax.grid(True, which=("both" if use_log else "major"), alpha=0.3)
        if multi_mode:
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=11)
        # Unified legend labels
        handles, labels_txt = ax.get_legend_handles_labels()
        labels_txt = [_legend_text_for_label(l) for l in labels_txt]
        leg = ax.legend(handles, labels_txt, loc="best")
        if leg is not None:
            leg.get_frame().set_linewidth(0.8)
        # remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(single_col_w, row_h), constrained_layout=True)
            if title_text:
                fig.suptitle(title_text)
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
    multi_w_inch = 12.0  # ~1800 px at 150 dpi
    multi_row_h = 2.2
    width_inch = single_col_w if not ((not separate_figs) and (nrows > 1)) else multi_w_inch
    row_h_eff = row_h if not ((not separate_figs) and (nrows > 1)) else multi_row_h
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(width_inch, max(row_h_eff, 0.9 * nrows + 0.9)), constrained_layout=True)
    if title_text:
        fig.suptitle(title_text)
    if nrows == 1:
        axes = [axes]

    for ax, key in zip(axes, selected_keys):
        _plot_one_key(ax, key)

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title_text else None)

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
    parser.add_argument("--keys", "-k", nargs="+", default=["all"], choices=KNOWN_KEYS + ["all", "iter_decay", "avg_iter_vs_dt"], help="Keys to plot. Default 'all'")
    parser.add_argument("--separate-figs", action="store_true", help="Save one image per key instead of a combined image")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    parser.add_argument("--save", default=None, help="Output image path. Auto-picked when omitted")
    parser.add_argument("--ylog", action="store_true", help="Use log scale for Y axis for all series")
    parser.add_argument("--start", type=int, default=None, help="Start frame index (inclusive) for timestep-level series and error slicing")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive) for timestep-level series and error slicing; negative or omitted means till end")
    parser.add_argument("--iter-decay", type=int, default=None, help="Plot opt_error decay for a single timestep index across compared variants")
    parser.add_argument("--dt-min", type=float, default=None, help="Minimum Δt for avg_iter_vs_dt")
    parser.add_argument("--dt-max", type=float, default=None, help="Maximum Δt for avg_iter_vs_dt")
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--show-grid", action="store_true", help="Show grid (off by default)")
    parser.add_argument("--title", type=str, default=None, help="Figure title (omit when not set)")
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

    # Auto-expand compare pairs for avg_iter_vs_dt: allow base prefix without '-dt'
    if args.keys is not None and ("avg_iter_vs_dt" in args.keys):
        expanded: List[Tuple[str, str]] = []
        for (base_prefix, variant) in compare_pairs:
            if "-dt" in base_prefix:
                expanded.append((base_prefix, variant))
                continue
            # Match all prefixes in scanned groups whose '-dt<..>' segment removed equals base_prefix
            for cand_prefix in list(groups_by_prefix.keys()):
                cand_no_dt = re.sub(r"-dt[0-9]*\.?[0-9]+", "", cand_prefix)
                if cand_no_dt != base_prefix:
                    continue
                dt_val = _parse_dt_from_prefix(cand_prefix)
                if args.dt_min is not None and (dt_val is None or dt_val < float(args.dt_min)):
                    continue
                if args.dt_max is not None and (dt_val is None or dt_val > float(args.dt_max)):
                    continue
                expanded.append((cand_prefix, variant))
        if expanded:
            compare_pairs = expanded

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

    # Determine selected keys (needed for output naming as well)
    selected_keys = None if (args.keys is None or ("all" in args.keys)) else args.keys

    # Build output path: scene-prefix-key-yScale
    out_path = args.save
    if out_path is None:
        if args.iter_decay is not None and selected_keys is not None and len(selected_keys) == 1 and selected_keys[0] == "iter_decay":
            # Default name: <scene>-<prefix>-iter<frame>-iter_decay-y<scale>.png
            base_prefix = compare_pairs[0][0] if len(compare_pairs) > 0 else "iter"
            scene = _extract_scene_from_prefix(base_prefix)
            yscale = "log" if args.ylog else "linear"
            base = f"{scene}-{base_prefix}-iter{int(args.iter_decay)}-iter_decay-y{yscale}"
            out_path = os.path.join(args.dir, base + ".png")
        else:
            # Determine y-scale label per key set; if multiple keys, put 'multi'
            if selected_keys is None or ("all" in (args.keys or [])):
                keys_for_name = [k for k in KNOWN_KEYS if any(k in groups_by_prefix[p][v] for p, v in compare_pairs if p in groups_by_prefix and v in groups_by_prefix[p])]
            else:
                keys_for_name = selected_keys
            # Resolve key part including derived keys
            if len(keys_for_name) == 1:
                key_part = keys_for_name[0]
            elif len(keys_for_name) == 0:
                if selected_keys is not None and len(selected_keys) == 1 and selected_keys[0] == "iter_decay":
                    key_part = "iter_decay"
                else:
                    key_part = "multi"
            else:
                key_part = "multi"
            # y-scale: single global switch
            yscale = "log" if args.ylog else "linear"
            # Optional dt range tag for avg_iter_vs_dt naming
            range_tag = ""
            if (selected_keys is not None and len(selected_keys) == 1 and selected_keys[0] == "avg_iter_vs_dt") or key_part == "avg_iter_vs_dt":
                if args.dt_min is not None or args.dt_max is not None:
                    lo = f"{args.dt_min:.3f}" if args.dt_min is not None else ""
                    hi = f"{args.dt_max:.3f}" if args.dt_max is not None else ""
                    if lo or hi:
                        range_tag = f"-dt{lo}-{hi}"
            # Take first compare pair to form scene/prefix
            base_prefix = compare_pairs[0][0] if len(compare_pairs) > 0 else "scene"
            scene = _extract_scene_from_prefix(base_prefix)
            base = f"{scene}-{base_prefix}-{key_part}{range_tag}-y{yscale}"
            out_path = os.path.join(args.dir, base + ".png")

    # Always use overlay logic (works for single or multiple groups)
    plot_groups_overlay(
        group_list,
        labels,
        keys=selected_keys,
        out_path=out_path,
        show=bool(args.show),
        ylog=bool(args.ylog),
        separate_figs=bool(args.separate_figs),
        start=args.start,
        end=args.end,
        iter_decay_frame=args.iter_decay,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        show_grid=bool(args.show_grid),
        smooth=args.smooth,
        ema=args.ema,
        show_raw=not args.no_raw,
        title_text=args.title,
    )


if __name__ == "__main__":
    main()

