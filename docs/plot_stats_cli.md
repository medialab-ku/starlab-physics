### plot_stats.py CLI Guide

A flexible CLI for plotting experiment statistics saved as .npy arrays. It supports arbitrary overlays across variants and fine-grained selection of metrics and frame ranges.

### File naming convention
- Files are expected to live directly under the stats directory you pass via `--dir` (non-recursive scan).
- Filenames follow:
  - `<prefix>-<variant>-<key>.npy`
  - `prefix`: `dt<dt>-tol<tol>-opt<max>` optionally with `-cfl`, e.g. `dt0.00200-tol4-opt1000-cfl`
- `variant`: one of
  - `iisph`
  - `2014Bender`
  - `ours` (과거 호환: `pcg-ours`, `nopcg-ours`도 읽힘)
  - `key`: one of
    - `elapsed_time_ms`, `opt_iter`, `opt_error`, `pcg_iter`, `pcg_error`

Example filename:
- `pillar-dt0.00200-tol4-opt1000-cfl-warmstart/ours` variant would generate files like:
  - `pillar-dt0.00200-tol4-opt1000-cfl-warmstart-ours-elapsed_time_ms.npy`

### Quick start
```bash
# Compare two variants under the default stats directory
python plot_stats.py \
  -c "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/2014Bender" \
  -k elapsed_time_ms opt_iter opt_error \
  --save data/stats/compare.png

# If your .npy files are inside a subdirectory, point --dir explicitly
python plot_stats.py \
  --dir data/stats/pillar-dt0.00200-tol4-opt1000-cfl-warmstart \
  -c "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/iisph" \
  -k elapsed_time_ms --show
```

### CLI options
- `--dir` (default: `data/stats`)
  - Directory containing `.npy` stats files; non-recursive.
- `--compare`, `-c` (required; multiple)
  - Items to compare, each in the form `prefix/variant`.
  - Example: `pillar-dt0.00200-tol4-opt1000/pcg-ours`.
- `--labels`, `-l` (multiple)
  - Custom legend labels. If omitted, labels are auto-generated.
- `--keys`, `-k` (multiple; default: `all`)
  - Metrics to plot. Valid values: `elapsed_time_ms`, `opt_iter`, `opt_error`, `pcg_iter`, `pcg_error`, `all`.
- `--separate-figs`
  - Save one image per key instead of a single multi-row image.
- `--show`
  - Display figures interactively.
- `--save` (path)
  - Output file path. If omitted, a name is auto-picked based on compared items.
- `--no-logy-errors`
  - Do not use log scale for error series.
- `--start`, `--end` (ints)
  - Timestep frame range (start, end) used for slicing series and computing means. Negative or omitted `end` means until the end.

### What gets plotted
- Keys and titles
  - `elapsed_time_ms` → "Elapsed Time (ms)"
  - `opt_iter` → "Iteration"
  - `opt_error` → "Error"
  - `pcg_iter` → "PCG Iteration"
  - `pcg_error` → "PCG Error"
- Scale
  - Error series (`*error`) are shown on a log scale by default. Disable with `--no-logy-errors`.
- Lines and means
  - All series are rendered with solid lines.
  - For `elapsed_time_ms` and `opt_iter`, a colored dotted horizontal mean line is drawn using values from the selected (start, end) range.

### Overlay behavior
- Overlays are always supported. Supply multiple `--compare` items; each becomes a legend entry.
- Typical variant overlays you might run:
  - `pcg-ours` vs `nopcg-ours` (same prefix)
  - `pcg-ours` across different prefixes (e.g., compare different `tol` or `opt`)
  - `iisph` vs `pcg-ours`
  - `2014Bender` vs `pcg-ours`

### Range slicing semantics
- Timestep-level keys (`elapsed_time_ms`, `opt_iter`, `pcg_iter`):
  - `start`/`end` slice frames directly and set x-axis to the actual frame numbers.
- Iteration-level keys (`opt_error`, `pcg_error`):
  - If `opt_iter` is present, the frame range [start, end) is mapped to an iteration subrange using per-timestep iteration counts.
  - x-axis becomes the iteration index within the selected subrange.

### Output
- Combined figure: one subplot per selected key (default).
- Separate figures: one file per key if `--separate-figs` is passed.
- Auto-generated output filename includes compared items; override with `--save`.

### Examples
```bash
# Compare ours-pcg vs ours-nopcg for Elapsed Time only, frames [200, 1200)
python plot_stats.py \
  -c "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/2014Bender" \
  -k elapsed_time_ms --start 200 --end 1200 --save data/stats/et-compare.png

# Compare across methods for Error and show as separate images per key
python plot_stats.py \
  -c "pillar-dt0.00200-tol4-opt1000/2014Bender" \
     "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/iisph" \
  -k opt_error --separate-figs --show

# Compare the same method but different tolerances (different prefixes)
python plot_stats.py \
  -c "pillar-dt0.00200-tol2-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/ours" \
  -k elapsed_time_ms opt_iter --save data/stats/tol-sweep.png
```

### Notes
- The script does not recurse into subdirectories. Point `--dir` at the folder that directly contains the `.npy` files you want to plot.
- If some keys are missing for a variant, that series is silently skipped for that key.
- Auto labels remove common prefix context when possible; override with `--labels` to control legend text explicitly.
