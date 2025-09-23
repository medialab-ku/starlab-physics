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
  - Example: `pillar-dt0.00200-tol4-opt1000/ours`.
- `--labels`, `-l` (multiple)
  - Custom legend labels. If omitted, labels are auto-generated.
- `--keys`, `-k` (multiple; default: `all`)
  - Metrics to plot. Valid: `elapsed_time_ms`, `opt_iter`, `opt_error`, `pcg_iter`, `pcg_error`, `iter_decay`, `avg_iter_vs_dt`, `all`.
  - `iter_decay`는 특정 프레임의 최적화 에러 감소 곡선을 그리는 파생 키입니다. `--iter-decay`와 함께 사용하세요.
  - `avg_iter_vs_dt`는 각 비교 항목의 Δt에 대해 평균 `opt_iter`(선택 범위 `--start/--end` 반영)를 계산해 Δt-평균 이터레이션 곡선을 그립니다.
- `--separate-figs`
  - Save one image per key instead of a single multi-row image.
- `--show`
  - Display figures interactively.
- `--save` (path)
  - Output file path. If omitted, a name is auto-picked based on compared items.
- `--ylog`
  - 모든 시리즈에 대해 Y축 로그 스케일을 적용합니다. 미지정 시 선형 스케일.
- `--start`, `--end` (ints)
  - Timestep frame range (start, end) used for slicing series and computing means. Negative or omitted `end` means until the end.
- `--iter-decay` (int)
  - `-k iter_decay`와 함께 사용합니다. 해당 프레임 인덱스의 per-iteration `opt_error` 감소를 플로팅합니다.
- `--smooth` (int, default: 0)
  - Moving average window size. 0 or 1 disables smoothing (default off).
- `--ema` (float, default: None)
  - Exponential moving average alpha in (0,1). If set, takes precedence over `--smooth`.
- `--no-raw`
  - When smoothing is enabled, hide the faint raw line background.
- `--dt-min`, `--dt-max` (floats)
  - `-k avg_iter_vs_dt`와 함께 사용합니다. Δt 범위를 제한합니다. 파일명에도 범위 태그가 포함됩니다.

#### avg_iter_vs_dt 전용 간소화 입력
- `-k avg_iter_vs_dt`일 때는 `-c`에 장황한 prefix를 모두 나열하지 않고, `scene-tol<tol>-opt<max>/variant` 형식의 "베이스 prefix"만 주면 됩니다.
- 스크립트가 동일한 `scene-tol-opt` 조합의 모든 `-dt<...>` prefix를 자동으로 검색해 `--dt-min/--dt-max` 범위에 해당하는 것만 확장합니다.

예시:
```bash
python plot_stats.py \
  -c "pillar-tol4-opt1000/ours" "pillar-tol4-opt1000/2014Bender" \
  -k avg_iter_vs_dt --dt-min 0.001 --dt-max 0.003 \
  --save data/stats/pillar-avg_iter_vs_dt.png
```
위 명령은 내부적으로 `pillar-dt0.00100-tol4-opt1000`, `pillar-dt0.00200-tol4-opt1000`, `pillar-dt0.00300-tol4-opt1000` 등으로 자동 확장합니다.

### What gets plotted
- Keys and titles
  - `elapsed_time_ms` → "Elapsed Time (ms)"
  - `opt_iter` → "Iteration"
  - `opt_error` → "Error"
  - `pcg_iter` → "PCG Iteration"
  - `pcg_error` → "PCG Error"
  - `iter_decay` → "Error (timestep <frame>)" (프레임은 `--iter-decay`로 지정)
  - `avg_iter_vs_dt` → "avg. iterations" vs "Δt"
- Scale
  - `--ylog` 제공 시 로그 스케일, 아니면 선형 스케일.
- Lines and means
  - All series are rendered with solid lines.
  - For `elapsed_time_ms` and `opt_iter`, a colored dotted horizontal mean line is drawn using values from the selected (start, end) range.
 - Smoothing (optional)
   - If `--ema` or `--smooth` is provided, a smoothed trend is plotted; the original series is shown faintly unless `--no-raw` is set.

### Overlay behavior
- Overlays are always supported. Supply multiple `--compare` items; each becomes a legend entry.
- Typical variant overlays you might run:
  - `pcg-ours` vs `nopcg-ours` (same prefix)
  - `pcg-ours` across different prefixes (e.g., compare different `tol` or `opt`)
  - `iisph` vs `pcg-ours`
  - `2014Bender` vs `pcg-ours`

### Range slicing semantics
- Timestep-level keys (`elapsed_time_ms`, `opt_iter`, `pcg_iter`):
  - `start`/`end` slice frames directly.
  - The x-axis is rendered in seconds by parsing `dt` from the filename prefix (e.g., `dt0.00200`).
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
  -k opt_error --ylog --separate-figs --show

# Compare the same method but different tolerances (different prefixes)
python plot_stats.py \
  -c "pillar-dt0.00200-tol2-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/ours" \
  -k elapsed_time_ms opt_iter --save data/stats/tol-sweep.png

# Plot optimizer error decay at a specific timestep across variants
python plot_stats.py \
  -c "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/2014Bender" \
  -k iter_decay --iter-decay 320 --ylog --save data/stats/pillar-iter320-iter_decay.png

# Average iterations vs Δt for ours vs Bender in [0.001, 0.003]
python plot_stats.py \
  -c "pillar-dt0.00100-tol4-opt1000/ours" \
     "pillar-dt0.00200-tol4-opt1000/ours" \
     "pillar-dt0.00300-tol4-opt1000/ours" \
     "pillar-dt0.00100-tol4-opt1000/2014Bender" \
     "pillar-dt0.00200-tol4-opt1000/2014Bender" \
     "pillar-dt0.00300-tol4-opt1000/2014Bender" \
  -k avg_iter_vs_dt --dt-min 0.001 --dt-max 0.003 --save data/stats/pillar-avg_iter_vs_dt.png
```

### Notes
- The script does not recurse into subdirectories. Point `--dir` at the folder that directly contains the `.npy` files you want to plot.
- If some keys are missing for a variant, that series is silently skipped for that key.
- Auto labels remove common prefix context when possible; override with `--labels` to control legend text explicitly.
- Legends unify method names to "Ours" and "Bender et al." automatically.
- Figures default to single-column width (~3.4 inches) suitable for 2-column papers.
