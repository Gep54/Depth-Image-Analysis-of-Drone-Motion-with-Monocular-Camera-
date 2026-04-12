# CV_see_distance

A small Python app for image processing and feature matching, aimed at monocular depth / motion analysis (e.g. drone imagery).

## Features

- Image preprocessing
- Feature extraction and matching (single pair or full sequence)
- Optional undistortion using calibrated intrinsics
- Two-view sparse reconstruction (essential matrix, triangulation, PLY/NPZ export)
- Incremental multi-view SfM (PnP + triangulation) with optional bundle adjustment
- Planar **(x, y)** priors on camera centers (optional sync table); **z** is never constrained
- **Step 4** map export: camera trajectory CSV, sparse points CSV, per-frame reprojection report, optional PLY
- **Step 5** robust global refinement: re-run joint BA with **soft_l1** / **huber** (etc.) on a saved NPZ
- **`reconstruct`** — single command: frames + intrinsics + distortion → full chain → **3D point cloud (PLY)**
- Camera calibration (chessboard helper)
- Result visualization and export

## Requirements

- Python 3.10+
- `scipy` — required for **step 3 bundle adjustment** (default); omit if you always pass `--no-bundle`

## Setup

1. Create a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run commands from the `SeeDistanceMain` folder (so imports resolve):

```bash
cd SeeDistanceMain
python cli.py <subcommand> ...
```

### One-command reconstruction (`reconstruct`)

From a **folder of frames** (lexicographically ordered), **camera intrinsics `K`**, and optional **distortion** coefficients, run the full pipeline and get a **colored sparse point cloud** (PLY):

1. *(Optional)* Step 1 — consecutive-pair match statistics CSV (`--match-stats`).
2. Steps 2–3 — two-view seed, incremental PnP + triangulation, first bundle adjustment (with optional planar **x,y** sync priors).
3. Step 5 — robust refinement (`soft_l1` by default; use `--no-refine` to skip).
4. Writes **`--output-ply`** (required). Optionally **`--output-npz`**, and **`--export-dir`** for step-4 CSV diagnostics.

**Minimal example**

```bash
cd SeeDistanceMain
python cli.py reconstruct \
  --frames-dir path/to/frames \
  --intrinsics path/to/K.npy \
  --output-ply map.ply
```

With **lens distortion** and saved bundle for later:

```bash
python cli.py reconstruct \
  --frames-dir path/to/frames \
  --intrinsics path/to/K.npy \
  --dist path/to/dist.npy \
  --output-ply map.ply \
  --output-npz map.npz
```

With **sync** (planar camera centers) and **export** folder (cameras/points/reprojection CSV):

```bash
python cli.py reconstruct \
  --frames-dir path/to/frames \
  --intrinsics path/to/K.npy \
  --dist path/to/dist.npy \
  --sync path/to/sync.csv \
  --output-ply map.ply \
  --export-dir path/to/report_out
```

Requires **`scipy`** unless you pass **`--no-bundle`** and **`--no-refine`** (not recommended).

### Step 1 — Multi-image geometric foundation

This step loads an **ordered** image sequence, applies **camera intrinsics** (and optional **distortion**), **undistorts** frames before detection (unless disabled), then runs **consecutive pairwise** feature matching and reports **statistics** (match counts and descriptor distances). It is the intended starting point for a larger SfM-style pipeline: consistent pinhole geometry and per-pair correspondence strength.

**Inputs**

- **Frames directory**: all `png` / `jpg` / … files are sorted by **filename** and processed in that order. Name files so lexical order equals time order (e.g. `frame_0001.jpg`).
- **Intrinsics `K`**: 3×3 matrix in `.npy`, `.txt`, or `.csv` (same as `data.load_intrinsics`).
- **Distortion** (optional): 1D coefficients in `.npy` or `.txt` (same layout as OpenCV calibration output). If omitted, distortion is treated as zero.

**CLI example**

```bash
cd SeeDistanceMain
python cli.py sequence-match --frames-dir path/to/frames --intrinsics path/to/K.npy --output match_stats.csv
```

With a distortion file and undistortion enabled (default):

```bash
python cli.py sequence-match --frames-dir path/to/frames --intrinsics path/to/K.npy --dist path/to/dist.npy --output match_stats.csv
```

Match on **raw** images (no `cv.undistort`):

```bash
python cli.py sequence-match --frames-dir path/to/frames --intrinsics path/to/K.npy --no-undistort --output match_stats.csv
```

**Statistics CSV columns**

| Column | Meaning |
|--------|---------|
| `pair_index` | Index of the pair `(i, i+1)` in the sorted frame list |
| `image_a`, `image_b` | Filenames of the two frames |
| `n_keypoints_a`, `n_keypoints_b` | Detected keypoints after preprocessing |
| `n_matches` | Number of mutual nearest-neighbour matches |
| `mean_match_distance`, `median_match_distance` | Match score statistics (Hamming for ORB, L2 for SIFT/KAZE) |

**API**

- `data.load_frames`, `data.load_intrinsics`, `data.load_distortion`, `data.undistort_image_bgr`
- `sequence_match.consecutive_pair_match_stats`

### Step 2 — Two-view reconstruction

From **two undistorted views** (same resolution), the pipeline **matches** features, estimates the **essential matrix** (RANSAC), recovers **relative rotation `R` and translation direction `t`** (OpenCV `recoverPose`), **triangulates** inlier correspondences, filters points **in front of** both cameras, and reports **mean reprojection error** in pixels. Translation and 3D coordinates are known **only up to scale** (classic monocular ambiguity).

**Outputs**

- **`--output-ply`**: ASCII PLY with optional vertex colors (sampled from image 1).
- **`--output-npz`**: `points_3d`, `colors_bgr`, `R`, `t`, `rvec`, `E`, `K`, `mean_reproj_error_px`, `n_inliers`, `inlier_kp_idx0`, `inlier_kp_idx1`, `inlier_uv0`, `inlier_uv1`.

**Coordinate frame**

- World frame = **first camera**; `P1 = K [I|0]`, `P2 = K [R|t]`.

**CLI examples**

Two explicit files:

```bash
cd SeeDistanceMain
python cli.py two-view --image1 path/a.jpg --image2 path/b.jpg --intrinsics path/K.npy \
  --output-ply sparse.ply --output-npz two_view.npz
```

Consecutive frames from a folder (same ordering as step 1; default pair is `0` → frames 0 and 1):

```bash
python cli.py two-view --frames-dir path/to/frames --intrinsics path/K.npy --pair-index 0 \
  --output-ply sparse.ply
```

With distortion correction (default: undistort before detection):

```bash
python cli.py two-view --frames-dir path/to/frames --intrinsics path/K.npy --dist path/dist.npy \
  --output-npz two_view.npz
```

Tuning RANSAC (epipolar threshold in pixels, confidence):

```bash
python cli.py two-view --image1 a.jpg --image2 b.jpg --intrinsics K.npy \
  --ransac-threshold 0.8 --ransac-prob 0.999
```

**Defaults**

- **`--feature`** defaults to **SIFT** for more stable two-view geometry than ORB on raw scenes.
- **`--preprocess`** defaults to **`none`** (edges are optional, as in step 1).

**API**

- `two_view.reconstruct_two_view`, `two_view.write_ply_ascii`, `two_view.save_two_view_npz`

### Step 3 — Incremental multi-view reconstruction and (x, y)-only sync priors

**Pipeline**

1. **Seed** — same two-view model as step 2 on the first two frames (`frames[0]`, `frames[1]`). World origin = **camera 0** (fixed identity pose in bundle adjustment).
2. **Register** each next frame with **consecutive** matching to the previous frame, **3D–2D** correspondences, and **`solvePnPRansac`**.
3. **Triangulate** new points from matches whose previous-frame keypoint was not yet in the map (stereo between `i-1` and `i`).
4. **Bundle adjustment** (optional, default **on**): joint nonlinear least squares on all **reprojection** residuals plus **weighted priors on camera center `(C_x, C_y)`** from your sync table. **There is no prior on `C_z`** — altitude / vertical motion stays a free DOF per camera.

**Sync table**

- Loaded with `data.load_sync` (`.csv` or `.tsv`).
- Required columns: **`image`** (filename, path, or basename), **`x`**, **`y`** in the same horizontal plane you want to anchor (e.g. local East–North).
- Optional: **`weight`** (higher = stronger) or **`sigma`** (std dev in the plane; weight = `1/sigma²`). Otherwise rows use `--sync-default-weight`.

**Gauge / first frame**

- Camera 0 is fixed at identity. Its world center is **`C = 0`**, so a sync row for the first image with nonzero `(x, y)` fights that gauge. Use **`(0, 0)`** for the first frame, omit it, or set its weight to **0**.

**Scale / alignment**

- Monocular structure is still **ambiguous up to similarity** before external data. Sync `(x, y)` should live in a **compatible** planar chart (same units and rough alignment as reconstruction), or use **modest** weights so BA can reconcile vision vs odometry.

**CLI examples**

```bash
cd SeeDistanceMain
python cli.py incremental-sfm --frames-dir path/to/frames --intrinsics path/K.npy \
  --output-ply map.ply --output-npz map.npz
```

With distortion and planar sync priors (x, y only in BA):

```bash
python cli.py incremental-sfm --frames-dir path/to/frames --intrinsics path/K.npy \
  --dist path/dist.npy --sync path/sync.csv --output-ply map.ply --output-npz map.npz
```

Incremental geometry only (no scipy BA):

```bash
python cli.py incremental-sfm --frames-dir path/to/frames --intrinsics path/K.npy --no-bundle \
  --output-npz map.npz
```

**API**

- `incremental_sfm.run_incremental_sfm`, `incremental_sfm.save_incremental_npz`
- `sync_priors.sync_xy_weight_per_frame`
- `bundle_adjust_xy.bundle_adjust_multiview_xy_priors`

### Step 4 — Map export and diagnostics

After **step 3**, the NPZ bundle contains poses, 3D points, colors, observations, and `K`. Step 4 turns that into **reviewable artifacts** (no new optimization):

| Output | Description |
|--------|-------------|
| `cameras.csv` | Per frame: **world camera center** `(center_x, center_y, center_z)` with `C = −Rᵀt`, plus `rvec` / `tvec` |
| `points.csv` | Sparse landmarks: `point_id`, `x, y, z`, BGR color |
| `reprojection_by_frame.csv` | Observation count and **RMS / mean / max** reprojection error (pixels) per frame |
| `points.ply` | Same cloud as step 3 PLY (optional; use `--no-ply` to skip) |
| `observations.csv` | Optional: every observation with `reproj_error_px` (`--observations-csv`; can be large) |

**CLI**

```bash
cd SeeDistanceMain
python cli.py export-map --input path/to/map.npz --output-dir path/to/export_folder
```

Skip PLY, add per-observation CSV:

```bash
python cli.py export-map --input map.npz --output-dir out --no-ply --observations-csv
```

**API**

- `incremental_sfm.load_incremental_npz`
- `sfm_export.export_incremental_map`, `sfm_export.reprojection_summary_by_frame`, `sfm_export.camera_centers_trajectory`

### Step 5 — Robust global refinement

Step 3’s first bundle adjustment uses a **linear** (L2) loss. Step 5 **reloads** an NPZ and runs **another** full joint optimization with a **robust** loss from SciPy (`soft_l1`, `huber`, `cauchy`, `arctan`, or `linear` again). Large reprojection errors (bad matches) or occasional bad sync rows are **down-weighted** relative to step 3.

**Caveat:** `scipy.optimize.least_squares` applies the same `loss` to **every** residual, including **planar (x, y) priors**. Keep sync weights moderate, or use `linear` loss if priors must stay strictly quadratic.

**Typical workflow**

```bash
python cli.py incremental-sfm ... --output-npz map.npz
python cli.py refine-map --input map.npz --output map_refined.npz
python cli.py export-map --input map_refined.npz --output-dir out
```

**CLI examples**

Default robust pass (`soft_l1`, `f_scale=1.5`):

```bash
cd SeeDistanceMain
python cli.py refine-map --input map.npz --output map_refined.npz
```

With sync priors during refinement and Huber loss:

```bash
python cli.py refine-map --input map.npz --output map_refined.npz --sync sync.csv --loss huber --f-scale 2.0
```

**API**

- `refine_map.refine_incremental_bundle`
- `bundle_adjust_xy.bundle_adjust_multiview_xy_priors` (`loss`, `f_scale` arguments)

### Other commands

- `python cli.py reconstruct ...` — **end-to-end** pipeline (see above)
- `python cli.py match --image1 ... --image2 ...` — match two images
- `python cli.py calibrate` — chessboard calibration (paths are configured inside `camera_calibration.py`)

## Project structure

- `SeeDistanceMain/` — main application code (`cli.py`, `data.py`, `mymatch.py`, `sequence_match.py`, `two_view.py`, `incremental_sfm.py`, `bundle_adjust_xy.py`, `sync_priors.py`, `sfm_export.py`, `refine_map.py`, …)
- `JupyterNotebooksPlayground/` — notebooks and generated outputs

## Notes

Generated images, cache files, and virtual environments should not be committed.
