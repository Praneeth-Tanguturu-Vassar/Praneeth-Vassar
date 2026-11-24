"""
Mango vs Non-Mango – End-to-End Pipeline
=========================================
Steps
  1. Load rasters (RGB + 128-band embeddings + GT mask) and tree bboxes
  2. Extract features  : 128 embedding mean per bbox  +  3 RGB at bbox centre
  3. Feature selection : MDI-based RF pass -> keep features covering
                         FI_THRESHOLD of total importance
  4. Spatial-block CV  : evaluate RF + XGB ensemble on the selected features
  5. Final training    : fit on all labelled data, save model + feature list
  6. Inference         : classify every bbox -> bboxes_classified.gpkg

Usage
-----
  Just run:  python run_pipeline.py
  Edit the CONFIG block below to change paths / settings.
"""

# =============================================================================
# CONFIG  –  edit this block to change paths and settings
# =============================================================================

# ---- Input paths ------------------------------------------------------------
BBOXES_PATH = "/home/vassarlabs/Downloads/AP/Blocks/pulicherla/Tree_geojson/Tree.shp"
RGB_PATH    = "/home/vassarlabs/Downloads/AP/Blocks/pulicherla/gl_data/Pulicherla_rgb"
EMB_PATH    = "/home/vassarlabs/Downloads/AP/Blocks/pulicherla/gl_data/pulicherla_block_embeddings"
GT_PATH     = "/home/vassarlabs/Documents/vs-code/GeoAI/Embeddings_RD/Mango_Non_Mango/Mango_NonMango.tif"

# ---- Output path ------------------------------------------------------------
OUT_DIR     = "/home/vassarlabs/Documents/vs-code/GeoAI/Embeddings_RD/Mango_Non_Mango/output/AP/pulicherla_block"

# ---- Feature selection ------------------------------------------------------
FEATURE_SELECTION = True    # True = select important features | False = use all 131
FI_THRESHOLD      = 0.95    # cumulative MDI importance to cover (only used when FEATURE_SELECTION=True)

# ---- Training ---------------------------------------------------------------
FOLDS        = 5
MAX_PER_CLASS = 25000       # max bboxes per class used for training
LABEL_POOL   = 300000       # bboxes to label from GT before sampling
RANDOM_RATIO = 1.0          # target non-mango:mango ratio
MIN_PURITY   = 0.6
N_JOBS       = 8
SEED         = 42

# ---- Inference --------------------------------------------------------------
PREDICT_ALL  = True         # always run inference after training
THRESHOLD    = 0.5          # prob_mango >= THRESHOLD -> mango

# ---- Performance ------------------------------------------------------------
CENTER_WIN   = 1
BLOCK_PX     = 50
WORKERS      = 4            # parallel workers for feature extraction
CHUNK        = 2000
MAX_EMB_GB   = 8.0

# =============================================================================
# END CONFIG
# =============================================================================

import json
import multiprocessing as mp
import os
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform as warp_transform
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ.setdefault("GDAL_CACHEMAX", "1024")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMB_COLS  = [f"e{i}" for i in range(128)]
RGB_COLS  = ["r", "g", "b"]
FEAT_COLS = EMB_COLS + RGB_COLS          # 131 features total

G = {}   # process-shared state (populated before forking workers)

STRIP = 1024   # row-strip width used in inference RGB sampling


# ============================================================================
# STEP 0 – helpers: rasters, tiles, VRT
# ============================================================================

def open_raster_or_tiles(path, out_dir, name):
    """Single GeoTIFF -> returned as-is; folder of tiles -> VRT mosaic.
    Corrupted / unreadable tiles are skipped with a warning."""
    if not os.path.isdir(path):
        return path

    all_tiles = sorted(
        os.path.join(d, f)
        for d, _, fs in os.walk(path)
        for f in fs if f.lower().endswith((".tif", ".tiff"))
    )
    if not all_tiles:
        raise FileNotFoundError(f"no .tif tiles found in {path}")

    print(f"\n{name}: scanning {len(all_tiles)} tiles for validity …", flush=True)
    good_tiles, bad_tiles, crs_set = [], [], set()
    for t in tqdm(all_tiles, desc=f"{name}: check tiles"):
        try:
            with rasterio.open(t) as s:
                _ = s.meta          # force a full header read
                crs_set.add(str(s.crs))
                good_tiles.append(t)
        except Exception as e:
            print(f"  [SKIP] not a valid raster -> {os.path.basename(t)}  ({e})",
                  flush=True)
            bad_tiles.append(t)

    print(f"{name}: {len(good_tiles)} valid tiles, {len(bad_tiles)} skipped",
          flush=True)
    if not good_tiles:
        raise SystemExit(f"{name}: no readable tiles found in {path}")

    if len(crs_set) > 1:
        raise ValueError(
            f"tiles in {path} have mixed CRS: {crs_set}. Reproject first."
        )

    os.makedirs(out_dir, exist_ok=True)
    vrt = os.path.join(out_dir, f"{name}_mosaic.vrt")
    lst = os.path.join(out_dir, f"{name}_tiles.txt")
    with open(lst, "w") as f:
        f.write("\n".join(good_tiles))
    print(f"{name}: mosaicking {len(good_tiles)} valid tiles -> {vrt}", flush=True)
    try:
        from osgeo import gdal
        gdal.BuildVRT(vrt, good_tiles)
    except ImportError:
        subprocess.run(["gdalbuildvrt", "-input_file_list", lst, vrt], check=True)
    return vrt


def list_tiles(path):
    """Return [path] for a file, or sorted tile list for a folder."""
    if not os.path.isdir(path):
        return [path]
    tiles = sorted(
        os.path.join(d, f)
        for d, _, fs in os.walk(path)
        for f in fs if f.lower().endswith((".tif", ".tiff"))
    )
    if not tiles:
        raise FileNotFoundError(f"no .tif tiles found in {path}")
    return tiles


# ============================================================================
# STEP 1 – feature extraction helpers
# ============================================================================

def window_for_bounds(src, bounds):
    w = from_bounds(*bounds, transform=src.transform)
    c0 = int(np.floor(w.col_off));  r0 = int(np.floor(w.row_off))
    c1 = int(np.ceil(w.col_off + w.width))
    r1 = int(np.ceil(w.row_off + w.height))
    return Window(c0, r0, max(c1 - c0, 1), max(r1 - r0, 1))


def clipped_window(src, win):
    c0 = max(win.col_off, 0);  r0 = max(win.row_off, 0)
    c1 = min(win.col_off + win.width,  src.width)
    r1 = min(win.row_off + win.height, src.height)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def label_from_gt(src, bounds, min_purity):
    """Return 1=mango, 0=non-mango, -1=ambiguous/no GT."""
    win = clipped_window(src, window_for_bounds(src, bounds))
    if win is None:
        return -1
    a = src.read(1, window=win)
    n1, n2 = int((a == 1).sum()), int((a == 2).sum())
    tot = n1 + n2
    if tot == 0:
        return -1
    if n1 / tot >= min_purity:
        return 1
    if n2 / tot >= min_purity:
        return 0
    return -1


def load_emb_array(emb, max_gb):
    gb = emb.count * emb.height * emb.width * 4 / 1e9
    if gb > max_gb:
        print(f"embeddings need {gb:.1f} GB > --max-emb-gb; using windowed reads")
        return None
    print(f"loading embeddings into RAM ({gb:.1f} GB) …")
    arr = np.empty((emb.count, emb.height, emb.width), np.float32)
    for r in tqdm(range(0, emb.height, 1024), desc="load emb"):
        h = min(1024, emb.height - r)
        a = emb.read(window=Window(0, r, emb.width, h)).astype(np.float32)
        if emb.nodata is not None and not np.isnan(emb.nodata):
            a[a == emb.nodata] = np.nan
        arr[:, r:r + h] = a
    return arr


def bounds_in(gdf, crs):
    return gdf.to_crs(crs).bounds.values   # (N, 4): minx miny maxx maxy


def centre_xy(gdf, crs):
    b = bounds_in(gdf, crs)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def bounds_to_rc(b, t):
    """Vectorised bbox bounds -> pixel window (r0, r1, c0, c1)."""
    c0 = np.floor((b[:, 0] - t.c) / t.a)
    c1 = np.ceil( (b[:, 2] - t.c) / t.a)
    r0 = np.floor((b[:, 3] - t.f) / t.e)
    r1 = np.ceil( (b[:, 1] - t.f) / t.e)
    c1 = np.maximum(c1, c0 + 1)
    r1 = np.maximum(r1, r0 + 1)
    return np.c_[r0, r1, c0, c1].astype(np.int64)


# ---- parallel worker functions (use G dict, inherited via fork) ------------

def _init_worker():
    G["rgb"] = rasterio.open(G["rgb_path"])
    G["gt"]  = rasterio.open(G["gt_path"])
    if G["emb_arr"] is None:
        G["emb"] = rasterio.open(G["emb_path"])


def _label_chunk(c):
    s, e = c
    b = G["gt_bounds"]
    return np.array(
        [label_from_gt(G["gt"], b[k], G["min_purity"]) for k in range(s, e)],
        dtype=np.int8
    )


def _emb_mean_item(k):
    r0, r1, c0, c1 = G["emb_rc"][k]
    if G["emb_arr"] is not None:
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, G["emb_h"]), min(c1, G["emb_w"])
        if r1 <= r0 or c1 <= c0:
            return None
        a = G["emb_arr"][:, r0:r1, c0:c1]
    else:
        w = clipped_window(G["emb"], Window(c0, r0, c1 - c0, r1 - r0))
        if w is None:
            return None
        a = G["emb"].read(window=w).astype(np.float32)
        nd = G["emb"].nodata
        if nd is not None and not np.isnan(nd):
            a[a == nd] = np.nan
    v = np.nanmean(a.reshape(a.shape[0], -1), axis=1)
    if np.isnan(v).any() or not v.any():
        return None
    return v.astype(np.float32)


def center_rgb(src, x, y, win=1):
    r, c = src.index(x, y)
    h = win // 2
    w = clipped_window(src, Window(c - h, r - h, win, win))
    if w is None:
        return None
    a = np.ma.masked_invalid(src.read([1, 2, 3], window=w, masked=True))
    a = a.reshape(3, -1)
    if a.mask.all():
        return None
    v = np.ma.filled(np.ma.median(a, axis=1), np.nan)
    return None if np.isnan(v).any() else v.astype(np.float32)


def _feat_chunk(c):
    s, e = c
    F  = np.full((e - s, 131), np.nan, np.float32)
    xy = G["rgb_xy"]
    for k in range(s, e):
        rv = center_rgb(G["rgb"], xy[k, 0], xy[k, 1], G["center_win"])
        if rv is None:
            continue
        ev = _emb_mean_item(k)
        if ev is None:
            continue
        F[k - s, :128] = ev
        F[k - s, 128:] = rv
    return F


def run_parallel(fn, n_items, workers, chunk, desc):
    chunks = [(s, min(s + chunk, n_items)) for s in range(0, n_items, chunk)]
    out = [None] * len(chunks)
    bar = tqdm(total=n_items, desc=desc, unit="obj")
    if workers <= 1:
        _init_worker()
        for i, c in enumerate(chunks):
            out[i] = fn(c)
            bar.update(c[1] - c[0])
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"),
                                 initializer=_init_worker) as ex:
            futs = {ex.submit(fn, c): i for i, c in enumerate(chunks)}
            for f in as_completed(futs):
                i = futs[f]
                out[i] = f.result()
                bar.update(chunks[i][1] - chunks[i][0])
    bar.close()
    return out


def extract_features(rgb_xy, emb_b, args, desc):
    G["rgb_xy"] = rgb_xy
    G["emb_rc"] = bounds_to_rc(emb_b, G["emb_tf"])
    F = np.vstack(run_parallel(_feat_chunk, len(rgb_xy), args.workers,
                               args.chunk, desc))
    rc    = G["emb_rc"]
    group = (rc[:, 0] // args.block_px) * 100_000 + (rc[:, 2] // args.block_px)
    return F, group


def sample_nonmango_points(gt, n, seed, per_block=5000):
    rng = np.random.default_rng(seed)
    pts = []
    for _, win in tqdm(list(gt.block_windows(1)), desc="scan GT for class 2"):
        a  = gt.read(1, window=win)
        rr, cc = np.nonzero(a == 2)
        if len(rr) == 0:
            continue
        k   = min(per_block, len(rr))
        idx = rng.choice(len(rr), k, replace=False)
        pts.append(np.c_[rr[idx] + win.row_off, cc[idx] + win.col_off])
    if not pts:
        return np.array([]), np.array([])
    pts = np.vstack(pts)
    if len(pts) > n:
        pts = pts[rng.choice(len(pts), n, replace=False)]
    rows = pts[:, 0] + rng.random(len(pts))
    cols = pts[:, 1] + rng.random(len(pts))
    xs, ys = gt.transform * (cols, rows)
    return np.asarray(xs), np.asarray(ys)


# ============================================================================
# STEP 2 – feature selection (MDI)
# ============================================================================

def select_important_features(X, y, feat_cols, threshold=0.90, n_jobs=4, seed=42):
    """Single RF fit -> MDI importance -> keep features that cumulatively cover
    `threshold` fraction of total importance.

    Returns
    -------
    sel_cols : list[str]  feature names (kept in original column order)
    sel_idx  : list[int]  corresponding column indices into feat_cols
    imp_df   : pd.DataFrame  all 131 features ranked by importance
    """
    print(f"\n[Step 2] Feature selection (MDI threshold={threshold:.0%}) …",
          flush=True)
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=n_jobs, random_state=seed,
    )
    rf.fit(X, y)

    imp    = rf.feature_importances_
    order  = np.argsort(imp)[::-1]            # highest first
    cum    = np.cumsum(imp[order]) / imp.sum()
    n_keep = int(np.searchsorted(cum, threshold)) + 1
    sel_idx  = sorted(order[:n_keep].tolist())
    sel_cols = [feat_cols[i] for i in sel_idx]

    imp_df = pd.DataFrame({
        "rank":    np.arange(1, len(feat_cols) + 1),
        "feature": [feat_cols[i] for i in order],
        "group":   ["embedding" if feat_cols[i].startswith("e") else "rgb"
                    for i in order],
        "mdi":     imp[order],
        "mdi_pct": 100 * imp[order] / imp.sum(),
        "cum_pct": 100 * cum,
    })

    # count embedding vs RGB in the selected set
    n_emb = sum(1 for c in sel_cols if c.startswith("e"))
    n_rgb = sum(1 for c in sel_cols if c in ("r", "g", "b"))

    print(f"\n  ┌─ Feature Selection Result ──────────────────────────────")
    print(f"  │  Total features      : {len(feat_cols)}")
    print(f"  │  Selected features   : {n_keep}  "
          f"(covers {threshold * 100:.0f}% MDI importance)")
    print(f"  │    Embedding bands   : {n_emb} / 128")
    print(f"  │    RGB bands         : {n_rgb} / 3")
    print(f"  │")
    print(f"  │  Top-10 by importance:")
    top10 = imp_df.head(10)
    for _, row in top10.iterrows():
        print(f"  │    rank {int(row['rank']):>3}  {row['feature']:<8}  "
              f"MDI={row['mdi']:.5f}  ({row['mdi_pct']:.2f}%)  [{row['group']}]")
    print(f"  │")
    print(f"  │  All {n_keep} selected features (sorted by index):")
    # print in rows of 10
    for i in range(0, len(sel_cols), 10):
        print(f"  │    {', '.join(sel_cols[i:i+10])}")
    print(f"  └─────────────────────────────────────────────────────────",
          flush=True)
    return sel_cols, sel_idx, imp_df


# ============================================================================
# STEP 3/4 – model definition + evaluation
# ============================================================================

def make_ensemble(scale_pos_weight=1.0, n_jobs=1, random_state=42):
    rf_clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=n_jobs, random_state=random_state,
    )
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", tree_method="hist", n_jobs=n_jobs,
        random_state=random_state,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[("rf", rf_clf), ("xgb", xgb_clf)],
            voting="soft", n_jobs=1,
        )),
    ])


def report(y, prob, title, thr=0.5):
    pred = (prob >= thr).astype(int)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1],
                                                 zero_division=0)
    mp_, mr_, mf_, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0)
    print(f"\n=== {title} ===")
    print(f"{'class':<11}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>10}")
    for i, nm in enumerate(["non-mango", "mango"]):
        print(f"{nm:<11}{p[i]:>10.3f}{r[i]:>9.3f}{f[i]:>9.3f}{int(s[i]):>10d}")
    print(f"{'macro avg':<11}{mp_:>10.3f}{mr_:>9.3f}{mf_:>9.3f}")
    print("confusion matrix [rows=true, cols=pred] (non-mango=0, mango=1)\n",
          confusion_matrix(y, pred, labels=[0, 1]))
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    print(f"ROC-AUC: {auc:.4f}")
    return dict(mango_precision=float(p[1]),    mango_recall=float(r[1]),
                mango_f1=float(f[1]),           nonmango_precision=float(p[0]),
                nonmango_recall=float(r[0]),    nonmango_f1=float(f[0]),
                macro_f1=float(mf_),            roc_auc=float(auc),
                n=int(len(y)))


# ============================================================================
# STEP 6 – inference helpers (single-pass RGB, vectorised emb)
# ============================================================================

def emb_features_infer(rc, arr, src, h, w):
    """Embedding mean over each bbox -> (K, 128); NaN where invalid."""
    K = len(rc)
    F = np.full((K, 128), np.nan, np.float32)
    single = ((rc[:, 1] - rc[:, 0]) == 1) & ((rc[:, 3] - rc[:, 2]) == 1)
    inb    = ((rc[:, 0] >= 0) & (rc[:, 0] < h) &
              (rc[:, 2] >= 0) & (rc[:, 2] < w))
    m = single & inb
    if arr is not None and m.any():
        F[m] = arr[:, rc[m, 0], rc[m, 2]].T
        todo = np.nonzero(~m & ~(single & ~inb))[0]
    else:
        todo = np.nonzero(~(single & ~inb))[0]
    for k in todo:
        r0, r1, c0, c1 = rc[k]
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, h), min(c1, w)
        if r1 <= r0 or c1 <= c0:
            continue
        if arr is not None:
            a = arr[:, r0:r1, c0:c1]
        else:
            a = src.read(window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
            if src.nodata is not None and not np.isnan(src.nodata):
                a[a == src.nodata] = np.nan
        F[k] = np.nanmean(a.reshape(a.shape[0], -1), axis=1)
    bad = np.isnan(F).any(axis=1) | ~F.any(axis=1)
    F[bad] = np.nan
    return F


def sample_rgb_infer(tiles, xy, win):
    """RGB at bbox centre from all tiles; single row-strip pass per tile."""
    out  = np.full((len(xy), 3), np.nan, np.float32)
    done = np.zeros(len(xy), bool)
    half = win // 2
    offs = [(dy, dx) for dy in range(-half, half + 1)
                     for dx in range(-half, half + 1)]
    for path in tqdm(tiles, desc="RGB tiles (inference)"):
        with rasterio.open(path) as src:
            l, b, r, t = src.bounds
            m   = (~done & (xy[:, 0] >= l) & (xy[:, 0] < r)
                         & (xy[:, 1] >  b) & (xy[:, 1] <= t))
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            tf, H, W = src.transform, src.height, src.width
            rows = np.clip(
                np.floor((xy[idx, 1] - tf.f) / tf.e).astype(np.int64), 0, H - 1)
            cols = np.clip(
                np.floor((xy[idx, 0] - tf.c) / tf.a).astype(np.int64), 0, W - 1)
            order = np.argsort(rows)
            rs    = rows[order]
            for s in range(0, H, STRIP):
                lo, hi = np.searchsorted(rs, s), np.searchsorted(rs, s + STRIP)
                if hi == lo:
                    continue
                sel  = order[lo:hi]
                r0   = max(s - half, 0)
                r1   = min(s + STRIP + half, H)
                data = src.read([1, 2, 3], window=Window(0, r0, W, r1 - r0))
                vals = []
                for dy, dx in offs:
                    rr = np.clip(rows[sel] + dy - r0, 0, r1 - r0 - 1)
                    cc = np.clip(cols[sel] + dx,       0, W - 1)
                    vals.append(data[:, rr, cc].astype(np.float32))
                v = vals[0] if len(vals) == 1 else np.median(np.stack(vals), axis=0)
                if src.nodata is not None:
                    nodata_mask = (
                        data[:, np.clip(rows[sel] - r0, 0, r1 - r0 - 1), cols[sel]]
                        == src.nodata
                    ).all(axis=0)
                    v[:, nodata_mask] = np.nan
                out[idx[sel]] = v.T
            done[idx] = True
    return out


# ============================================================================
# MAIN
# ============================================================================

def main():
    # -- read settings from CONFIG block at top of file ---------------------
    out_dir          = OUT_DIR
    bboxes_path      = BBOXES_PATH
    rgb_path_input   = RGB_PATH
    emb_path_input   = EMB_PATH
    gt_path          = GT_PATH
    feature_selection = FEATURE_SELECTION
    fi_threshold     = FI_THRESHOLD if feature_selection else 0.0
    folds            = FOLDS
    max_per_class    = MAX_PER_CLASS
    label_pool       = LABEL_POOL
    random_ratio     = RANDOM_RATIO
    min_purity       = MIN_PURITY
    n_jobs           = N_JOBS
    seed             = SEED
    predict_all      = PREDICT_ALL
    threshold        = THRESHOLD
    center_win       = CENTER_WIN
    block_px         = BLOCK_PX
    workers          = WORKERS
    chunk            = CHUNK
    max_emb_gb       = MAX_EMB_GB

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("\n" + "=" * 60)
    print("  Mango vs Non-Mango Pipeline")
    print("=" * 60)
    print(f"  bboxes : {bboxes_path}")
    print(f"  rgb    : {rgb_path_input}")
    print(f"  emb    : {emb_path_input}")
    print(f"  gt     : {gt_path}")
    print(f"  out    : {out_dir}")
    print(f"  feat.sel: {'ON  (threshold=' + str(fi_threshold) + ')' if feature_selection else 'OFF (all 131 features)'}")
    print(f"  predict : {'YES' if predict_all else 'NO'}")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1 – Feature Extraction                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "─" * 60)
    print("[Step 1/5] Scanning & loading rasters …")
    print("─" * 60, flush=True)

    rgb_path = open_raster_or_tiles(rgb_path_input, out_dir, "rgb")
    emb_path = open_raster_or_tiles(emb_path_input, out_dir, "emb")

    print(f"\n[Step 1/5] Opening rasters …", flush=True)
    with rasterio.open(rgb_path) as rgb, \
         rasterio.open(emb_path) as emb, \
         rasterio.open(gt_path)  as gt:

        assert emb.count == 128, f"expected 128 embedding bands, got {emb.count}"
        t = emb.transform
        assert t.b == 0 and t.d == 0 and t.e < 0, "embeddings must be north-up"

        print(f"  RGB  : {rgb.width}×{rgb.height} px | {rgb.count} bands | CRS={rgb.crs.to_epsg()}")
        print(f"  EMB  : {emb.width}×{emb.height} px | {emb.count} bands | CRS={emb.crs.to_epsg()}")
        print(f"  GT   : {gt.width}×{gt.height} px  | {gt.count} band  | CRS={gt.crs.to_epsg()}", flush=True)

        rgb_crs, emb_crs, gt_crs = rgb.crs, emb.crs, gt.crs
        G.update(
            rgb_path=rgb_path, emb_path=emb_path, gt_path=gt_path,
            emb_tf=t,  emb_h=emb.height, emb_w=emb.width,
            min_purity=min_purity, center_win=center_win,
            emb_arr=load_emb_array(emb, max_emb_gb),
        )

        print(f"\n[Step 1/5] Reading bboxes …", flush=True)
        gdf = gpd.read_file(bboxes_path)[["geometry"]].reset_index(drop=True)
        N   = len(gdf)
        print(f"  Loaded {N} tree bboxes from {os.path.basename(bboxes_path)}")
        print(f"  Workers: {workers} | Chunk: {chunk}", flush=True)

        # 1a) label a random pool from GT
        print(f"\n[Step 1/5] Labelling bboxes from GT mask (pool={min(N, label_pool)}) …",
              flush=True)
        pool = rng.permutation(N)[:min(N, label_pool)]
        G["gt_bounds"] = bounds_in(gdf.iloc[pool], gt_crs)
        lab = np.concatenate(
            run_parallel(_label_chunk, len(pool), workers,
                         chunk, "label bboxes from GT")
        )
        m_idx, n_idx = pool[lab == 1], pool[lab == 0]
        print(f"  Labelled pool result:")
        print(f"    Mango       : {len(m_idx)}")
        print(f"    Non-mango   : {len(n_idx)}")
        print(f"    Unlabelled  : {int((lab < 0).sum())}", flush=True)
        if len(m_idx) == 0:
            raise SystemExit("no mango bboxes found; check GT raster / CRS")

        # 1b) stratified random sample per class
        m_sel = rng.choice(m_idx, min(len(m_idx), max_per_class), replace=False)
        n_sel = rng.choice(n_idx, min(len(n_idx), max_per_class), replace=False)
        print(f"\n  Sampled for training:")
        print(f"    Mango       : {len(m_sel)}")
        print(f"    Non-mango   : {len(n_sel)}", flush=True)

        # 1c) top-up non-mango with random GT==2 pixels
        n_need = int(random_ratio * len(m_sel)) - len(n_sel)
        xs = ys = np.array([])
        if random_ratio > 0 and n_need > 0:
            print(f"\n  Topping up non-mango with {n_need} random GT==2 pixels …",
                  flush=True)
            xs, ys = sample_nonmango_points(gt, n_need, seed)
            print(f"  Random non-mango pixels added: {len(xs)}", flush=True)

        sel    = np.concatenate([m_sel, n_sel])
        y_all  = np.r_[np.ones(len(m_sel)),  np.zeros(len(n_sel)),
                        np.zeros(len(xs))].astype(int)
        source = np.r_[["bbox"] * len(sel), ["random_gt2"] * len(xs)]
        g_sel  = gdf.iloc[sel]

        rgb_xy = centre_xy(g_sel, rgb_crs)
        emb_b  = bounds_in(g_sel, emb_crs)
        if len(xs):
            rx, ry = warp_transform(gt_crs, rgb_crs, list(xs), list(ys))
            ex, ey = warp_transform(gt_crs, emb_crs, list(xs), list(ys))
            rgb_xy = np.vstack([rgb_xy, np.c_[rx, ry]])
            emb_b  = np.vstack([emb_b,  np.c_[ex, ey, ex, ey]])

    # 1d) extract 131 features in parallel
    print(f"\n[Step 1/5] Extracting 131 features "
          f"(128 embedding + 3 RGB) for {len(rgb_xy)} samples …", flush=True)
    class _Cfg:
        pass
    cfg          = _Cfg()
    cfg.workers  = workers
    cfg.chunk    = chunk
    cfg.block_px = block_px

    F, group = extract_features(rgb_xy, emb_b, cfg, "extract features")
    df = pd.DataFrame(F, columns=FEAT_COLS)
    df["label"]  = y_all
    df["source"] = source
    df["group"]  = group
    df = df.dropna(subset=FEAT_COLS).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "train_table.csv"), index=False)

    X_all = df[FEAT_COLS].values.astype(np.float32)
    y     = df.label.values.astype(int)
    print(f"\n  Feature extraction complete:")
    print(f"    Total usable samples : {len(y)}")
    print(f"    Mango                : {int(y.sum())}")
    print(f"    Non-mango            : {int((y == 0).sum())}")
    print(f"    Saved -> train_table.csv", flush=True)

    # ------------------------------------------------------------------ #
    # Step 2 – Feature Selection                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "─" * 60)
    print("[Step 2/5] Feature selection")
    print("─" * 60, flush=True)

    if feature_selection:
        print(f"  Mode      : ON")
        print(f"  Threshold : {fi_threshold:.0%} cumulative MDI importance")
        print(f"  Fitting RF on {len(y)} samples to rank 131 features …",
              flush=True)
        sel_cols, sel_idx, imp_df = select_important_features(
            X_all, y, FEAT_COLS,
            threshold=fi_threshold,
            n_jobs=n_jobs,
            seed=seed,
        )
        imp_df.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
        with open(os.path.join(out_dir, "selected_features.json"), "w") as fh:
            json.dump(
                {"features": sel_cols, "indices": sel_idx,
                 "fi_threshold": fi_threshold,
                 "n_selected": len(sel_cols), "n_total": len(FEAT_COLS)},
                fh, indent=2,
            )
        print(f"  Saved -> feature_importance.csv, selected_features.json",
              flush=True)
        X          = X_all[:, sel_idx]
        active_cols = sel_cols
    else:
        print("  FEATURE_SELECTION=False -> using all 131 features")
        sel_idx     = list(range(len(FEAT_COLS)))
        sel_cols    = list(FEAT_COLS)
        active_cols = list(FEAT_COLS)
        X           = X_all
        with open(os.path.join(out_dir, "selected_features.json"), "w") as fh:
            json.dump(
                {"features": sel_cols, "indices": sel_idx,
                 "fi_threshold": 0.0,
                 "n_selected": len(sel_cols), "n_total": len(FEAT_COLS)},
                fh, indent=2,
            )

    # ------------------------------------------------------------------ #
    # Step 3 – Spatial-block CV                                            #
    # ------------------------------------------------------------------ #
    n_groups = len(np.unique(df.group.values))
    _folds   = folds
    if n_groups < _folds:
        print(f"  only {n_groups} spatial blocks -> using {max(n_groups, 2)} folds")
        _folds = max(n_groups, 2)

    print("\n" + "─" * 60)
    print(f"[Step 3/5] Spatial CV  ({_folds} folds | {len(active_cols)} features)")
    print("─" * 60, flush=True)

    skf = StratifiedGroupKFold(_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    bar = tqdm(list(skf.split(X, y, df.group.values)), desc="CV folds")
    for tr, te in bar:
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m   = make_ensemble(spw, n_jobs, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        _, _, f_, _ = precision_recall_fscore_support(
            y[te], (oof[te] >= 0.5).astype(int), labels=[1], zero_division=0)
        bar.set_postfix(mango_f1=f"{f_[0]:.3f}")

    metrics = {"all_samples": report(y, oof, "Spatial CV – all samples")}
    is_box  = (df.source == "bbox").values
    if is_box.any() and len(np.unique(y[is_box])) == 2:
        metrics["bbox_only"] = report(
            y[is_box], oof[is_box], "Spatial CV – real bboxes only")
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("  Saved -> metrics.json", flush=True)

    # ------------------------------------------------------------------ #
    # Step 4 – Final model training                                        #
    # ------------------------------------------------------------------ #
    print("\n" + "─" * 60)
    print(f"[Step 4/5] Final model training")
    print("─" * 60, flush=True)
    print(f"  Training RF + XGB ensemble on {len(y)} samples "
          f"with {len(active_cols)} features …", flush=True)

    spw   = (y == 0).sum() / max((y == 1).sum(), 1)
    final = make_ensemble(spw, n_jobs, seed).fit(X, y)
    joblib.dump(final, os.path.join(out_dir, "mango_ensemble.joblib"))
    print(f"  Saved -> mango_ensemble.joblib", flush=True)

    # ------------------------------------------------------------------ #
    # Step 5 – Inference on all bboxes                                     #
    # ------------------------------------------------------------------ #
    if not predict_all:
        print("\nPipeline complete (PREDICT_ALL=False – no inference run).")
        return

    print("\n" + "─" * 60)
    print(f"[Step 5/5] Inference on all {N} bboxes")
    print("─" * 60, flush=True)
    print(f"  Using {len(active_cols)} selected features for prediction …",
          flush=True)

    rgb_tiles = list_tiles(rgb_path_input)
    with rasterio.open(rgb_tiles[0]) as s:
        infer_rgb_crs = s.crs

    with rasterio.open(emb_path) as emb:
        assert emb.count == 128
        t_inf   = emb.transform
        eb      = gdf.to_crs(emb.crs).bounds.values
        rc_inf  = bounds_to_rc(eb, t_inf)
        arr_inf = load_emb_array(emb, max_emb_gb)

        b_rgb   = gdf.to_crs(infer_rgb_crs).bounds.values
        xy_inf  = np.c_[(b_rgb[:, 0] + b_rgb[:, 2]) / 2,
                        (b_rgb[:, 1] + b_rgb[:, 3]) / 2]
        print(f"  Sampling RGB at {N} bbox centres …", flush=True)
        rgb_inf = sample_rgb_infer(rgb_tiles, xy_inf, center_win)

        print(f"  Running prediction …", flush=True)
        prob = np.full(N, np.nan, np.float32)
        for s_i in tqdm(range(0, N, 200_000), desc="features + predict"):
            e_i  = min(s_i + 200_000, N)
            F131 = np.hstack([
                emb_features_infer(rc_inf[s_i:e_i], arr_inf, emb,
                                   emb.height, emb.width),
                rgb_inf[s_i:e_i],
            ])
            Fsel = F131[:, sel_idx]   # apply same feature selection as training
            ok   = np.nonzero(~np.isnan(Fsel).any(axis=1))[0]
            for j in range(0, len(ok), 100_000):
                ii = ok[j:j + 100_000]
                prob[s_i + ii] = final.predict_proba(Fsel[ii])[:, 1]

    gdf["prob_mango"] = prob
    gdf["pred"] = np.where(
        np.isnan(prob), "nodata",
        np.where(prob >= threshold, "mango", "non-mango"),
    )
    vc = gdf["pred"].value_counts()
    print(f"\n  Prediction summary (threshold={threshold:.2f}):")
    for k, v in vc.items():
        print(f"    {k:<12} {v:>10d}  ({100 * v / N:.1f}%)")

    out_gpkg = os.path.join(out_dir, "bboxes_classified.gpkg")
    print(f"\n  Writing {out_gpkg} …", flush=True)
    try:
        gdf.to_file(out_gpkg, engine="pyogrio")
    except Exception:
        gdf.to_file(out_gpkg)
    print("  Saved -> bboxes_classified.gpkg", flush=True)

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Output folder: {out_dir}")
    print("─" * 60)
    for fname in ["train_table.csv", "feature_importance.csv",
                  "selected_features.json", "mango_ensemble.joblib",
                  "metrics.json", "bboxes_classified.gpkg"]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            size = os.path.getsize(p)
            print(f"  {fname:<35} {size/1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
