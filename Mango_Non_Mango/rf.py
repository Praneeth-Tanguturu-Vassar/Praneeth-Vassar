"""
Mango vs non-mango tree classifier  (fast version for ~1M+ bboxes)
------------------------------------------------------------------
Inputs
  --bboxes : YOLO tree boxes as georeferenced polygons
  --rgb    : Google satellite GeoTIFF or folder of tiles (0.15 m, 3 bands)
  --emb    : annual embeddings GeoTIFF or folder of tiles (128 bands)
  --gt     : GT mask GeoTIFF (1 = mango, 2 = non-mango, 0 = background)

Features (131) = 128 embedding mean over bbox + 3 RGB at bbox centre
Labels: mango = 1, non-mango = 0

Speed-ups
  * random sampler: only a random pool of bboxes is labelled and a capped,
    random set per class (--max-per-class) is used for training
  * embeddings are loaded into RAM once (fast numpy slicing, no per-bbox I/O)
  * feature extraction runs in parallel worker processes, with tqdm bars
  * full prediction of all bboxes is optional (--predict-all)
"""
import argparse
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
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds
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

EMB_COLS = [f"e{i}" for i in range(128)]
RGB_COLS = ["r", "g", "b"]
FEAT_COLS = EMB_COLS + RGB_COLS
G = {}   # shared state; inherited by forked workers


# ----------------------------------------------------------------------------
# Feature selection (MDI-based, single RF pass)
# ----------------------------------------------------------------------------
def select_important_features(X, y, feat_cols, threshold=0.90, n_jobs=4, seed=42):
    """Fit one RF on all training data, rank by MDI, return features covering
    `threshold` of total importance (e.g. 0.90 = 90%).

    Returns
    -------
    sel_cols : list[str]  – selected feature names (ordered by index, not importance)
    sel_idx  : list[int]  – corresponding column indices into feat_cols
    """
    from sklearn.ensemble import RandomForestClassifier as _RFC
    rf = _RFC(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=n_jobs, random_state=seed,
    )
    print(f"feature selection: fitting RF on {len(y)} samples …", flush=True)
    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]            # highest importance first
    cum = np.cumsum(imp[order]) / imp.sum()
    n_keep = int(np.searchsorted(cum, threshold)) + 1
    sel_idx = sorted(order[:n_keep].tolist())
    sel_cols = [feat_cols[i] for i in sel_idx]
    print(f"feature selection: keeping {n_keep}/{len(feat_cols)} features "
          f"(cover {threshold * 100:.0f}% MDI importance)", flush=True)
    print(f"  selected: {sel_cols[:10]}{'…' if n_keep > 10 else ''}", flush=True)
    return sel_cols, sel_idx


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Tiles -> VRT
# ----------------------------------------------------------------------------
def open_raster_or_tiles(path, out_dir, name):
    if not os.path.isdir(path):
        return path
    tiles = sorted(os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs
                   if f.lower().endswith((".tif", ".tiff")))
    if not tiles:
        raise FileNotFoundError(f"no .tif tiles found in {path}")
    crs_set = set()
    for t in tqdm(tiles, desc=f"{name}: check CRS"):
        with rasterio.open(t) as s:
            crs_set.add(str(s.crs))
    if len(crs_set) > 1:
        raise ValueError(f"tiles in {path} have different CRS: {crs_set}. "
                         "Reproject them to one CRS first.")
    os.makedirs(out_dir, exist_ok=True)
    vrt = os.path.join(out_dir, f"{name}_mosaic.vrt")
    lst = os.path.join(out_dir, f"{name}_tiles.txt")
    with open(lst, "w") as f:
        f.write("\n".join(tiles))
    print(f"{name}: mosaicking {len(tiles)} tiles -> {vrt}")
    try:
        from osgeo import gdal
        gdal.BuildVRT(vrt, tiles)
    except ImportError:
        subprocess.run(["gdalbuildvrt", "-input_file_list", lst, vrt], check=True)
    return vrt


# ----------------------------------------------------------------------------
# Raster helpers (no boundless reads: they are very slow)
# ----------------------------------------------------------------------------
def window_for_bounds(src, bounds):
    w = from_bounds(*bounds, transform=src.transform)
    c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
    c1 = int(np.ceil(w.col_off + w.width))
    r1 = int(np.ceil(w.row_off + w.height))
    return Window(c0, r0, max(c1 - c0, 1), max(r1 - r0, 1))


def clipped_window(src, win):
    c0, r0 = max(win.col_off, 0), max(win.row_off, 0)
    c1 = min(win.col_off + win.width, src.width)
    r1 = min(win.row_off + win.height, src.height)
    if c1 <= c0 or r1 <= r0:
        return None
    return Window(c0, r0, c1 - c0, r1 - r0)


def label_from_gt(src, bounds, min_purity):
    """1 = mango, 0 = non-mango, -1 = no/ambiguous GT inside bbox."""
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


def load_emb_array(emb, max_gb):
    gb = emb.count * emb.height * emb.width * 4 / 1e9
    if gb > max_gb:
        print(f"embeddings need {gb:.1f} GB > --max-emb-gb; using windowed reads")
        return None
    print(f"loading embeddings into RAM ({gb:.1f} GB) ...")
    arr = np.empty((emb.count, emb.height, emb.width), np.float32)
    for r in tqdm(range(0, emb.height, 1024), desc="load emb"):
        h = min(1024, emb.height - r)
        a = emb.read(window=Window(0, r, emb.width, h)).astype(np.float32)
        if emb.nodata is not None and not np.isnan(emb.nodata):
            a[a == emb.nodata] = np.nan
        arr[:, r:r + h] = a
    return arr


def bounds_in(gdf, crs):
    return gdf.to_crs(crs).bounds.values   # minx, miny, maxx, maxy


def centre_xy(gdf, crs):
    b = bounds_in(gdf, crs)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def bounds_to_rc(b, t):
    """Vectorised bounds -> pixel window (r0, r1, c0, c1), north-up raster."""
    c0 = np.floor((b[:, 0] - t.c) / t.a)
    c1 = np.ceil((b[:, 2] - t.c) / t.a)
    r0 = np.floor((b[:, 3] - t.f) / t.e)
    r1 = np.ceil((b[:, 1] - t.f) / t.e)
    c1 = np.maximum(c1, c0 + 1)
    r1 = np.maximum(r1, r0 + 1)
    return np.c_[r0, r1, c0, c1].astype(np.int64)


# ----------------------------------------------------------------------------
# Parallel workers
# ----------------------------------------------------------------------------
def _init_worker():
    G["rgb"] = rasterio.open(G["rgb_path"])
    G["gt"] = rasterio.open(G["gt_path"])
    if G["emb_arr"] is None:
        G["emb"] = rasterio.open(G["emb_path"])


def _label_chunk(c):
    s, e = c
    b = G["gt_bounds"]
    return np.array([label_from_gt(G["gt"], b[k], G["min_purity"])
                     for k in range(s, e)], dtype=np.int8)


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
    if np.isnan(v).any() or not v.any():      # nodata / uncovered VRT gap
        return None
    return v.astype(np.float32)


def _feat_chunk(c):
    s, e = c
    F = np.full((e - s, 131), np.nan, np.float32)
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
    rc = G["emb_rc"]
    group = (rc[:, 0] // args.block_px) * 100000 + (rc[:, 2] // args.block_px)
    return F, group


def sample_nonmango_points(gt, n, seed, per_block=5000):
    """Random pixels where GT == 2 -> (xs, ys) in GT CRS."""
    rng = np.random.default_rng(seed)
    pts = []
    for _, win in tqdm(list(gt.block_windows(1)), desc="scan GT for class 2"):
        a = gt.read(1, window=win)
        rr, cc = np.nonzero(a == 2)
        if len(rr) == 0:
            continue
        k = min(per_block, len(rr))
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


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
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
    print("confusion matrix [rows=true, cols=pred] (non-mango, mango)\n",
          confusion_matrix(y, pred, labels=[0, 1]))
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan")
    print(f"ROC-AUC: {auc:.4f}")
    return dict(mango_precision=float(p[1]), mango_recall=float(r[1]),
                mango_f1=float(f[1]), nonmango_precision=float(p[0]),
                nonmango_recall=float(r[0]), nonmango_f1=float(f[0]),
                macro_f1=float(mf_), roc_auc=float(auc), n=int(len(y)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--max-per-class", type=int, default=25000,
                    help="random sampler: max bboxes used per class for training")
    ap.add_argument("--label-pool", type=int, default=300000,
                    help="random bboxes to label from the GT before sampling")
    ap.add_argument("--random-ratio", type=float, default=1.0,
                    help="target non-mango:mango ratio; random GT==2 pixels fill "
                         "the gap left by non-mango bboxes (0 = disable)")
    ap.add_argument("--min-purity", type=float, default=0.6)
    ap.add_argument("--center-win", type=int, default=1)
    ap.add_argument("--block-px", type=int, default=50,
                    help="spatial CV block size in embedding pixels")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int,
                    default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=8, help="threads for RF/XGB")
    ap.add_argument("--max-emb-gb", type=float, default=8.0)
    ap.add_argument("--predict-all", action="store_true",
                    help="also classify every bbox and write a GPKG (slow for 1M+)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rgb_path = open_raster_or_tiles(args.rgb, args.out_dir, "rgb")
    emb_path = open_raster_or_tiles(args.emb, args.out_dir, "emb")

    with rasterio.open(rgb_path) as rgb, rasterio.open(emb_path) as emb, \
            rasterio.open(args.gt) as gt:
        assert emb.count == 128, f"expected 128 embedding bands, got {emb.count}"
        t = emb.transform
        assert t.b == 0 and t.d == 0 and t.e < 0, "embeddings must be north-up"
        rgb_crs, emb_crs, gt_crs = rgb.crs, emb.crs, gt.crs
        G.update(rgb_path=rgb_path, emb_path=emb_path, gt_path=args.gt,
                 emb_tf=t, emb_h=emb.height, emb_w=emb.width,
                 min_purity=args.min_purity, center_win=args.center_win,
                 emb_arr=load_emb_array(emb, args.max_emb_gb))

        gdf = gpd.read_file(args.bboxes)[["geometry"]].reset_index(drop=True)
        N = len(gdf)
        print(f"{N} bboxes | workers={args.workers}")

        # 1) random pool -> GT label
        pool = rng.permutation(N)[:min(N, args.label_pool)]
        G["gt_bounds"] = bounds_in(gdf.iloc[pool], gt_crs)
        lab = np.concatenate(run_parallel(_label_chunk, len(pool), args.workers,
                                          args.chunk, "label bboxes from GT"))
        m_idx, n_idx = pool[lab == 1], pool[lab == 0]
        print(f"pool labelled: mango={len(m_idx)}, non-mango={len(n_idx)}, "
              f"unlabelled={int((lab < 0).sum())}")
        if len(m_idx) == 0:
            raise SystemExit("no mango bboxes found in pool; check GT / CRS")

        # 2) random sampler per class
        m_sel = rng.choice(m_idx, min(len(m_idx), args.max_per_class), replace=False)
        n_sel = rng.choice(n_idx, min(len(n_idx), args.max_per_class), replace=False)

        # 3) top up non-mango with random GT==2 pixels
        n_need = int(args.random_ratio * len(m_sel)) - len(n_sel)
        xs = ys = np.array([])
        if args.random_ratio > 0 and n_need > 0:
            xs, ys = sample_nonmango_points(gt, n_need, args.seed)
            print(f"random non-mango pixels added: {len(xs)}")

        sel = np.concatenate([m_sel, n_sel])
        y = np.r_[np.ones(len(m_sel)), np.zeros(len(n_sel)),
                  np.zeros(len(xs))].astype(int)
        source = np.r_[["bbox"] * len(sel), ["random_gt2"] * len(xs)]
        g_sel = gdf.iloc[sel]
        rgb_xy = centre_xy(g_sel, rgb_crs)
        emb_b = bounds_in(g_sel, emb_crs)
        if len(xs):
            rx, ry = warp_transform(gt_crs, rgb_crs, list(xs), list(ys))
            ex, ey = warp_transform(gt_crs, emb_crs, list(xs), list(ys))
            rgb_xy = np.vstack([rgb_xy, np.c_[rx, ry]])
            emb_b = np.vstack([emb_b, np.c_[ex, ey, ex, ey]])

    # 4) features (parallel)
    F, group = extract_features(rgb_xy, emb_b, args, "extract features")
    df = pd.DataFrame(F, columns=FEAT_COLS)
    df["label"], df["source"], df["group"] = y, source, group
    df = df.dropna(subset=FEAT_COLS).reset_index(drop=True)
    df.to_csv(os.path.join(args.out_dir, "train_table.csv"), index=False)
    X = df[FEAT_COLS].values
    y = df.label.values
    print(f"training set: {len(y)} (mango={int(y.sum())}, "
          f"non-mango={int((y == 0).sum())})")

    # 5) spatially-blocked CV
    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    bar = tqdm(list(skf.split(X, y, df.group.values)), desc="CV folds")
    for tr, te in bar:
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        m = make_ensemble(spw, args.n_jobs, args.seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        _, r_, f_, _ = precision_recall_fscore_support(
            y[te], (oof[te] >= 0.5).astype(int), labels=[1], zero_division=0)[:4]
        bar.set_postfix(mango_f1=f"{f_[0]:.3f}")

    metrics = {"all_samples": report(y, oof, "Spatial CV - all samples")}
    is_box = (df.source == "bbox").values
    if is_box.any() and len(np.unique(y[is_box])) == 2:
        metrics["bbox_only"] = report(y[is_box], oof[is_box],
                                      "Spatial CV - real bboxes only")
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 6) final model
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    final = make_ensemble(spw, args.n_jobs, args.seed).fit(X, y)
    joblib.dump(final, os.path.join(args.out_dir, "mango_ensemble.joblib"))
    print("saved model + metrics.json + train_table.csv")

    # 7) optional: classify every bbox
    if args.predict_all:
        rgb_xy = centre_xy(gdf, rgb_crs)
        emb_b = bounds_in(gdf, emb_crs)
        F, _ = extract_features(rgb_xy, emb_b, args, "features (all bboxes)")
        prob = np.full(N, np.nan, np.float32)
        ok = np.nonzero(~np.isnan(F).any(axis=1))[0]
        for s in tqdm(range(0, len(ok), 100000), desc="predict"):
            ii = ok[s:s + 100000]
            prob[ii] = final.predict_proba(F[ii])[:, 1]
        gdf["prob_mango"] = prob
        gdf["pred"] = np.where(np.isnan(prob), "nodata",
                               np.where(prob >= 0.5, "mango", "non-mango"))
        gdf.to_file(os.path.join(args.out_dir, "bboxes_classified.gpkg"),
                    driver="GPKG")
        print("saved bboxes_classified.gpkg")


if __name__ == "__main__":
    main()