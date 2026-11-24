"""
Feature relevance for mango vs non-mango (131 features)
--------------------------------------------------------
Features : e0..e127 (embedding mean over bbox) + r, g, b (centre pixel)
Scores   : 1) RF impurity importance (MDI), averaged over spatial CV folds
           2) Permutation importance (drop in ROC-AUC on held-out spatial fold)
Outputs  : feature_importance.csv
           feature_importance_all.png   (all 131 features, sorted, + cumulative curve)
           feature_importance_top<N>.png (top-N, horizontal)
           perm_importance_top<N>.png    (if permutation enabled)
           important_features.png / .csv (features covering --cum-threshold)
           topk_eval.csv                 (CV AUC using only the top-k features)

Usage
  # A) reuse train_table.csv written by mango_classifier.py (fastest)
  python feature_importance.py --train-table out/train_table.csv --out-dir fi_out

  # B) extract straight from rasters (standalone, no other script needed)
  #    --rgb / --emb can be a single GeoTIFF or a FOLDER of tiles (mosaicked to a VRT)
  python feature_importance.py --bboxes trees.gpkg --rgb rgb_tiles/ --emb emb_tiles/ \
         --gt gt.tif --out-dir fi_out
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

EMB_COLS = [f"e{i}" for i in range(128)]
RGB_COLS = ["r", "g", "b"]
FEAT_COLS = EMB_COLS + RGB_COLS
COLORS = {"embedding": "#4C72B0", "rgb": "#DD8452"}


def make_rf(seed, n_jobs):
    # same RF settings as the reference ensemble (no scaling needed for RF)
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced", n_jobs=n_jobs, random_state=seed,
    )


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
GDAL_TYPES = {"uint8": "Byte", "int8": "Int8", "uint16": "UInt16",
              "int16": "Int16", "uint32": "UInt32", "int32": "Int32",
              "float32": "Float32", "float64": "Float64"}


def list_tiles(folder, recursive=False):
    """Raster tiles in `folder` (top level only unless recursive)."""
    exts = (".tif", ".tiff")
    if recursive:
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(folder) for f in fs]
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
    return sorted(f for f in files
                  if os.path.isfile(f) and f.lower().endswith(exts))


def filter_tiles(files, want_bands):
    """Drop unreadable tiles and tiles whose band count / CRS don't match.
    want_bands: exact band count required (e.g. 128 for embeddings, 3 for RGB)."""
    import rasterio
    from collections import Counter
    ok, bad, wrong = [], [], []
    for p in files:
        try:
            with rasterio.open(p) as s:
                if s.count != want_bands:
                    wrong.append(p)
                    continue
                ok.append((p, s.crs.to_string() if s.crs else None))
        except Exception:
            bad.append(p)
    if not ok:
        return [], bad, wrong, []
    main_crs = Counter(c for _, c in ok).most_common(1)[0][0]
    keep = [p for p, c in ok if c == main_crs]
    other_crs = [p for p, c in ok if c != main_crs]
    return keep, bad, wrong, other_crs


def resolve_raster(path, name, out_dir, want_bands):
    """A single file is used as-is; a folder of tiles becomes one VRT mosaic."""
    if not os.path.isdir(path):
        return path
    keep, bad, wrong, other = filter_tiles(list_tiles(path), want_bands)
    if not keep:   # nothing usable at top level -> look in subfolders
        keep, bad, wrong, other = filter_tiles(list_tiles(path, True), want_bands)
    if not keep:
        raise SystemExit(f"{name}: no readable {want_bands}-band tiles in {path}")
    print(f"{name}: using {len(keep)} tiles from {path}")
    if bad:
        print(f"  skipped {len(bad)} unreadable/corrupt tiles: "
              f"{', '.join(os.path.basename(p) for p in bad[:10])}"
              f"{' ...' if len(bad) > 10 else ''}")
    if wrong:
        print(f"  skipped {len(wrong)} tiles without {want_bands} bands "
              f"(e.g. {os.path.basename(wrong[0])})")
    if other:
        print(f"  skipped {len(other)} tiles in a different CRS "
              f"(e.g. {os.path.basename(other[0])})")
    vrt = os.path.join(out_dir, f"{name}_mosaic.vrt")
    return write_vrt(keep, vrt)


def write_vrt(files, vrt_path):
    """Mosaic tiles into a VRT. Uses gdalbuildvrt if installed, otherwise
    writes the VRT XML directly (tiles must share CRS, resolution, bands)."""
    import shutil
    import subprocess
    from xml.sax.saxutils import escape
    import rasterio

    if shutil.which("gdalbuildvrt"):
        lst = vrt_path + ".txt"
        with open(lst, "w") as f:
            f.write("\n".join(os.path.abspath(p) for p in files))
        subprocess.run(["gdalbuildvrt", "-q", "-overwrite", "-input_file_list", lst,
                        vrt_path],
                       check=True)
        return vrt_path

    metas = []
    for p in files:
        with rasterio.open(p) as s:
            metas.append(dict(path=os.path.abspath(p), crs=s.crs, res=s.res,
                              count=s.count, dtype=s.dtypes[0], nodata=s.nodata,
                              w=s.width, h=s.height, b=s.bounds))
    m0 = metas[0]
    for m in metas[1:]:
        if m["crs"] != m0["crs"] or m["count"] != m0["count"] or \
                not np.allclose(m["res"], m0["res"]):
            raise SystemExit(
                f"{m['path']} differs in CRS/resolution/band count from "
                f"{m0['path']}; install GDAL (gdalbuildvrt) or reproject tiles")
    rx, ry = m0["res"]
    minx = min(m["b"].left for m in metas)
    maxy = max(m["b"].top for m in metas)
    maxx = max(m["b"].right for m in metas)
    miny = min(m["b"].bottom for m in metas)
    W, H = int(round((maxx - minx) / rx)), int(round((maxy - miny) / ry))
    dtype = GDAL_TYPES.get(m0["dtype"], "Float32")
    nodata = m0["nodata"]
    if nodata is None and m0["dtype"].startswith("float"):
        nodata = "nan"   # gaps between tiles stay masked, not 0

    out = [f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
           f"  <SRS>{escape(m0['crs'].to_wkt())}</SRS>",
           f"  <GeoTransform>{minx}, {rx}, 0, {maxy}, 0, {-ry}</GeoTransform>"]
    for band in range(1, m0["count"] + 1):
        out.append(f'  <VRTRasterBand dataType="{dtype}" band="{band}">')
        if nodata is not None:
            out.append(f"    <NoDataValue>{nodata}</NoDataValue>")
        for m in metas:
            xo = int(round((m["b"].left - minx) / rx))
            yo = int(round((maxy - m["b"].top) / ry))
            out += ["    <SimpleSource>",
                    f'      <SourceFilename relativeToVRT="0">'
                    f'{escape(m["path"])}</SourceFilename>',
                    f"      <SourceBand>{band}</SourceBand>",
                    f'      <SrcRect xOff="0" yOff="0" xSize="{m["w"]}" '
                    f'ySize="{m["h"]}"/>',
                    f'      <DstRect xOff="{xo}" yOff="{yo}" xSize="{m["w"]}" '
                    f'ySize="{m["h"]}"/>',
                    "    </SimpleSource>"]
        out.append("  </VRTRasterBand>")
    out.append("</VRTDataset>")
    with open(vrt_path, "w") as f:
        f.write("\n".join(out))
    return vrt_path


def read_bboxes(path):
    """A single GPKG/GeoJSON/SHP, or a folder of them merged into one layer."""
    import geopandas as gpd
    if not os.path.isdir(path):
        return gpd.read_file(path).reset_index(drop=True)
    exts = (".gpkg", ".geojson", ".json", ".shp", ".fgb", ".kml", ".parquet")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(path)
                   for f in fs if f.lower().endswith(exts))
    if not files:
        found = sorted({os.path.splitext(f)[1].lower() or f
                        for _, _, fs in os.walk(path) for f in fs})
        raise SystemExit(f"no vector bbox files (.gpkg/.geojson/.shp/.fgb/.kml/"
                         f".parquet) in {path}\n  file types found there: {found}")
    parts = []
    for f in files:
        g = gpd.read_parquet(f) if f.lower().endswith(".parquet") else gpd.read_file(f)
        if len(g):
            parts.append(g)
    crs = parts[0].crs
    gdf = pd.concat([p.to_crs(crs) for p in parts], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    print(f"bboxes: {len(files)} files in {path} -> {len(gdf)} boxes")
    return gdf


# ----------------------------------------------------------------------------
# Feature extraction (same logic as the reference mango_classifier.py)
# ----------------------------------------------------------------------------
def window_for_bounds(src, bounds):
    """Pixel window covering every pixel the bounds touch (min 1x1)."""
    from rasterio.windows import Window, from_bounds
    w = from_bounds(*bounds, transform=src.transform)
    c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
    c1 = int(np.ceil(w.col_off + w.width))
    r1 = int(np.ceil(w.row_off + w.height))
    return Window(c0, r0, max(c1 - c0, 1), max(r1 - r0, 1))


def emb_mean(src, bounds):
    """Mean of all embedding pixels touched by the bbox -> (128,) or None."""
    win = window_for_bounds(src, bounds)
    arr = src.read(window=win, boundless=True, masked=True)
    arr = np.ma.masked_invalid(arr)
    if arr.mask.all():
        return None
    v = np.ma.filled(arr.reshape(arr.shape[0], -1).mean(axis=1), np.nan)
    return None if np.isnan(v).any() else v.astype(np.float32)


def center_rgb(src, x, y, win=1):
    """RGB at bbox centre (single pixel by default; odd win -> median)."""
    from rasterio.windows import Window
    r, c = src.index(x, y)
    h = win // 2
    a = src.read([1, 2, 3], window=Window(c - h, r - h, win, win),
                 boundless=True, masked=True)
    a = np.ma.masked_invalid(a).reshape(3, -1)
    if a.mask.all():
        return None
    v = np.ma.filled(np.ma.median(a, axis=1), np.nan)
    return None if np.isnan(v).any() else v.astype(np.float32)


def label_from_gt(src, bounds, min_purity):
    """1 = mango, 0 = non-mango, -1 = no/ambiguous GT inside bbox."""
    win = window_for_bounds(src, bounds)
    a = src.read(1, window=win, boundless=True, fill_value=0)
    n1, n2 = int((a == 1).sum()), int((a == 2).sum())
    tot = n1 + n2
    if tot == 0:
        return -1
    if n1 / tot >= min_purity:
        return 1
    if n2 / tot >= min_purity:
        return 0
    return -1


def pixel_ranges(src, bounds, pad=0):
    """(n,4) map bounds -> (n,4) int pixel ranges r0,r1,c0,c1 (each >= 1 px)."""
    t = src.transform
    minx, miny, maxx, maxy = bounds.T
    c0 = np.floor((minx - t.c) / t.a).astype(np.int64) - pad
    c1 = np.ceil((maxx - t.c) / t.a).astype(np.int64) + pad
    r0 = np.floor((maxy - t.f) / t.e).astype(np.int64) - pad
    r1 = np.ceil((miny - t.f) / t.e).astype(np.int64) + pad
    c1 = np.maximum(c1, c0 + 1)
    r1 = np.maximum(r1, r0 + 1)
    return np.c_[r0, r1, c0, c1]


def chunked_reduce(src, rng, bands, func, out_dim, name, target_mb=256):
    """Apply func(masked (bands,h,w) array) -> (out_dim,) for every pixel range,
    reading the raster once per chunk instead of once per box."""
    from rasterio.windows import Window
    import time
    n = len(rng)
    out = np.full((n, out_dim), np.nan, dtype=np.float64)
    itemsize = np.dtype(src.dtypes[0]).itemsize
    chunk = int(np.sqrt(target_mb * 1e6 / (len(bands) * itemsize)))
    chunk = int(np.clip(chunk, 256, 8192))
    margin = int(max((rng[:, 1] - rng[:, 0]).max(), (rng[:, 3] - rng[:, 2]).max()))
    H, W = src.height, src.width
    cr, cc = rng[:, 0] // chunk, rng[:, 2] // chunk
    keys = cr * 1_000_000 + cc
    order = np.argsort(keys, kind="stable")
    uniq, starts = np.unique(keys[order], return_index=True)
    ends = np.r_[starts[1:], len(order)]
    t0 = time.time()
    for j, (k, s0, s1) in enumerate(zip(uniq, starts, ends)):
        idx = order[s0:s1]
        wr0, wc0 = (k // 1_000_000) * chunk, (k % 1_000_000) * chunk
        wr1, wc1 = wr0 + chunk + margin, wc0 + chunk + margin
        rr0, cc0 = max(wr0, 0), max(wc0, 0)
        rr1, cc1 = min(wr1, H), min(wc1, W)
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        arr = src.read(bands, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0),
                       masked=True)
        arr = np.ma.masked_invalid(arr)
        for i in idx:
            r0, r1, c0, c1 = rng[i]
            r0, c0 = max(r0, rr0) - rr0, max(c0, cc0) - cc0
            r1, c1 = min(r1, rr1) - rr0, min(c1, cc1) - cc0
            if r1 <= r0 or c1 <= c0:
                continue
            out[i] = func(arr[:, r0:r1, c0:c1])
        if (j + 1) % 20 == 0 or j + 1 == len(uniq):
            el = time.time() - t0
            eta = el / (j + 1) * (len(uniq) - j - 1)
            print(f"  {name}: chunk {j + 1}/{len(uniq)}  "
                  f"elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)
    return out


def _mean(sub):
    v = sub.reshape(sub.shape[0], -1).mean(axis=1)
    return np.ma.filled(v.astype(np.float64), np.nan)


def _median(sub):
    v = np.ma.median(sub.reshape(sub.shape[0], -1), axis=1)
    return np.ma.filled(np.ma.asarray(v, dtype=np.float64), np.nan)


def _gt_counts(sub):
    a = np.ma.filled(sub[0], 0)
    return np.array([(a == 1).sum(), (a == 2).sum()], dtype=np.float64)


def features_for(rgb_xy, emb_bounds, rgb, emb, args):
    """RGB at points + embedding mean over bounds -> (rgb (n,3), emb (n,128), groups)."""
    pts = np.c_[rgb_xy, rgb_xy]                       # point -> zero-size bounds
    rgbv = chunked_reduce(rgb, pixel_ranges(rgb, pts, args.center_win // 2),
                          [1, 2, 3], _median, 3, "rgb")
    erng = pixel_ranges(emb, emb_bounds)
    embv = chunked_reduce(emb, erng, list(range(1, 129)), _mean, 128, "emb")
    cx = (emb_bounds[:, 0] + emb_bounds[:, 2]) / 2
    cy = (emb_bounds[:, 1] + emb_bounds[:, 3]) / 2
    t = emb.transform
    er = np.floor((cy - t.f) / t.e).astype(np.int64)
    ec = np.floor((cx - t.c) / t.a).astype(np.int64)
    groups = (er // args.block_px) * 100000 + (ec // args.block_px)
    return rgbv, embv, groups


def to_frame(rgbv, embv, groups, labels, source, bbox_idx):
    df = pd.DataFrame(embv.astype(np.float32), columns=EMB_COLS)
    df[RGB_COLS] = rgbv.astype(np.float32)
    df.insert(0, "group", groups)
    df.insert(0, "source", source)
    df.insert(0, "label", labels)
    df.insert(0, "bbox_idx", bbox_idx)
    return df.dropna(subset=FEAT_COLS).reset_index(drop=True)


def extract_bbox_features(gdf, rgb, emb, gt, args):
    import time
    t0 = time.time()
    c = gdf.to_crs(rgb.crs).geometry.centroid
    rgb_xy = np.c_[c.x.values, c.y.values]
    emb_b = gdf.to_crs(emb.crs).geometry.bounds.values
    gt_b = gdf.to_crs(gt.crs).geometry.bounds.values
    rgbv, embv, groups = features_for(rgb_xy, emb_b, rgb, emb, args)
    cnt = chunked_reduce(gt, pixel_ranges(gt, gt_b), [1], _gt_counts, 2, "gt")
    n1, n2 = np.nan_to_num(cnt[:, 0]), np.nan_to_num(cnt[:, 1])
    tot = n1 + n2
    with np.errstate(invalid="ignore", divide="ignore"):
        labels = np.where(tot == 0, -1,
                 np.where(n1 / tot >= args.min_purity, 1,
                 np.where(n2 / tot >= args.min_purity, 0, -1)))
    df = to_frame(rgbv, embv, groups, labels, "bbox", gdf.index.values)
    print(f"  bbox features: {len(df)}/{len(gdf)} usable "
          f"({(time.time() - t0) / 60:.1f} min)")
    return df


def sample_nonmango_points(gt, n, seed, per_block=5000):
    """Random pixels where GT == 2 -> (xs, ys) in GT CRS."""
    rng = np.random.default_rng(seed)
    pts = []
    for _, win in gt.block_windows(1):
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
    rows = pts[:, 0] + rng.random(len(pts))   # jitter inside the pixel
    cols = pts[:, 1] + rng.random(len(pts))
    xs, ys = gt.transform * (cols, rows)
    return np.asarray(xs), np.asarray(ys)


def extract_random_features(xs, ys, rgb, emb, gt, args):
    from rasterio.warp import transform as warp_transform
    if len(xs) == 0:
        return pd.DataFrame()
    rx, ry = warp_transform(gt.crs, rgb.crs, list(xs), list(ys))
    ex, ey = warp_transform(gt.crs, emb.crs, list(xs), list(ys))
    ex, ey = np.asarray(ex), np.asarray(ey)
    rgbv, embv, groups = features_for(np.c_[rx, ry], np.c_[ex, ey, ex, ey],
                                      rgb, emb, args)
    return to_frame(rgbv, embv, groups, np.zeros(len(xs), int), "random_gt2",
                    np.full(len(xs), -1))


def spatial_order(gdf):
    """Order boxes so each batch covers a compact area (fewer raster reads)."""
    try:
        return np.argsort(gdf.hilbert_distance().values, kind="stable")
    except Exception:
        b = gdf.geometry.bounds
        cx, cy = (b.minx + b.maxx).values / 2, (b.miny + b.maxy).values / 2
        strip = (cy.max() - cy.min()) / 50 or 1.0
        return np.lexsort((cx, np.floor((cy - cy.min()) / strip)))


def build_table_from_rasters(args):
    import glob
    import time
    import rasterio
    import sys
    mc = sys.modules[__name__]   # extraction helpers live in this file

    rgb_path = resolve_raster(args.rgb, "rgb", args.out_dir, 3)
    emb_path = resolve_raster(args.emb, "emb", args.out_dir, 128)
    gdf = read_bboxes(args.bboxes)
    print(f"{len(gdf)} tree boxes in file", flush=True)
    if args.sample_frac < 1:
        gdf = gdf.sample(frac=args.sample_frac, random_state=args.seed)
        print(f"using a random {args.sample_frac:.0%} of boxes: {len(gdf)}",
              flush=True)
    gdf = gdf.reset_index(drop=True)
    gdf = gdf.iloc[spatial_order(gdf)]      # keeps index = bbox id

    bs = args.batch_size
    nb = int(np.ceil(len(gdf) / bs))
    part_dir = os.path.join(args.out_dir, "parts")
    os.makedirs(part_dir, exist_ok=True)
    done = len(glob.glob(os.path.join(part_dir, "bbox_*.csv")))
    if done:
        print(f"resuming: {done} batches already in {part_dir} "
              f"(delete that folder to start fresh)", flush=True)
    print(f"processing {len(gdf)} boxes in {nb} batches of {bs}", flush=True)

    with rasterio.open(rgb_path) as rgb, rasterio.open(emb_path) as emb, \
            rasterio.open(args.gt) as gt:
        assert emb.count == 128, "expected 128 embedding bands"
        t0, n_run = time.time(), 0
        for k in range(nb):
            fp = os.path.join(part_dir, f"bbox_{k:04d}.csv")
            if os.path.exists(fp):
                continue
            part = gdf.iloc[k * bs:(k + 1) * bs]
            df = mc.extract_bbox_features(part, rgb, emb, gt, args)
            df.to_csv(fp + ".tmp", index=False)
            os.replace(fp + ".tmp", fp)          # only complete batches count
            n_run += 1
            el = time.time() - t0
            left = nb - k - 1
            print(f"batch {k + 1}/{nb} done ({len(df)} usable) | "
                  f"elapsed {el / 60:.1f} min | ETA {el / n_run * left / 60:.1f} min",
                  flush=True)

        df_box = pd.concat([pd.read_csv(os.path.join(part_dir, f"bbox_{k:04d}.csv"))
                            for k in range(nb)], ignore_index=True)
        lab = df_box[df_box.label >= 0]
        n_m, n_nm = int((lab.label == 1).sum()), int((lab.label == 0).sum())
        print(f"labelled bboxes: mango={n_m}, non-mango={n_nm}, "
              f"no/ambiguous GT={int((df_box.label < 0).sum())}", flush=True)
        df_rand = pd.DataFrame()
        n_need = int(args.random_ratio * n_m) - n_nm
        if args.random_ratio > 0 and n_need > 0:
            xs, ys = mc.sample_nonmango_points(gt, n_need, args.seed)
            df_rand = mc.extract_random_features(xs, ys, rgb, emb, gt, args)
            print(f"random non-mango samples added: {len(df_rand)}", flush=True)
    return pd.concat([lab, df_rand], ignore_index=True)


# ----------------------------------------------------------------------------
# Importance
# ----------------------------------------------------------------------------
def compute_importance(X, y, groups, args):
    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
    mdi, perm, aucs = [], [], []
    for k, (tr, te) in enumerate(skf.split(X, y, groups)):
        rf = make_rf(args.seed, args.n_jobs).fit(X[tr], y[tr])
        mdi.append(rf.feature_importances_)
        aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
        if not args.no_perm:
            pi = permutation_importance(
                rf, X[te], y[te], scoring="roc_auc",
                n_repeats=args.perm_repeats, random_state=args.seed,
                n_jobs=args.n_jobs,
            )
            perm.append(pi.importances_mean)
        print(f"  fold {k + 1}/{args.folds}: AUC={aucs[-1]:.4f}")
    print(f"RF spatial-CV AUC (all 131 features): "
          f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    mdi = np.vstack(mdi)
    res = pd.DataFrame({
        "feature": FEAT_COLS,
        "group": ["embedding"] * 128 + ["rgb"] * 3,
        "mdi": mdi.mean(0),
        "mdi_std": mdi.std(0),
    })
    if perm:
        perm = np.vstack(perm)
        res["perm_auc_drop"] = perm.mean(0)
        res["perm_std"] = perm.std(0)
    return res


def cv_auc(X, y, groups, args):
    skf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y, groups):
        m = make_rf(args.seed, args.n_jobs).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def legend_handles():
    return [Patch(color=COLORS["embedding"], label="embedding band"),
            Patch(color=COLORS["rgb"], label="RGB")]


def plot_all(res, col, err, path, title):
    d = res.sort_values(col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(26, 7))
    x = np.arange(len(d))
    ax.bar(x, d[col], yerr=d[err], color=d.group.map(COLORS),
           error_kw=dict(lw=0.5, alpha=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels(d.feature, rotation=90, fontsize=6)
    ax.set_xlim(-1, len(d))
    ax.set_ylabel(title)
    ax.set_title(f"{title} — all {len(d)} features (sorted)")

    # cumulative share (only meaningful for non-negative scores)
    pos = d[col].clip(lower=0)
    cum = pos.cumsum() / pos.sum()
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="k", lw=1.2)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative share")
    for t, ls in [(0.80, ":"), (0.90, "--"), (0.95, "-.")]:
        n = int(np.searchsorted(cum.values, t) + 1)
        ax2.axhline(t, color="grey", ls=ls, lw=0.8)
        ax2.axvline(n - 1, color="grey", ls=ls, lw=0.8)
        ax2.text(n, t - 0.05, f"{int(t * 100)}% → {n} feats", fontsize=8)
    ax.legend(handles=legend_handles(), loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top(res, col, err, n, path, title):
    d = res.sort_values(col, ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * n)))
    ax.barh(d.feature, d[col], xerr=d[err], color=d.group.map(COLORS),
            error_kw=dict(lw=0.6, alpha=0.7))
    ax.set_xlabel(title)
    ax.set_title(f"Top {n} features — {title}")
    ax.legend(handles=legend_handles(), loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_important(res, thr, path):
    """Only the features that together cover `thr` of total MDI importance."""
    d = res.sort_values("mdi", ascending=False).copy()
    d["cum"] = d.mdi.cumsum() / d.mdi.sum()
    imp = d[d.cum.shift(fill_value=0) < thr].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(imp))))
    ax.barh(imp.feature, imp.relevance_pct, color=imp.group.map(COLORS))
    ax.set_xlabel("Relevance score (%)")
    ax.set_title(f"Important features ({len(imp)} of {len(d)}, "
                 f"covering {int(thr * 100)}% importance)")
    ax.legend(handles=legend_handles(), loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return imp.feature.iloc[::-1].tolist()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-table", help="train_table.csv from mango_classifier.py")
    ap.add_argument("--bboxes"); ap.add_argument("--rgb")
    ap.add_argument("--emb"); ap.add_argument("--gt")
    ap.add_argument("--out-dir", default="fi_out")
    # same extraction knobs as the reference script (used only in raster mode)
    ap.add_argument("--min-purity", type=float, default=0.6)
    ap.add_argument("--center-win", type=int, default=1)
    ap.add_argument("--random-ratio", type=float, default=1.0)
    ap.add_argument("--block-px", type=int, default=50)
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="use a random fraction of the boxes (0.5 = 50%%)")
    ap.add_argument("--batch-size", type=int, default=25000,
                    help="boxes processed per batch; each batch is saved, so a "
                         "stopped run resumes where it left off")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap training samples (stratified random subset) to "
                         "speed up RF / permutation; 0 = use all")
    # importance settings
    ap.add_argument("--bbox-only", action="store_true",
                    help="drop random GT==2 points, rank on real tree boxes only")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-perm", action="store_true",
                    help="skip permutation importance (much faster)")
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--rank-by", choices=["mdi", "perm"], default="mdi")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--cum-threshold", type=float, default=0.90,
                    help="important_features.png keeps features covering this "
                         "share of total importance (e.g. 0.8, 0.9, 0.95)")
    ap.add_argument("--topk", default="5,10,20,30,50,80,131",
                    help="comma list of k for 'top-k features only' CV check "
                         "('' to skip)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.train_table:
        train = pd.read_csv(args.train_table)
    elif all([args.bboxes, args.rgb, args.emb, args.gt]):
        train = build_table_from_rasters(args)
        tt = os.path.join(args.out_dir, "train_table.csv")
        train.to_csv(tt, index=False)
        print(f"saved {tt} -> next time use --train-table {tt} to skip extraction")
    else:
        ap.error("give --train-table OR all of --bboxes --rgb --emb --gt")

    if args.bbox_only and "source" in train:
        train = train[train.source == "bbox"]
    train = train.dropna(subset=FEAT_COLS)
    if args.max_train and len(train) > args.max_train:
        frac = args.max_train / len(train)
        train = pd.concat([train[train.label == v].sample(frac=frac,
                           random_state=args.seed) for v in (0, 1)],
                          ignore_index=True)
        print(f"capped training set to {len(train)} samples (--max-train)")
    X = train[FEAT_COLS].values.astype(np.float32)
    y = train.label.astype(int).values
    groups = train.group.values
    print(f"samples: {len(y)} (mango={int(y.sum())}, non-mango={int((y == 0).sum())})")
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise SystemExit("need both mango and non-mango samples; check that the "
                         "bboxes overlap the GT mask and GT uses 1=mango, 2=non-mango")
    n_groups = len(np.unique(groups))
    if n_groups < args.folds:
        print(f"only {n_groups} spatial blocks -> using {max(n_groups, 2)} folds "
              f"(lower --block-px for more blocks)")
        args.folds = max(n_groups, 2)

    # 1) relevance scores
    res = compute_importance(X, y, groups, args)
    if args.rank_by == "perm" and "perm_auc_drop" not in res:
        print("permutation disabled -> ranking by mdi")
        args.rank_by = "mdi"
    key = "mdi" if args.rank_by == "mdi" else "perm_auc_drop"
    res["relevance_pct"] = 100 * res.mdi / res.mdi.sum()
    res = res.sort_values(key, ascending=False).reset_index(drop=True)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    res.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False)
    print("\nTop 20:")
    print(res.head(20).to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nRGB ranks:", res[res.group == "rgb"][["feature", "rank"]]
          .to_dict("records"))

    # 2) bar graphs
    plot_all(res, "mdi", "mdi_std",
             os.path.join(args.out_dir, "feature_importance_all.png"),
             "RF importance (MDI)")
    plot_top(res, "mdi", "mdi_std", args.top_n,
             os.path.join(args.out_dir, f"feature_importance_top{args.top_n}.png"),
             "RF importance (MDI)")
    sel = plot_important(res, args.cum_threshold,
                         os.path.join(args.out_dir, "important_features.png"))
    pd.Series(sel, name="feature").to_csv(
        os.path.join(args.out_dir, "important_features.csv"), index=False)
    print(f"\n{len(sel)} features cover {int(args.cum_threshold * 100)}% "
          f"importance -> important_features.png / .csv")
    if "perm_auc_drop" in res:
        plot_top(res, "perm_auc_drop", "perm_std", args.top_n,
                 os.path.join(args.out_dir, f"perm_importance_top{args.top_n}.png"),
                 "Permutation importance (ROC-AUC drop)")

    # 3) does a smaller feature set hold up? (CV AUC with only top-k)
    if args.topk:
        rows = []
        for k in [int(s) for s in args.topk.split(",") if s.strip()]:
            k = min(k, len(FEAT_COLS))
            cols = res.feature.head(k).tolist()
            idx = [FEAT_COLS.index(c) for c in cols]
            auc = cv_auc(X[:, idx], y, groups, args)
            rows.append(dict(k=k, cv_auc=auc))
            print(f"  top-{k:>3} features -> spatial-CV AUC {auc:.4f}")
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "topk_eval.csv"),
                                  index=False)

    print(f"\nSaved CSV + plots in {args.out_dir}")


if __name__ == "__main__":
    main()
