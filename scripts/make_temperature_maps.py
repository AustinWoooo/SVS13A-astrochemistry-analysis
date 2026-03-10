#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 固定路徑
OUT_DIR  = "/almabeegfs/scratch/ssp202525/results/rotational_diagram"
SUMMARY_CSV = os.path.join(OUT_DIR, "rd_summary.csv")
MAP_DIR = os.path.join(OUT_DIR, "temp_maps")
os.makedirs(MAP_DIR, exist_ok=True)



# 只畫哪些旗標
ACCEPT_FLAGS = {"ok","low_R2"}     # 想放寬可用 {"ok","low_R2"}

# 溫度可信範圍（過濾極端/怪值）
T_CLIP = (5.0, 800.0)       # (minK, maxK)，None 代表不設界

# 是否強制在像素上標數字（True=永遠標；False=自動判斷，只在點很少時標）
FORCE_LABELS = False


def pick_temperature(row):
    """優先用 ratio 的中位數；若沒有/NaN，退回 RD 的 Trot_K；最後回 np.nan。"""
    try:
        meth = str(row.get("method", "")).lower()
        t50  = float(row.get("T_ratio_p50", np.nan))
        trot = float(row.get("Trot_K", np.nan))
        t = t50 if (meth == "ratio" and np.isfinite(t50)) else trot
        # 範圍過濾
        if T_CLIP is not None and np.isfinite(t):
            tmin, tmax = T_CLIP
            if t < tmin or t > tmax:
                return np.nan
        return t if np.isfinite(t) else np.nan
    except Exception:
        return np.nan

def make_map_for(mol, comp, df):
    """回傳 (Tmap, nx, ny)，若無點則回 (None,0,0)"""
    sub = df[(df["molecule"]==mol) & (df["component"]==comp)].copy()
    if sub.empty:
        return None, 0, 0
    # 只留通過旗標的
    sub = sub[sub["flag"].isin(ACCEPT_FLAGS)].copy()
    if sub.empty:
        return None, 0, 0
    # 取溫度
    sub["T_use"] = sub.apply(pick_temperature, axis=1)
    sub = sub[np.isfinite(sub["T_use"])]
    if sub.empty:
        return None, 0, 0
    xs = sub["x"].astype(int).to_numpy()
    ys = sub["y"].astype(int).to_numpy()
    Ts = sub["T_use"].astype(float).to_numpy()

    nx = int(xs.max())+1
    ny = int(ys.max())+1
    Tmap = np.full((ny, nx), np.nan)
    for x,y,t in zip(xs,ys,Ts):
        if 0 <= x < nx and 0 <= y < ny:
            Tmap[y, x] = t
    return Tmap, nx, ny

def robust_limits(*arrays, vmin=None, vmax=None):
    """合併多個 array 的有效值做 robust 5–95 百分位範圍；允許外部覆寫。"""
    vals = []
    for a in arrays:
        if a is None: continue
        v = np.asarray(a, float)
        v = v[np.isfinite(v)]
        if v.size: vals.append(v)
    if not vals:
        return (0.0, 1.0)
    v = np.concatenate(vals)
    lo = np.nanpercentile(v, 5.0) if vmin is None else vmin
    hi = np.nanpercentile(v, 95.0) if vmax is None else vmax
    if not np.isfinite(lo): lo = np.nanmin(v)
    if not np.isfinite(hi): hi = np.nanmax(v)
    if hi <= lo:
        pad = 1.0 if np.isfinite(hi) else 1.0
        return (float(lo - pad), float(hi + pad))
    return float(lo), float(hi)

def annotate_sparse(ax, Tmap, max_labels=60):
    """像素很少或強制時，在圖上標整數溫度。"""
    if Tmap is None: return
    ny, nx = Tmap.shape
    coords = np.argwhere(np.isfinite(Tmap))
    if (not FORCE_LABELS) and (coords.shape[0] > max_labels):
        return
    for (y,x) in coords:
        ax.text(x, y, f"{int(round(Tmap[y,x]))}",
                ha="center", va="center", fontsize=7, color="white")


def main():
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"找不到 {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)

    # ---- 健壯化：補欄位 / 型別清理 ----
    for col in ("component","x","y"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "molecule" not in df.columns:
        raise ValueError("summary 缺 'molecule' 欄位")
    df["molecule"] = df["molecule"].astype(str)

    # 缺的欄位補上預設
    if "flag" not in df.columns:        df["flag"] = "ok"
    if "method" not in df.columns:      df["method"] = ""
    for c in ("Trot_K","T_ratio_p16","T_ratio_p50","T_ratio_p84"):
        if c not in df.columns:          df[c] = np.nan
    # -----------------------------------

    molecules = sorted(df["molecule"].dropna().unique())
    if not molecules:
        raise ValueError("summary 裡沒有任何 molecule")

    for mol in molecules:
        # 兩個 component 的貼圖（make_map_for 內部已做 flag/溫度範圍過濾與 pick_temperature）
        T1, nx1, ny1 = make_map_for(mol, 1, df)
        T2, nx2, ny2 = make_map_for(mol, 2, df)

        if (T1 is None) and (T2 is None):
            # 整個分子都沒有可畫的像素 → 產出一份只有說明的 PDF
            pdf_path = os.path.join(MAP_DIR, f"tempmap_{mol}.pdf")
            with PdfPages(pdf_path) as pdf:
                fig = plt.figure(figsize=(6.4, 5.2)); plt.axis("off")
                msg = f"{mol}\n(no valid pixels with flags {sorted(ACCEPT_FLAGS)})"
                plt.text(0.05, 0.9, msg, fontsize=12, va="top")
                pdf.savefig(fig, dpi=160); plt.close(fig)
            print(f"[OK] {pdf_path}")
            continue

        # 同一分子共用色標；若想固定範圍，改成 vmin,vmax=(0,300) 等
        vmin, vmax = robust_limits(T1, T2)

        pdf_path = os.path.join(MAP_DIR, f"tempmap_{mol}.pdf")
        with PdfPages(pdf_path) as pdf:
            for comp, Tmap in ((1, T1), (2, T2)):
                if Tmap is None:
                    fig = plt.figure(figsize=(6.4, 5.2)); plt.axis("off")
                    plt.text(0.05, 0.9,
                             f"{mol} — component {comp}\n(no valid pixels with flags {sorted(ACCEPT_FLAGS)})",
                             fontsize=12, va="top")
                    pdf.savefig(fig, dpi=160); plt.close(fig)
                    continue

                fig, ax = plt.subplots(figsize=(6.4, 5.2))
                im = ax.imshow(Tmap, origin="lower", vmin=vmin, vmax=vmax,
                               cmap="inferno", interpolation="nearest")
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("T (K)")
                ax.set_title(f"{mol} — component {comp}   (flags: {','.join(sorted(ACCEPT_FLAGS))})")
                ax.set_xlabel("x pixel")
                ax.set_ylabel("y pixel")
                annotate_sparse(ax, Tmap, max_labels=60)
                plt.tight_layout()
                pdf.savefig(fig, dpi=160)
                plt.close(fig)

        print(f"[OK] {pdf_path}")

    print(f"[DONE] All temperature maps saved to: {MAP_DIR}")

if __name__ == "__main__":
    main()
