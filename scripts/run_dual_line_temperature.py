#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.backends.backend_pdf import PdfPages


# Constants
KB = 1.380649e-23
H  = 6.62607015e-34
C  = 2.99792458e8
SQRT_2PI = math.sqrt(2.0 * math.pi)

FIT_DIR  = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/joint_fitting_2G"
LINE_CSV = "/almabeegfs/scratch/ssp202525/data/molecular_line/line.csv"
OUT_DIR  = "/almabeegfs/scratch/ssp202525/results/rd_dual"

AMP_MIN = 1e-7
MIN_POINTS_RD = 3
EU_TOL = 0.5
FRAC_ERR = 0.20
MC_N = 200000
R2_MIN = 0.80
TMIN, TMAX = 5.0, 800.0
SAME_AXES = True

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rd_points"), exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------
def normalize_molecule_key(row):
    mol = str(row.get("molecular") or "").strip()
    name = str(row.get("name") or "").strip().lower()
    if ("acetaldehyde" in name) or mol.upper().startswith("CH3CHO"):
        return "acetaldehyde"
    if ("methyl" in name and "formate" in name) or mol.upper().startswith("CH3OCHO"):
        return "methyl_formate"
    if ("glycolaldehyde" in name) or mol.upper().startswith("CH2(OH)CHO"):
        return "glycolaldehyde"
    if ("ethylene glycol" in name) or "agg" in mol.lower() or "(ch2oh)2" in mol.lower():
        return "ethylene_glycol"
    if ("propanenitrile" in name) or mol.upper().startswith("C2H5CN"):
        return "propanenitrile"
    return re.sub(r"\s+", "_", name) if name else re.sub(r"[^A-Za-z0-9]+", "_", mol).lower()

def freq_tag(molkey, nu_GHz):
    code = int(round(float(nu_GHz) * 1e4))
    return f"{molkey}_{code}"

def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def integrated_intensity(amp, fwhm_kms=None, sigma_kms=None):
    if amp is None or (fwhm_kms is None and sigma_kms is None):
        return None
    if fwhm_kms is not None and np.isfinite(fwhm_kms):
        return 1.064467 * float(amp) * float(fwhm_kms)
    if sigma_kms is not None and np.isfinite(sigma_kms):
        return SQRT_2PI * float(amp) * float(sigma_kms)
    return None

def Nu_from_area(nu_GHz, Aul_s, I_K_kms):
    # Nu = (8πk ν^2)/(h c^3 Aul) ∫Tb dv ; ν in Hz; ∫Tb dv in K m/s
    nu_Hz = float(nu_GHz) * 1e9
    I_K_ms = float(I_K_kms) * 1e3
    val = (8.0 * math.pi * KB * nu_Hz**2) / (H * C**3 * float(Aul_s)) * I_K_ms
    return val if np.isfinite(val) and val > 0 else None

def wls_fit(x, y, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if w is None: w = np.ones_like(x)
    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    XtWX = X.T @ W @ X
    beta = np.linalg.inv(XtWX) @ (X.T @ W @ y)
    a, b = beta[0], beta[1]
    yhat = a + b*x
    ybar = np.sum(w*y)/np.sum(w)
    ss_res = np.sum(w*(y - yhat)**2)
    ss_tot = np.sum(w*(y - ybar)**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, R2

def fit_T_from_line(x, y):
    a, b, R2 = wls_fit(x, y, w=None)
    if not (np.isfinite(b) and b < 0):
        return np.nan, a, b, R2
    return -1.0/b, a, b, R2

def merge_same_Eu(points, tol=0.5):
    """points: list of dict(Eu_K, ln_Nu_over_gu, ...). Merge by Eu within tol (K)."""
    if not points: return []
    pts = sorted(points, key=lambda p: p["Eu_K"])
    out = []
    cur_Eu = pts[0]["Eu_K"]; Nu_sum = 0.0
    for p in pts:
        if abs(p["Eu_K"] - cur_Eu) <= tol:
            Nu_sum += math.exp(p["ln_Nu_over_gu"])
        else:
            out.append({"Eu_K": cur_Eu, "ln_Nu_over_gu": math.log(Nu_sum)})
            cur_Eu = p["Eu_K"]; Nu_sum = math.exp(p["ln_Nu_over_gu"])
    out.append({"Eu_K": cur_Eu, "ln_Nu_over_gu": math.log(Nu_sum)})
    return out

def two_line_T(Eu1, Nu1, gu1, Eu2, Nu2, gu2, allow_reverse=True):
    # 按能階排序：El < Eh
    if Eu1 <= Eu2:
        El, Nl, gl = Eu1, Nu1, gu1
        Eh, Nh, gh = Eu2, Nu2, gu2
    else:
        El, Nl, gl = Eu2, Nu2, gu2
        Eh, Nh, gh = Eu1, Nu1, gu1

    # 低Eu / 高Eu
    r = (Nl/gl) / (Nh/gh)
    if r <= 0:
        return math.nan, False
    lr = math.log(r)
    if not np.isfinite(lr) or lr <= 0:
        # 高Eu線沒有變弱（或 r≈1），物理上不該給正溫
        return math.nan, False

    T = (Eh - El) / lr
    return (T if np.isfinite(T) and T > 0 else math.nan), True


def best_T_from_pairs(lines, frac_err=0.2, n=2000, seed=0):
    """
    lines: list of dicts with {'Eu':K, 'Nu':, 'gu':}
    Return: dict with T (best ΔEu pair), and MC percentiles if possible.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            T, ok = two_line_T(lines[i]["Eu"], lines[i]["Nu"], lines[i]["gu"],
                               lines[j]["Eu"], lines[j]["Nu"], lines[j]["gu"])
            if ok:
                pairs.append((abs(lines[j]["Eu"]-lines[i]["Eu"]), (i,j), T))
    if not pairs:
        return {"ok": False, "T": np.nan}
    pairs.sort(reverse=True, key=lambda t: t[0])
    _, (i, j), T0 = pairs[0]
    out = {"ok": True, "pair": (i, j), "T": T0}
    # Monte Carlo on multiplicative errors of Nu -> additive in ln-space
    Ts = []
    for _ in range(n):
        lns = []
        for L in lines:
            jitter = math.exp(rng.normal(0.0, frac_err))
            lns.append({"Eu": L["Eu"], "Nu": L["Nu"]*jitter, "gu": L["gu"]})
        Tmc, okmc = two_line_T(lns[i]["Eu"], lns[i]["Nu"], lns[i]["gu"],
                               lns[j]["Eu"], lns[j]["Nu"], lns[j]["gu"])
        if okmc: Ts.append(Tmc)
    if Ts:
        T16, T50, T84 = np.percentile(Ts, [16,50,84])
        out.update({"T16": T16, "T50": T50, "T84": T84})
    return out

@dataclass
class RDPlotItem:
    Eu: np.ndarray
    lnNuGu: np.ndarray
    title: str
    trend_x: np.ndarray | None
    trend_y: np.ndarray | None

def qc_flag(n, R2, T, tmin, tmax):
    if n < 2: return "n<2"
    if not (np.isfinite(T) and tmin <= T <= tmax): return "T_out"
    if np.isfinite(R2) and R2 < R2_MIN: return "low_R2"
    return "ok"

def molecule_axes_limits(items):
    xs, ys = [], []
    for it in items:
        if it.Eu.size > 0:
            xs.append(it.Eu); ys.append(it.lnNuGu)
    if not xs: return None, None
    x_all = np.concatenate(xs); y_all = np.concatenate(ys)
    xr = (float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    yr = (float(np.nanmin(y_all)), float(np.nanmax(y_all)))
    pad_x = 0.05*(xr[1]-xr[0] if xr[1]>xr[0] else 1)
    pad_y = 0.08*(yr[1]-yr[0] if yr[1]>yr[0] else 1)
    return (xr[0]-pad_x, xr[1]+pad_x), (yr[0]-pad_y, yr[1]+pad_y)

# ----------------------------
# Load line catalog
# ----------------------------
raw = pd.read_csv(LINE_CSV)

# Aij fallback from logAij if present
if ("Aij" in raw.columns) and ("log(Aij)" in raw.columns):
    m = raw["Aij"].isna() & raw["log(Aij)"].notna()
    raw.loc[m, "Aij"] = (10.0 ** raw.loc[m, "log(Aij)"].astype(float))

colmap = {
    "restfreq(GHz)": "restfreq_GHz",
    "E_up(K)": "E_u",
    "E_low(K)": "E_l",
    "g_u": "g_u",
    "Aij": "Aij",
    "Linelist": "linelist",
    "name": "name",
    "molecular": "molecular",
    "window": "window",
    "log(Aij)": "logAij",
}
for k, v in colmap.items():
    if k in raw.columns and v not in raw.columns:
        raw[v] = raw[k]

line_rows = []
for _, r in raw.iterrows():
    if pd.isna(r.get("restfreq_GHz")) or pd.isna(r.get("E_u")):
        continue
    molkey = normalize_molecule_key(r)
    tag = freq_tag(molkey, float(r["restfreq_GHz"]))
    gu = r.get("g_u"); Aij = r.get("Aij")
    line_rows.append({
        "molkey": molkey,
        "tag": tag,
        "freq_GHz": float(r["restfreq_GHz"]),
        "Eu_K": float(r["E_u"]),
        "gu": np.nan if pd.isna(gu) else float(gu),
        "Aij": np.nan if pd.isna(Aij) else float(Aij),
    })
line_df = pd.DataFrame(line_rows)

# ----------------------------
# Iterate fit files
# ----------------------------
fit_files = [f for f in os.listdir(FIT_DIR) if f.startswith("fit_results_") and f.endswith("_joint.csv")]
if not fit_files:
    raise FileNotFoundError(f"No fit_results_*_joint.csv under {FIT_DIR}")


summary = []
plots_per_molecule = {}

R_A1 = re.compile(r"A1\[(.+)\]")
R_A2 = re.compile(r"A2\[(.+)\]")

for fname in sorted(fit_files):
    molkey = fname.replace("fit_results_", "").replace("_joint.csv", "")
    fpath = os.path.join(FIT_DIR, fname)
    df = pd.read_csv(fpath)

    if "model" in df.columns:
        df["model_norm"] = df["model"].astype(str).str.upper()
        df = df[df["model_norm"].isin(["1G","2G"])].copy()
        if df.empty:
            print(f"[INFO] no 1G/2G rows for {molkey}, skip.")
            continue

    COL_FWHM1 = pick_col(df, ["FWHM1 (km/s)", "FWHM1", "FWHM1_kms", "FWHM1_km_s"])
    COL_FWHM2 = pick_col(df, ["FWHM2 (km/s)", "FWHM2", "FWHM2_kms", "FWHM2_km_s"])
    COL_SIGMA1 = pick_col(df, ["σ1 (km/s)", "sigma1 (km/s)", "σ1", "sigma1"])
    COL_SIGMA2 = pick_col(df, ["σ2 (km/s)", "sigma2 (km/s)", "σ2", "sigma2"])
    COL_X = pick_col(df, ["x","X","ix","i"])
    COL_Y = pick_col(df, ["y","Y","jy","j"])

    my_lines = line_df[line_df["molkey"] == molkey].copy()
    if my_lines.empty:
        print(f"[WARN] line catalog has no entries for {molkey}, skip.")
        continue

    tag2Eu = {r["tag"]: r["Eu_K"] for _, r in my_lines.iterrows()}
    A1_cols = [f"A1[{t}]" for t in my_lines["tag"] if f"A1[{t}]" in df.columns]
    A2_cols = [f"A2[{t}]" for t in my_lines["tag"] if f"A2[{t}]" in df.columns]
    A1_cols.sort(key=lambda c: tag2Eu.get(R_A1.match(c).group(1), np.inf)) if A1_cols else None
    A2_cols.sort(key=lambda c: tag2Eu.get(R_A2.match(c).group(1), np.inf)) if A2_cols else None

    if not A1_cols and not A2_cols:
        print(f"[INFO] no A1/A2 columns for {molkey}, skip.")
        continue

    ldict = {r["tag"]: r for _, r in my_lines.iterrows()}
    mol_plots = []

# >>> PATCH 1: 取出 model，1G 只跑 comp1；2G 跑 comp1、comp2
    for _, r in df.iterrows():
        x = int(r[COL_X]) if COL_X and not pd.isna(r[COL_X]) else None
        y = int(r[COL_Y]) if COL_Y and not pd.isna(r[COL_Y]) else None

        model_str = str(r.get("model", "NA"))
        model_norm = model_str.upper()

        # 1G 通常只有第一成分有效，2G 才同時有兩個成分
        comps = (1,) if model_norm == "1G" else (1, 2)

        for comp in comps:
            cols = A1_cols if comp == 1 else A2_cols
            if not cols:
                continue

            fwhm = r[COL_FWHM1] if (comp == 1 and COL_FWHM1) else (r[COL_FWHM2] if (comp == 2 and COL_FWHM2) else None)
            sigma = r[COL_SIGMA1] if (comp == 1 and COL_SIGMA1) else (r[COL_SIGMA2] if (comp == 2 and COL_SIGMA2) else None)

            raw_pts = []
            lines_for_ratio = []   # for two-line method, we need Nu (not ln)
            for col in cols:
                amp = r[col]
                if not np.isfinite(amp) or amp <= AMP_MIN:
                    continue
                m = R_A1.match(col) if comp==1 else R_A2.match(col)
                tag = m.group(1)
                lr = ldict.get(tag)
                if lr is None:
                    continue

                I_k_kms = integrated_intensity(float(amp),
                                               fwhm_kms=fwhm if (fwhm is not None and np.isfinite(fwhm)) else None,
                                               sigma_kms=sigma if (sigma is not None and np.isfinite(sigma)) else None)
                if I_k_kms is None or I_k_kms <= 0:
                    continue

                gu = lr["gu"]; Aij = lr["Aij"]
                if (gu is None) or (Aij is None) or (np.isnan(gu) or np.isnan(Aij)):
                    continue

                Nu = Nu_from_area(lr["freq_GHz"], Aij, I_k_kms)
                if Nu is None: continue

                raw_pts.append({"Eu_K": lr["Eu_K"], "ln_Nu_over_gu": math.log(Nu/gu),
                                "freq_GHz": lr["freq_GHz"], "tag": tag, "I_K_kms": I_k_kms})
                lines_for_ratio.append({"Eu": lr["Eu_K"], "Nu": Nu, "gu": gu})
            
            # ====== Debug: 檢查為什麼會 insufficient（特別是 ethylene_glycol）======
            DEBUG = (molkey == "ethylene_glycol")
            if DEBUG:
                print(f"[DBG] mol={molkey} model={model_str} comp={comp} @(x={x},y={y})")
                print(f"      cols(用來取振幅) = {cols}")

                # 看每個原始點到底存了什麼
                for p in raw_pts:
                    print(f"      line tag={p['tag']}  Eu={p['Eu_K']:.3f}  ln(Nu/gu)={p['ln_Nu_over_gu']:.6f}")

                # 如果可以做 ratio（至少 2 條），先用 ΔEu 最大的那對算一次 T，印出來
                if len(lines_for_ratio) >= 2:
                    lines_sorted = sorted(
                        [(L['Eu'], idx) for idx, L in enumerate(lines_for_ratio)],
                        key=lambda z: z[0]
                    )
                    i = lines_sorted[0][1]
                    j = lines_sorted[-1][1]
                    L1, L2 = lines_for_ratio[i], lines_for_ratio[j]
                    T_try, ok_try = two_line_T(L1["Eu"], L1["Nu"], L1["gu"], L2["Eu"], L2["Nu"], L2["gu"], allow_reverse=True)
                    print(f"      try-ratio with ΔEu={abs(L2['Eu']-L1['Eu']):.3f}K -> T={T_try}  ok={ok_try}")
                else:
                    print(f"      lines_for_ratio 條數 = {len(lines_for_ratio)}（<2，無法做 ratio）")

                # 額外把每條線的細節都印出來（檢查是 Aij/gu 還是面積出問題）
                for L in lines_for_ratio:
                    print(f"      DETAIL: Eu={L['Eu']:.3f}  Nu={L['Nu']:.6e}  gu={L['gu']}")
            # ====== Debug end ======

            

            
            points_df = pd.DataFrame(raw_pts)
            points_name = f"{molkey}_comp{comp}_x{x}_y{y}.csv"
            points_path = os.path.join(OUT_DIR, "rd_points", points_name)
            points_df.to_csv(points_path, index=False)

            xv = points_df["Eu_K"].to_numpy(float) if len(points_df)>0 else np.array([])
            yv = points_df["ln_Nu_over_gu"].to_numpy(float) if len(points_df)>0 else np.array([])
            order = np.argsort(xv); xv, yv = xv[order], yv[order]

            a = b = R2 = np.nan
            T_rd = np.nan
            T16 = T50 = T84 = np.nan
            method = "none"

            if len(xv) >= MIN_POINTS_RD:
                T_rd, a, b, R2 = fit_T_from_line(xv, yv)
                method = "RD"
            elif len(xv) >= 2:
                res = best_T_from_pairs(lines_for_ratio, frac_err=FRAC_ERR, n=MC_N, seed=0)
                if res.get("ok", False):
                    T_rd = res["T"]
                    T16 = res.get("T16", np.nan)
                    T50 = res.get("T50", np.nan)
                    T84 = res.get("T84", np.nan)
                    method = "ratio"
                else:
                    method = "insufficient"

            if np.isfinite(T_rd) and method == "RD" and np.isfinite(b) and b < 0 and len(xv)>0:
                xline = np.linspace(xv.min(), xv.max(), 100)
                yline = a + b*xline
            else:
                xline = yline = None

            flag = qc_flag(len(xv), R2, T_rd, TMIN, TMAX)

            if method == "ratio" and np.isfinite(T50):
                title = f"{molkey} c{comp} @({x},{y}) [{model_str}]  T~{T50:.0f}K [{int(T16):d},{int(T84):d}]  n={len(xv)}  ({method})"
            elif np.isfinite(T_rd):
                title = f"{molkey} c{comp} @({x},{y}) [{model_str}]  T={T_rd:.0f}K  R²={R2:.2f}  n={len(xv)}  ({method})"
            else:
                title = f"{molkey} c{comp} @({x},{y}) [{model_str}]  (n={len(xv)}; {method})"


            mol_plots.append(RDPlotItem(Eu=xv, lnNuGu=yv, title=f"{title} [{flag}]", trend_x=xline, trend_y=yline))

            
            summary.append({
            "molecule": molkey, "component": comp, "x": x, "y": y,
            "model": model_str,  # <--- 新增
            "n_points": len(xv), "method": method,
            "Trot_K": T_rd,
            "T_ratio_p16": T16, "T_ratio_p50": T50, "T_ratio_p84": T84,
            "slope": b, "intercept": a, "R2": R2, "flag": flag
        })

    plots_per_molecule[molkey] = mol_plots


def _score(rec):
    method_rank = {"RD":3, "ratio":2, "insufficient":1, "none":0}.get(rec.get("method"), 0)
    r2 = rec.get("R2")
    r2 = -1.0 if (r2 is None or (isinstance(r2, float) and np.isnan(r2))) else float(r2)
    npts = int(rec.get("n_points", 0))
    # ratio 的相對不確定度（愈小愈好）；非 ratio 給一個較差的固定值
    T16 = rec.get("T_ratio_p16"); T50 = rec.get("T_ratio_p50"); T84 = rec.get("T_ratio_p84")
    if rec.get("method") == "ratio" and all(np.isfinite([T16, T50, T84])) and T50 and T50 > 0:
        unc = (T84 - T16) / T50
        unc_score = -float(unc)  # 越小越好 → 分數越大
    else:
        unc_score = -1e9
    # 回傳一個 tuple，python 會逐項比較（越大越好）
    return (method_rank, r2, npts, unc_score)

best = {}
for rec in summary:
    key = (rec["molecule"], rec["x"], rec["y"], rec["component"])
    if key not in best or _score(rec) > _score(best[key]):
        best[key] = rec

summary_df = pd.DataFrame(list(best.values()))
summary_df.to_csv(os.path.join(OUT_DIR, "rd_summary.csv"), index=False)


for molkey, items in plots_per_molecule.items():
    if not items:
        continue
    pdf_path = os.path.join(OUT_DIR, f"rd_plots_{molkey}.pdf")
    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(8.4, 6.4)); plt.axis("off")
        txt = [f"Rotational Diagram — {molkey}",
       f"Rules: merge |ΔEu|≤{EU_TOL}K; RD if n≥{MIN_POINTS_RD}; else two-line ratio (MC {MC_N}, frac_err={FRAC_ERR*100:.0f}%).",
       f"QC: R²≥{R2_MIN}, {TMIN}≤T≤{TMAX} K."]

        plt.text(0.05, 0.9, "\n".join(txt), fontsize=13, va="top")
        pdf.savefig(fig, dpi=160); plt.close(fig)

        xlim = ylim = None
        if SAME_AXES:
            xlim, ylim = molecule_axes_limits(items)

        for i in range(0, len(items), 4):
            chunk = items[i:i+4]
            fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
            axes = axes.flatten()
            for ax_idx in range(4):
                ax = axes[ax_idx]
                if ax_idx < len(chunk):
                    it = chunk[ax_idx]
                    if it.Eu.size > 0:
                        ax.scatter(it.Eu, it.lnNuGu, s=24)
                        if it.trend_x is not None and it.trend_y is not None:
                            ax.plot(it.trend_x, it.trend_y, lw=1.6)
                    ax.set_title(it.title, fontsize=9)
                    ax.set_xlabel(r"$E_u$ (K)")
                    ax.set_ylabel(r"$\ln(N_u/g_u)$")
                    if xlim: ax.set_xlim(*xlim)
                    if ylim: ax.set_ylim(*ylim)
                else:
                    ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, dpi=160)
            plt.close(fig)

print("[DONE] Outputs under:", OUT_DIR)
