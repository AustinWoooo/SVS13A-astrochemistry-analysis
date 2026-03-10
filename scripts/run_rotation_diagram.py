#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# Paths
# ----------------------------
FIT_DIR  = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/joint_fitting_2G"
LINE_CSV = "/almabeegfs/scratch/ssp202525/data/molecular_line/line.csv"
OUT_DIR  = "/almabeegfs/scratch/ssp202525/results/rotational_diagram"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "rd_points"), exist_ok=True)
# 這次不逐張存 PNG，改集中到每分子的 PDF

# ----------------------------
# Physics constants
# ----------------------------
KB = 1.380649e-23
H  = 6.62607015e-34
C  = 2.99792458e8
SQRT_2PI = math.sqrt(2.0 * math.pi)

# ----------------------------
# Config
# ----------------------------
USE_JY_INPUT = False
DEFAULT_BEAM_BMAJ_ARCSEC = None
DEFAULT_BEAM_BMIN_ARCSEC = None
AMP_MIN = 1e-6
ONLY_USE_2G_ROWS = True
MIN_POINTS_TO_FIT = 2

# ----------------------------
# Helpers
# ----------------------------
def Jy2Tbri(I_jy, bmaj_arcsec, bmin_arcsec, nu_GHz):
    return 1.222e6 * I_jy / ((nu_GHz**2) * bmaj_arcsec * bmin_arcsec)

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

def Nu_over_gu(Eu_K, gu, Aul_s, nu_GHz, I_K_kms):
    if not (np.isfinite(Eu_K) and np.isfinite(gu) and gu > 0 and
            np.isfinite(Aul_s) and Aul_s > 0 and np.isfinite(nu_GHz) and
            np.isfinite(I_K_kms) and I_K_kms > 0):
        return None
    nu_Hz = float(nu_GHz) * 1e9
    I_K_ms = float(I_K_kms) * 1e3
    val = (8.0 * math.pi * KB * nu_Hz**2) / (H * C**3 * float(Aul_s) * float(gu)) * I_K_ms
    if not np.isfinite(val) or val <= 0:
        return None
    return val

def wls_fit(x, y, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if w is None:
        w = np.ones_like(x)
    W = np.diag(w)
    X = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    a, b = beta[0], beta[1]
    yhat = a + b*x
    ss_res = np.sum(w*(y - yhat)**2)
    ybar  = np.sum(w*y)/np.sum(w)
    ss_tot = np.sum(w*(y - ybar)**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, R2

@dataclass
class RDPlotItem:
    # what we need to draw one panel
    Eu: np.ndarray
    lnNuGu: np.ndarray
    title: str
    trend_x: np.ndarray | None
    trend_y: np.ndarray | None

# ----------------------------
# Load line catalog
# ----------------------------
raw = pd.read_csv(LINE_CSV)
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
# 用於每分子 PDF 的 plot item 收集器
plots_per_molecule = {}

for fname in sorted(fit_files):
    molkey = fname.replace("fit_results_", "").replace("_joint.csv", "")
    fpath = os.path.join(FIT_DIR, fname)
    df = pd.read_csv(fpath)

    if ONLY_USE_2G_ROWS and "model" in df.columns:
        df = df[df["model"].astype(str).str.upper() == "2G"].copy()
        if df.empty:
            print(f"[INFO] no 2G rows for {molkey}, skip.")
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

    A1_cols = [f"A1[{t}]" for t in my_lines["tag"] if f"A1[{t}]" in df.columns]
    A2_cols = [f"A2[{t}]" for t in my_lines["tag"] if f"A2[{t}]" in df.columns]
    if not A1_cols and not A2_cols:
        print(f"[INFO] no A1/A2 columns for {molkey}, skip.")
        continue

    ldict = {r["tag"]: r for _, r in my_lines.iterrows()}
    # 收集此分子的所有圖面項
    mol_plots = []

    for _, r in df.iterrows():
        x = int(r[COL_X]) if COL_X and not pd.isna(r[COL_X]) else None
        y = int(r[COL_Y]) if COL_Y and not pd.isna(r[COL_Y]) else None

        for comp in (1, 2):
            cols = A1_cols if comp == 1 else A2_cols
            if not cols:
                continue
            fwhm = r[COL_FWHM1] if (comp == 1 and COL_FWHM1) else (r[COL_FWHM2] if (comp == 2 and COL_FWHM2) else None)
            sigma = r[COL_SIGMA1] if (comp == 1 and COL_SIGMA1) else (r[COL_SIGMA2] if (comp == 2 and COL_SIGMA2) else None)

            pts = []
            for col in cols:
                amp = r[col]
                if not np.isfinite(amp) or amp <= AMP_MIN:
                    continue
                tag = re.match(r"A[12]\[(.+)\]", col).group(1)
                lr = ldict.get(tag)
                if lr is None:
                    continue
                amp_K = float(amp)
                if USE_JY_INPUT:
                    if DEFAULT_BEAM_BMAJ_ARCSEC is None or DEFAULT_BEAM_BMIN_ARCSEC is None:
                        raise ValueError("Set beam for Jy->K")
                    amp_K = Jy2Tbri(amp_K, DEFAULT_BEAM_BMAJ_ARCSEC, DEFAULT_BEAM_BMIN_ARCSEC, lr["freq_GHz"])

                I_k_kms = integrated_intensity(amp_K,
                                               fwhm_kms=fwhm if (fwhm is not None and np.isfinite(fwhm)) else None,
                                               sigma_kms=sigma if (sigma is not None and np.isfinite(sigma)) else None)
                if I_k_kms is None or I_k_kms <= 0:
                    continue
                gu = lr["gu"]; Aij = lr["Aij"]
                if (gu is None) or (Aij is None) or (np.isnan(gu) or np.isnan(Aij)):
                    continue
                Nu_gu = Nu_over_gu(lr["Eu_K"], gu, Aij, lr["freq_GHz"], I_k_kms)
                if Nu_gu is None:
                    continue
                pts.append({
                    "Eu_K": lr["Eu_K"],
                    "ln_Nu_over_gu": math.log(Nu_gu),
                    "freq_GHz": lr["freq_GHz"],
                    "tag": tag,
                    "I_K_kms": I_k_kms
                })

            # 存 RD 點 (即使 <2 點也存)
            points_df = pd.DataFrame(pts)
            points_name = f"{molkey}_comp{comp}_x{x}_y{y}.csv"
            points_path = os.path.join(OUT_DIR, "rd_points", points_name)
            points_df.to_csv(points_path, index=False)

            # 擬合
            if len(points_df) >= MIN_POINTS_TO_FIT:
                xv = points_df["Eu_K"].values.astype(float)
                yv = points_df["ln_Nu_over_gu"].values.astype(float)
                a, b, R2 = wls_fit(xv, yv, w=None)
                Trot = -1.0/b if (np.isfinite(b) and b < 0) else np.nan
                # 線條
                xline = np.linspace(xv.min(), xv.max(), 100)
                yline = a + b*xline
                title = f"{molkey} comp{comp} @({x},{y})  Trot={Trot:.0f} K  R²={R2:.2f}  n={len(points_df)}"
            else:
                a = b = R2 = np.nan
                Trot = np.nan
                xline = yline = None
                title = f"{molkey} comp{comp} @({x},{y})  (n={len(points_df)})"

            # 收集到此分子的 PDF 佇列
            mol_plots.append(RDPlotItem(
                Eu=points_df["Eu_K"].values if len(points_df)>0 else np.array([]),
                lnNuGu=points_df["ln_Nu_over_gu"].values if len(points_df)>0 else np.array([]),
                title=title,
                trend_x=xline,
                trend_y=yline
            ))

            # 摘要
            summary.append({
                "molecule": molkey, "component": comp, "x": x, "y": y,
                "n_points": len(points_df), "Trot_K": Trot, "slope": b, "intercept": a, "R2": R2
            })

    # 收好
    plots_per_molecule[molkey] = mol_plots

# ----------------------------
# 寫出 summary
# ----------------------------
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(OUT_DIR, "rd_summary.csv"), index=False)

# ----------------------------
# 產生每分子的 PDF（每頁 4 張圖，2x2）
# ----------------------------
for molkey, items in plots_per_molecule.items():
    if not items:
        continue
    pdf_path = os.path.join(OUT_DIR, f"rd_plots_{molkey}.pdf")
    with PdfPages(pdf_path) as pdf:
        # 每 4 張為一頁
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
                else:
                    ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, dpi=160)
            plt.close(fig)

print("[DONE] RD points  ->", os.path.join(OUT_DIR, "rd_points"))
print("[DONE] RD summary ->", os.path.join(OUT_DIR, "rd_summary.csv"))
print("[DONE] PDFs       ->", os.path.join(OUT_DIR, "rd_plots_<molecule>.pdf"))
print("每個分子一份 PDF，2x2 佈局，一張圖代表一個像素的單一 component（G1 或 G2）的 RD。")
