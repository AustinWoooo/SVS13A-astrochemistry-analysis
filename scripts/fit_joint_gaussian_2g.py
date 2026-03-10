#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from astropy import units as u
from astropy.coordinates import Angle
import os
import glob
from scipy.optimize import differential_evolution
from mpl_toolkits.axes_grid1 import make_axes_locatable
import aplpy  
import matplotlib.lines as mlines

# --- APLpy map settings ---
M0_DIR    = "/almabeegfs/scratch/ssp202525/moment_fig/mom0/third_carta_channels"  
CENTER_RA  = 52.265666   # (deg)
CENTER_DEC = 31.267669   # (deg)
FOV_DEG    = 0.00020     # ~3"
M0_VMIN, M0_VMAX = 0.0, 0.11   # Jy/beam·km/s 顏色條範圍（可依每條線調整）

# VLA positions
VLA4A_RA, VLA4A_DEC = 52.265610, 31.267694
VLA4B_RA, VLA4B_DEC = 52.265716, 31.267694

OUT_DIR      = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/joint_fitting_2G"
SUBIMAGE_DIR = "/almabeegfs/scratch/ssp202525/data/subimage"
M1_DIR      = "/almabeegfs/scratch/ssp202525/moment_fig/mom1"
SELECTED_DIR = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/select_target"
TEMP_CSV = "/almabeegfs/scratch/ssp202525/results/rd_dual/rd_summary.csv"

os.makedirs(OUT_DIR, exist_ok=True)

def _read_temperature_table(csv_path=TEMP_CSV):
    if not os.path.exists(csv_path):
        print(f"[TEMP] file not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    # 標準化欄名（保險起見）
    df.columns = [c.strip() for c in df.columns]
    # 只留我們在圖上要顯示的成功點
    df = df[df["flag"]=="ok"].copy()
    # 方便後續查詢
    key_cols = ["molecule","component","x","y","method"]
    for k in ["x","y","component"]:
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("Int64")
    return df

TEMP_TABLE = _read_temperature_table()

def format_T_row(row):
    """
    依 CSV 一列資料輸出溫度字串。
    - RD：優先顯示 T±err（若有誤差欄位），否則顯示 T 與 n_points。
    - ratio：用 p16/p50/p84，誤差取 (p84 - p16)/2，輸出 T±err。
    """
    m = str(row["method"]).lower()

    # 小工具：找可能的 RD 誤差欄位名（若你表裡有不同欄名可在此補上）
    def _find_rd_err(r):
        for cand in ["Trot_err_K", "Trot_sigma", "Trot_unc_K", "Trot_dK", "Trot_eK", "Trot_err"]:
            if cand in r.index and pd.notna(r[cand]):
                try:
                    return float(r[cand])
                except Exception:
                    pass
        return None

    if m == "rd" and pd.notna(row.get("Trot_K")):
        t = round(float(row["Trot_K"]))
        err = _find_rd_err(row)
        n = int(row.get("n_points", 0)) if pd.notna(row.get("n_points")) else 0
        if err is not None and np.isfinite(err):
            return f"T={t}±{int(round(err))} K (RD, n={n})"
        else:
            return f"T={t} K (RD, n={n})"

    if m == "ratio" and pd.notna(row.get("T_ratio_p50")):
        p50 = float(row["T_ratio_p50"])
        t   = int(round(p50))
        p16 = row.get("T_ratio_p16")
        p84 = row.get("T_ratio_p84")
        # 盡量用對稱誤差；若只給一邊，就用那一邊的絕對值
        if pd.notna(p16) and pd.notna(p84):
            err = 0.5 * abs(float(p84) - float(p16))
            return f"T={t}±{int(round(err))} K (ratio)"
        elif pd.notna(p84):
            err = abs(float(p84) - p50)
            return f"T={t}±{int(round(err))} K (ratio)"
        elif pd.notna(p16):
            err = abs(p50 - float(p16))
            return f"T={t}±{int(round(err))} K (ratio)"
        else:
            # 後備：若有單一誤差欄
            if "T_ratio_err" in row.index and pd.notna(row["T_ratio_err"]):
                return f"T={t}±{int(round(float(row['T_ratio_err'])))} K (ratio)"
            return f"T={t} K (ratio)"

    return None


def get_T_labels_for_point(mol, x, y):
    """
    回傳 {1: '...', 2: '...'}；若沒結果或 flag!=ok 就沒有那個 key。
    RD 優先，沒有 RD 才用 ratio。
    """
    labels = {}
    if TEMP_TABLE is None:
        return labels
    sub = TEMP_TABLE[(TEMP_TABLE["molecule"]==mol) &
                     (TEMP_TABLE["x"]==x) &
                     (TEMP_TABLE["y"]==y)].copy()
    if sub.empty:
        return labels
    # RD 優先
    for comp in (1,2):
        rows = sub[sub["component"]==comp]
        if rows.empty: 
            continue
        pick = None
        rd = rows[rows["method"].str.lower()=="rd"]
        if not rd.empty:
            pick = rd.iloc[0]
        else:
            ratio = rows[rows["method"].str.lower()=="ratio"]
            if not ratio.empty:
                pick = ratio.iloc[0]
        if pick is not None:
            txt = format_T_row(pick)
            if txt:
                labels[comp] = txt
    return labels

def load_targets_csv(path):
    df = pd.read_csv(path)
    if not set(["x","y"]).issubset({c.lower() for c in df.columns}):
        xs = df.iloc[:,0].astype(int).tolist()
        ys = df.iloc[:,1].astype(int).tolist()
    else:
        xcol = [c for c in df.columns if c.lower()=="x"][0]
        ycol = [c for c in df.columns if c.lower()=="y"][0]
        xs = df[xcol].astype(int).tolist()
        ys = df[ycol].astype(int).tolist()
    return list(zip(xs, ys))



def joint_residual_1g(theta, packs):
    """
    1-Gaussian（不含基线）的联合残差
    theta = [v0, sig, A_0, A_1, ..., A_{m-1}]  # 长度 2 + m
    """
    v0, sig = float(theta[0]), abs(float(theta[1]))
    res_all = []
    for j, pk in enumerate(packs):
        A = float(theta[2 + j])  # 每条谱只有一个振幅
        v = pk['v']; y = pk['y']; yerr = pk.get('yerr', None)
        model = A * np.exp(-((v - v0)**2) / (2 * sig**2))
        r = y - model
        if yerr is not None:
            r = r / np.where(yerr > 0, yerr, 1.0)
        res_all.append(r)
    return np.concatenate(res_all)


def _build_bounds_for_global(packs, v0_guess, v0_window=2.0, allow_absorption=False):
    
    m = len(packs)
    # v0：以初猜 ± v0_window
    v0_lo = float(v0_guess - v0_window)
    v0_hi = float(v0_guess + v0_window)

    # sig：>0；上界給每條視窗寬度的最大值（保守一點再乘個 1.2）
    spans = [float(pk['v'].max() - pk['v'].min()) for pk in packs]
    sig_lo = 0.01
    sig_hi = max(spans) * 1.2 if len(spans) else 5.0

    bounds = [(v0_lo, v0_hi), (sig_lo, sig_hi)]

    # 為了讓 DE 有限界：每條線各自用資料百分位數估一組合理 A、C 範圍
    for pk in packs:
        y = np.asarray(pk['y'], float)
        if not np.isfinite(y).any():
            # 萬一資料都是 NaN，給一個小範圍避免崩
            A_lo, A_hi = (0.0 if not allow_absorption else -1.0), 1.0
            C_lo, C_hi = -1.0, 1.0
        else:
            med = float(np.nanmedian(y))
            p1, p99 = np.nanpercentile(y, [1, 99])
            amp_scale = max(1e-12, (p99 - med))
            # A：發射線預設不允許負值；若要吸收線，改 allow_absorption=True
            A_lo = (0.0 if not allow_absorption else -3.0 * amp_scale)
            A_hi = 3.0 * amp_scale
            # C：小基線常數，給個保守區間
            C_lo = p1 - abs(med) * 2.0
            C_hi = p99 + abs(med) * 2.0

        bounds.append((float(A_lo), float(A_hi)))
        bounds.append((float(C_lo), float(C_hi)))
    return bounds

def _joint_sse(theta, packs):
    r = joint_residual(theta, packs)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1e30
    return float(np.sum(r*r))

def fit_joint_gaussians(packs, 
                        v0_guess1=7.0, sig_guess1=0.3, 
                        v0_guess2=8.0, sig_guess2=0.6,
                        amp_guesses=None, use_bounds=True):

    m = len(packs)
    if amp_guesses is None:
        amp_guesses = []
        for pk in packs:
            yy = pk['y']
            amp_guesses.append(float(np.nanmax(yy) - np.nanmedian(yy)))

    # 參數：前4個是兩個高斯的中心/寬度，後面每條光譜只有 A1, A2
    theta0 = [v0_guess1, sig_guess1, v0_guess2, sig_guess2]
    for Aj in amp_guesses:
        theta0 += [max(Aj, 0.0), max(Aj/2, 0.0)]  # 只放 A1, A2

    if use_bounds:
        span = max(np.ptp(pk['v']) for pk in packs)  # NumPy 2.0 用 np.ptp
        lb = [v0_guess1-2, 0.01, v0_guess2-2, 0.01] \
             + sum(([0.0, 0.0] for _ in range(m)), [])
        ub = [v0_guess1+2, span,  v0_guess2+2, span] \
             + sum(([np.inf, np.inf] for _ in range(m)), [])
    else:
        lb = -np.inf*np.ones_like(theta0, float)
        ub =  np.inf*np.ones_like(theta0, float)

    res = least_squares(joint_residual, x0=np.array(theta0, float),
                        bounds=(np.array(lb), np.array(ub)), args=(packs,),
                        max_nfev=20000)

    out = {'success': res.success, 'cost': res.cost}
    if res.success:
        theta = res.x
        out['v0_1'], out['sig1'], out['v0_2'], out['sig2'] = theta[:4]
        out['A1'], out['A2'] = [], []
        for j in range(m):
            out['A1'].append(theta[4 + 2*j])
            out['A2'].append(theta[4 + 2*j + 1])
        out['FWHM1'] = 2.35482 * out['sig1']
        out['FWHM2'] = 2.35482 * out['sig2']
    return out


def fit_joint_gaussians_global(
    packs,
    v0_guess=8.0,
    sig_guess=0.6,         
    allow_absorption=False,
    use_polish=True,
    v0_window=2.0,
    de_maxiter=200,
    de_popsize=15,
    de_seed=None,
):
    

    m = len(packs)
    if m == 0:
        return {'success': False, 'cost': np.inf}

    # 1) 全域：DE
    bounds = _build_bounds_for_global(packs, v0_guess, v0_window, allow_absorption)
    result_de = differential_evolution(
        func=_joint_sse,
        bounds=bounds,
        args=(packs,),
        strategy="best1bin",
        maxiter=de_maxiter,
        popsize=de_popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=de_seed,
        polish=False,      
        updating="deferred"
    )

    theta_de = result_de.x

    
    if use_polish:
        
        if not allow_absorption:
            lb = [v0_guess - v0_window, 0.01] + sum(([0.0, -np.inf] for _ in range(m)), [])
        else:
            lb = [v0_guess - v0_window, 0.01] + sum(([-np.inf, -np.inf] for _ in range(m)), [])
        
        spans = [float(pk['v'].max() - pk['v'].min()) for pk in packs]
        sig_hi = max(spans) * 1.2 if len(spans) else 5.0
        ub = [v0_guess + v0_window, sig_hi] + sum(([np.inf, np.inf] for _ in range(m)), [])

        res_ls = least_squares(
            joint_residual,
            x0=np.array(theta_de, float),
            bounds=(np.array(lb, float), np.array(ub, float)),
            args=(packs,),
            max_nfev=20000
        )
        success = res_ls.success
        theta = res_ls.x if success else theta_de
        cost = (res_ls.cost if success else _joint_sse(theta_de, packs))
    else:
        success = result_de.success
        theta = theta_de
        cost = result_de.fun

    out = {'success': bool(success), 'cost': float(cost)}
    if success:
        v0  = float(theta[0])
        sig = float(abs(theta[1]))
        A_list, C_list = [], []
        for j in range(m):
            A_list.append(float(theta[2 + 2*j]))
            C_list.append(float(theta[2 + 2*j + 1]))
        out.update({
            'v0': v0,
            'sig': sig,
            'A': A_list,
            'C': C_list,
            'FWHM': 2.35482 * sig
        })
    return out

def _chi2(y, y_model, yerr=None, eps=1e-300):
    mask = np.isfinite(y) & np.isfinite(y_model)
    if yerr is not None:
        mask &= np.isfinite(yerr)
    y = np.asarray(y)[mask]
    y_model = np.asarray(y_model)[mask]
    if yerr is not None:
        yerr = np.asarray(yerr)[mask]
        var = np.maximum(yerr**2, eps)
        return np.sum((y - y_model)**2 / var), y.size
    else:
        
        rss = np.sum((y - y_model)**2)
        return rss, y.size

def _aic(y, y_model, yerr=None, k_params=3, use_aicc=False, eps=1e-300):
    chi2, n = _chi2(y, y_model, yerr=yerr, eps=eps)
    aic = 2.0 * k_params + chi2
    if use_aicc and (n - k_params - 1) > 0:
        aic += (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
    return aic


def joint_gaussian_model(v, v0_1, sig1, A1, v0_2, sig2, A2):
    
    g1 = A1 * np.exp(-((v - v0_1)**2) / (2 * sig1**2))
    g2 = A2 * np.exp(-((v - v0_2)**2) / (2 * sig2**2))
    return g1 + g2


def joint_residual(theta, packs):
    """
    theta = [v0_1, sig1, v0_2, sig2,
             A1_0, A2_0, A1_1, A2_1, ..., A1_{m-1}, A2_{m-1}]
    """
    v0_1, sig1, v0_2, sig2 = theta[0], abs(theta[1]), theta[2], abs(theta[3])
    res_all = []
    for j, pk in enumerate(packs):
        A1 = theta[4 + 2*j]
        A2 = theta[4 + 2*j + 1]

        v = pk['v']; y = pk['y']
        yerr = pk.get('yerr', None)

        mod = (A1 * np.exp(-((v - v0_1)**2)/(2*sig1**2))
             + A2 * np.exp(-((v - v0_2)**2)/(2*sig2**2)))  # 沒有 +C

        r = y - mod
        if yerr is not None:
            r = r / np.where(yerr > 0, yerr, 1.0)
        res_all.append(r)
    return np.concatenate(res_all)




def Jy2Tbri(I, bmaj_arcsec, bmin_arcsec, fre_GHz):
    
    I   = np.asarray(I,   float)
    fre = np.asarray(fre_GHz, float)
    fac = 1.222e6 / (bmaj_arcsec * bmin_arcsec)
    return I * (fac / (fre**2))

def robust_m1(v, s, center=8.0, half_width=5.0):
    m = np.isfinite(v) & np.isfinite(s) & (v >= center-half_width) & (v <= center+half_width)
    if m.sum() == 0:
        return np.nan
    w = np.clip(s[m], 0, None)
    if np.all(w == 0):
        w = np.abs(s[m])
    return float(np.nansum(v[m]*w) / np.nansum(w))

def subtract_baseline_edges(y, frac=0.2):
    y = np.asarray(y, float)
    n = len(y); k = max(1, int(n*frac))
    base = float(np.nanmedian(np.r_[y[:k], y[-k:]]))
    return y - base, base

def estimate_m1_offset(targets, m1_map, cube, vels):
    diffs = []
    if m1_map is None:
        return 0.0
    for (x0, y0) in targets[:min(50, len(targets))]:
        m1 = local_value(m1_map, x0, y0, box=1)
        if not np.isfinite(m1): 
            continue
        spec = cube[y0, x0, :]
        m1_loc = robust_m1(vels, spec, center=8.0, half_width=5.0)
        if np.isfinite(m1_loc):
            diffs.append(m1_loc - m1)  
    return float(np.nanmedian(diffs)) if len(diffs) >= 3 else 0.0



def load_moment_map(path, expect_unit='km/s'):
    """讀 CASA/CARTA 做的 moment map。自動把 m/s 轉成 km/s。"""
    if path is None:
        return None
    hdu = fits.open(path)
    m = np.squeeze(hdu[0].data).astype(float)
    hdr = hdu[0].header
    hdu.close()
    bunit = (hdr.get('BUNIT') or hdr.get('CUNIT1') or '').lower()
    # 常見：BUNIT='m/s' 或 'km/s'；有時是 'm s-1' 之類
    if 'm/s' in bunit or ('m' in bunit and 's' in bunit and '-1' in bunit):
        m = m / 1000.0
    return m

def local_value(map2d, x, y, box=1):
    """在 (x,y) 周圍 (2*box+1)^2 取中位數，忽略 NaN。"""
    if map2d is None: 
        return np.nan
    H, W = map2d.shape
    xs = slice(max(0, x-box), min(W, x+box+1))
    ys = slice(max(0, y-box), min(H, y+box+1))
    sub = map2d[ys, xs]
    if np.isfinite(sub).any():
        return float(np.nanmedian(sub))
    return np.nan




def beam_in_pixels(header):
    # BMAJ/BMIN: synthesized beam FWHM in degrees
    bmaj_deg = header.get('BMAJ')   # deg
    bmin_deg = header.get('BMIN')   # deg
    cdelt1   = abs(header.get('CDELT1'))  # deg/pix
    cdelt2   = abs(header.get('CDELT2'))  # deg/pix
    # 等效圓形 FWHM（幾何平均），轉成像素
    fwhm_eq_x_pix = np.sqrt(bmaj_deg*bmin_deg) / cdelt1
    fwhm_eq_y_pix = np.sqrt(bmaj_deg*bmin_deg) / cdelt2
    # 取平均當作「一個 beam」的像素尺度
    return float(0.5*(fwhm_eq_x_pix + fwhm_eq_y_pix))


def select_peaks_with_minsep(score_map, r_pix, k_max=30, thr=None):
    """
    score_map: 2D numpy array (e.g., moment0 或 SNR_M0)
    r_pix: 至少間距（像素），建議用 beam_in_pixels(header)
    k_max: 最多取多少個點
    thr: 分數下限（如 SNR_M0 >= 5）
    回傳: [(x,y,score), ...]（x 是列中的 index? 下方會統一 y,x）
    """
    m = np.array(score_map, dtype=float)
    if thr is not None:
        m = np.where(m >= thr, m, -np.inf)

    picked = []
    mask = np.zeros_like(m, dtype=bool)

    # 建一個圓形模板，之後用來畫禁區
    R = int(np.ceil(r_pix))
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    disk = (xx*xx + yy*yy) <= (r_pix*r_pix)

    for _ in range(k_max):
        # 在未被禁區的地方找最大值
        cur = np.where(~mask, m, -np.inf)
        idx = np.unravel_index(np.nanargmax(cur), cur.shape)
        y0, x0 = int(idx[0]), int(idx[1])
        val = cur[y0, x0]
        if not np.isfinite(val) or (thr is not None and val < thr):
            break
        picked.append((x0, y0, float(val)))

        # 畫禁區
        ys = slice(max(0, y0-R), min(m.shape[0], y0+R+1))
        xs = slice(max(0, x0-R), min(m.shape[1], x0+R+1))
        submask = mask[ys, xs]
        subdisk = disk[
            (slice(R-(y0-ys.start), R+(ys.stop-1-y0)+1),
             slice(R-(x0-xs.start), R+(xs.stop-1-x0)+1))
        ]
        submask |= subdisk
        mask[ys, xs] = submask

    # 統一回傳 (x,y)
    print(picked)
    return picked


#  Gaussian noise‐histogram fit 
def Gaussian(x, mean, amplitude, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

def calculate_noise(Map, pl=False, pr=False, start=None, end=None, cell=None):
    """
    Estimate noise by fitting a Gaussian to the histogram of pixel values.
    Returns [mean, peak_amplitude, sigma].
    """
    Map = np.squeeze(np.array(Map))
    # downsample cube for speed
    if Map.ndim >= 3:
        if Map.shape[0] > 200:
            idx = np.linspace(0, Map.shape[0]-1, 8, dtype=int)
            Map = Map[idx]
        if Map.shape[1] > 500 or Map.shape[2] > 500:
            yi = np.linspace(0, Map.shape[1]-1, min(300,Map.shape[1]), dtype=int)
            xi = np.linspace(0, Map.shape[2]-1, min(300,Map.shape[2]), dtype=int)
            Map = Map[:, yi][:, :, xi]
    # default histogram range
    if start is None:
        mid  = np.nanmedian(Map)
        disp = np.nanstd(Map)
        start, end = mid-5*disp, mid+5*disp
        cell = (end-start)/300
    bins = int((end-start)/cell + 1)
    flat = Map.reshape(-1)
    flat = flat[~np.isnan(flat)]
    Hist, edges = np.histogram(flat, bins=bins, range=(start-cell/2, end+cell/2))
    xs = 0.5 * (edges[:-1] + edges[1:])
    p0 = [xs[np.argmax(Hist)], max(Hist), (end-start)*0.1]
    popt, _ = curve_fit(Gaussian, xs, Hist, p0=p0, maxfev=100000)
    if pl:
        xcont = np.linspace(start, end, 1000)
        plt.step(xs, Hist, where='mid', label='hist')
        plt.plot(xcont, Gaussian(xcont, *popt), 'r-', label='fit')
        plt.legend(); plt.show()
    if pr:
        print(f"Noise sigma = {abs(popt[2]):.4g}, mean = {popt[0]:.4g}")
        noise_sigma = abs(popt[2])
    return [popt[0], popt[1], abs(popt[2])]

def draw_simple_panel(ax, rax, packs, jout, v01, v02, s1, s2,
                      xmin=5.0, xmax=10.0,
                      data_color="0.35", fit_color="black",
                      T_labels=None, t_fontsize=13):
    """
    Simple版：灰色資料 + 黑色總fit曲線 + 橘/紫填色分量 + 殘差。
    額外：在主圖角落標 T（若提供 T_labels={1:'...',2:'...'}）。
    """
    comp1_fill = (1.0, 0.6, 0.2, 0.35)    # 橘
    comp2_fill = (0.55, 0.45, 0.95, 0.35) # 紫
    lw_data, lw_fit = 0.9, 1.6

    has_comp2 = np.isfinite(v02) and np.isfinite(s2) \
                and (np.nanmax(np.abs(jout['A2'])) > 1e-6)

    resid_max = 0.0
    for j, pk in enumerate(packs):
        vj = pk['v']; yj = pk['y']
        A1j = float(jout['A1'][j])
        A2j = float(jout['A2'][j]) if has_comp2 else 0.0

        g1 = A1j * np.exp(-((vj - v01)**2)/(2*s1**2))
        g2 = (A2j * np.exp(-((vj - v02)**2)/(2*s2**2))) if has_comp2 else 0.0
        ymod = g1 + g2

        ax.step(vj, yj, where='mid', lw=lw_data, color=data_color, alpha=0.9)
        if np.any(np.asarray(g1) != 0): 
            ax.fill_between(vj, 0, g1, step='mid', color=comp1_fill)
        if has_comp2 and np.any(np.asarray(g2) != 0): 
            ax.fill_between(vj, 0, g2, step='mid', color=comp2_fill)
        ax.step(vj, ymod, where='mid', lw=lw_fit, color=fit_color)

        resid = yj - ymod
        rax.step(vj, resid, where='mid', lw=0.9, color=data_color, alpha=0.95)
        resid_max = max(resid_max, float(np.nanmax(np.abs(resid))))

    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("T (K)")
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=9)

    if T_labels:
        y_base = 0.98     # 第一行的頂端位置（axes fraction）
        dy = 0.16         # 行距（依你的版面可再微調）
        x_left = 0.015

        if 1 in T_labels:
            ax.text(x_left, y_base, T_labels[1],
                    transform=ax.transAxes, ha="left", va="top",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc=(1.0, 0.6, 0.2, 0.85), ec="none"),
                    fontsize=t_fontsize)
            y_base -= dy

        if 2 in T_labels:
            ax.text(x_left, y_base, T_labels[2],
                    transform=ax.transAxes, ha="left", va="top",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc=(0.55, 0.45, 0.95, 0.85), ec="none"),
                    fontsize=t_fontsize)

    rax.axhline(0.0, ls='-', lw=0.8, color='0.5')
    if np.isfinite(resid_max) and resid_max > 0:
        rax.set_ylim(-1.1*resid_max, 1.1*resid_max)
    rax.set_ylabel("resid")
    rax.set_xlabel("Velocity (km/s)")
    rax.tick_params(axis="both", labelsize=9)


#  1D Gaussian spectrum fit 
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x-mean)**2)/(2*stddev**2))

def fallback_range(data, window=5):
    idx = np.nanargmax(data)
    return max(0, idx-window), min(len(data)-1, idx+window)

def auto_pick_channel_range(spectrum, sigma_cut=3, plot=False):
    x = np.arange(len(spectrum))
    peaks, props = find_peaks(spectrum,
                              height=np.nanmax(spectrum)*0.3,
                              prominence=np.nanmax(spectrum)*0.1,
                              distance=5)
    if peaks.size==0:
        return fallback_range(spectrum)
    peak = peaks[np.argmax(props['peak_heights'])]
    p0 = [spectrum[peak], peak, 5, np.nanmin(spectrum)]
    try:
        popt, _ = curve_fit(gaussian, x, spectrum, p0=p0, maxfev=10000)
        mu, sig = popt[1], abs(popt[2])
        lo = int(max(0, np.floor(mu - sigma_cut*sig)))
        hi = int(min(len(spectrum)-1, np.ceil(mu + sigma_cut*sig)))
        if plot:
            plt.figure()
            plt.plot(x, spectrum, 'k-')
            plt.plot(x, gaussian(x, *popt), 'r--')
            plt.axvspan(lo, hi, color='orange', alpha=0.3)
            plt.show()
        return lo, hi
    except:
        return fallback_range(spectrum)

def pick_window_around_vsys(vels, v_sys=8.0, half_width=3.0):
    # 先做升冪排序來用 searchsorted
    order = np.argsort(vels)           # 若原本已升冪，這就是 [0,1,2,...]
    vv = vels[order]                   # 升冪的速度軸

    lo_s = np.searchsorted(vv, v_sys - half_width, side='left')
    hi_s = np.searchsorted(vv, v_sys + half_width, side='right') - 1

    lo_s = np.clip(lo_s, 0, len(vv)-1)
    hi_s = np.clip(hi_s, 0, len(vv)-1)
    if lo_s > hi_s:
        lo_s, hi_s = hi_s, lo_s

    inds = order[lo_s:hi_s+1]
    lo0 = int(np.min(inds))
    hi0 = int(np.max(inds))
    if lo0 > hi0:
        lo0, hi0 = hi0, lo0
    return lo0, hi0


def load_cube_and_vels(path):
    hdul = fits.open(path)
    header = hdul[0].header
    raw = hdul[0].data
    hdul.close()

    # → 統一成 (y, x, chan)
    cube = np.transpose(np.squeeze(raw), (1, 2, 0))

    # 取得 REST 頻率
    rest_Hz = header.get('RESTFRQ') or header.get('RESTFREQ')
    if rest_Hz is None:
        raise ValueError(f"[{path}] RESTFRQ/RESTFREQ missing.")
    rest = rest_Hz * u.Hz

    # 頻譜座標
    spec_wcs = WCS(header).sub(['spectral'])
    nchan = cube.shape[2]
    pix = np.arange(nchan)
    world = np.asarray(spec_wcs.all_pix2world(pix, 0)).squeeze()

    ctype = (header.get('CTYPE3') or '').upper()
    cunit = (header.get('CUNIT3') or '').strip()
    unit  = u.Unit(cunit) if cunit else None

    if 'FREQ' in ctype:
        if unit is None: unit = u.Hz
        freq = (world * unit).to(u.Hz)
        vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value
        freq_GHz = freq.to(u.GHz).value
    elif any(k in ctype for k in ['VRAD','VELO','VOPT']):
        if unit is None: unit = u.m/u.s
        vels = (world * unit).to(u.km/u.s).value
        # 從速度反推頻率（radio 定義）
        v = vels * (u.km/u.s)
        freq = v.to(u.Hz, equivalencies=u.doppler_radio(rest))
        freq_GHz = freq.to(u.GHz).value
    else:
        # 後備：當作頻率
        freq = (world * (unit or u.Hz)).to(u.Hz)
        vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value
        freq_GHz = freq.to(u.GHz).value

    return cube, header, vels, freq_GHz


def fit_gaussian(x, y, yerr=None, plot=False, allow_absorption=False,
                 v0_guess=None, sig_guess=None, dv_bound=5.0):
    dx = max(1e-6, np.median(np.abs(np.diff(x))))

    amp0 = np.nanmax(y) if not allow_absorption else np.nanmax(np.abs(y))
    sig0_default = max(5*dx/2.355, dx)
    sig0 = sig_guess if (sig_guess is not None and np.isfinite(sig_guess) and sig_guess > 0) else sig0_default
    mu0  = v0_guess if (v0_guess is not None and np.isfinite(v0_guess)) else 8.0

    p0 = (amp0, mu0, sig0)
    amp_lo = -np.inf if allow_absorption else 0.0
    bounds = ([amp_lo, mu0 - dv_bound, dx/2],
              [np.inf, mu0 + dv_bound, (x.max()-x.min())/2])

    popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=bounds,
                           sigma=yerr, absolute_sigma=(yerr is not None), maxfev=20000)

    if plot:
        plt.figure(figsize=(6,4))
        if yerr is not None:
            plt.errorbar(x, y, yerr=yerr, fmt='.')
        else:
            plt.plot(x, y, '.')
        plt.plot(x, gaussian(x, *popt), '-')
        plt.show()
    return popt, pcov




#  Main pipeline 
def main():
    mol2formula = {
        "acetaldehyde":    r"$\mathrm{CH_3CHO}$",
        "methyl_formate":  r"$\mathrm{CH_3OCHO}$",
        "glycolaldehyde":  r"$\mathrm{CH_2OHCHO}$",
        "ethylene_glycol": r"$\mathrm{HOCH_2CH_2OH}$",
        "propanenitrile":  r"$\mathrm{C_2H_5CN}$",
    }

    # 自動掃描 subimage
    all_fits = sorted(glob.glob(os.path.join(SUBIMAGE_DIR, "*.fits")))
    infiles = {}
    for p in all_fits:
        base = os.path.basename(p).replace(".fits","")
        tail = base.rsplit("_", 1)[-1]
        if tail.replace(".", "").isdigit():   # 只收尾碼是數字的
            infiles[base] = p


    # 對應 M1（有就用，沒有就 None）
    m1_files = {}
    for base in infiles:
        p = os.path.join(M1_DIR, f"{base}_M1_kms.fits")
        m1_files[base] = p if os.path.exists(p) else None

    groups = {}
    for base in infiles:
        mol = base.rsplit("_", 1)[0]    # e.g. methyl_formate
        groups.setdefault(mol, []).append(base)
    for mol in groups:
        groups[mol] = sorted(groups[mol], key=lambda b: float(b.rsplit("_",1)[1]))

    mol_targets = {
        "acetaldehyde":    os.path.join(SELECTED_DIR, "acetaldehyde_2313698_M0_targets.csv"),
        "ethylene_glycol": os.path.join(SELECTED_DIR, "ethylene_glycol_2329874_M0_targets.csv"),
        "glycolaldehyde":  os.path.join(SELECTED_DIR, "glycolaldehyde_2178307_M0_targets.csv"),
        "methyl_formate":  os.path.join(SELECTED_DIR, "methyl_formate_2182979_M0_targets.csv"),
        "propanenitrile":  os.path.join(SELECTED_DIR, "propanenitrile_2206609_M0_targets.csv"),
    }

    # 固定視窗參數（維持你原本設定）
    half_width = 2.0
    XMIN, XMAX = 5.0, 10.0
    overview_items = []

    
    for mol, bases in groups.items():
        # 讀 targets（找不到就跳過）
        tpath = mol_targets.get(mol)
        if (tpath is None) or (not os.path.exists(tpath)):
            print(f"[{mol}] skip: no targets CSV")
            continue
        targets = load_targets_csv(tpath)

        # 建 banks（每條 transition 的 cube/vels/noise/m1）
        banks = {}
        for base in bases:   # ← 修正：以前寫 keys 會 NameError
            cube_jy, header, vels, freq_GHz = load_cube_and_vels(infiles[base])

            # 光束（deg → arcsec）
            bmaj_as = float(header['BMAJ']) * 3600.0
            bmin_as = float(header['BMIN']) * 3600.0

            # Jy/beam → K（用 per-channel 頻率，靠 numpy 廣播）
            cube_K = Jy2Tbri(cube_jy, bmaj_as, bmin_as, freq_GHz[None, None, :])

            # 噪聲（單位：K）
            _, _, noise_sigma = calculate_noise(cube_K, pl=False, pr=False)

            m1_locmap = load_moment_map(m1_files.get(base, None))

            banks[base] = dict(
                cube=cube_K, header=header, vels=vels, freq_GHz=freq_GHz,
                noise=noise_sigma, m1=m1_locmap
            ) 
        # Δv (M1 偏移)
        dv_m1 = {}
        for base in bases:
            bank = banks[base]
            dv = estimate_m1_offset(targets, bank['m1'], bank['cube'], bank['vels'])
            dv_m1[base] = float(dv)
            print(f"[{mol}::{base}] m1 offset Δv ≈ {dv_m1[base]:.2f} km/s (cube_m1 - moment1)")

        # —— 每個「分子」各自開一份 PDF 與結果表 —— 
        ncols, nrows = 2, 2
        slots = ncols * nrows
        pdf_path = os.path.join(OUT_DIR, f"fit_pages_{mol}_joint.pdf")
        pp = PdfPages(pdf_path)

        pdf_simple_path = os.path.join(OUT_DIR, f"fit_pages_{mol}_simple.pdf")
        pp_simple = PdfPages(pdf_simple_path)
        def new_page():
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9),
                                    constrained_layout=True, sharex=True)
            return fig, np.atleast_1d(axes).ravel()
        def new_page_simple():
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9),
                                    constrained_layout=True, sharex=True)
            return fig, np.atleast_1d(axes).ravel()
        fig, axes = new_page()
        fig_s, axes_s = new_page_simple()
        plot_idx = 0
        plot_idx_simple = 0
        results = []

        for (x0, y0) in targets:
            print(f"\n=== [{mol}] JOINT @ (x={x0}, y={y0}) ===")

            packs = []
            v_guess_candidates = []
            used_bases = []  


            for base in bases:
                bank = banks[base]
                spec_full = bank['cube'][y0, x0, :]
                vels_full = bank['vels']

                m1_here = local_value(bank['m1'], x0, y0, box=1)
                v_guess_list = []
                if np.isfinite(m1_here):
                    v_guess_list.append(m1_here + dv_m1.get(base, 0.0))
                vm1_self = robust_m1(vels_full, spec_full, center=8.0, half_width=5.0)
                if np.isfinite(vm1_self):
                    v_guess_list.append(vm1_self)

                v_guess = float(np.nanmedian(v_guess_list)) if v_guess_list else 8.0
                v_guess = float(np.clip(v_guess, XMIN, XMAX))
                v_guess_candidates.append(v_guess)

                lo, hi = pick_window_around_vsys(vels_full, v_sys=v_guess, half_width=half_width)
                v_fit = vels_full[lo:hi+1]
                y_fit = spec_full[lo:hi+1]

                mask = (v_fit >= XMIN) & (v_fit <= XMAX) & np.isfinite(y_fit)
                if mask.sum() < 4:
                    print("  [skip] window too narrow in 5–10 km/s")
                    continue
                v_fit = v_fit[mask]
                y_fit = y_fit[mask]

                y_detr, base_const = subtract_baseline_edges(y_fit, frac=0.25)
                yerr = np.full_like(v_fit, bank['noise']) if (bank['noise'] > 0) else None
                packs.append({'v': v_fit, 'y': y_detr, 'yerr': yerr})
                used_bases.append(base)

            if len(packs) == 0:
                print("  [skip] empty packs")
                continue

            v0_guess = float(np.nanmedian(v_guess_candidates)) if v_guess_candidates else 8.0
            sig_guess = 0.6

            # >>> BEGIN AIC BLOCK
            m = len(packs)
            span = max(np.ptp(pk['v']) for pk in packs)
            amp_guesses_1g = [max(np.nanmax(pk['y']) - np.nanmedian(pk['y']), 0.0) for pk in packs]

            theta0_1g = [v0_guess, 0.6] + [float(A) for A in amp_guesses_1g]  # v0, sig, A_j...

            lb_1g = [v0_guess - 2, 0.01] + [0.0]*m
            ub_1g = [v0_guess + 2, span] + [np.inf]*m

            res1 = least_squares(
                joint_residual_1g, x0=np.array(theta0_1g, float),
                bounds=(np.array(lb_1g, float), np.array(ub_1g, float)),
                args=(packs,), max_nfev=20000
            )
            theta1 = res1.x
            v01_1g, s1_1g = float(theta1[0]), float(abs(theta1[1]))

            # AIC1
            chi2_1 = 0.0
            for j, pk in enumerate(packs):
                v, y = pk['v'], pk['y']; yerr = pk.get('yerr', None)
                Aj = theta1[2 + j]
                ymod = Aj * np.exp(-((v - v01_1g)**2)/(2*s1_1g**2))
                c2, _ = _chi2(y, ymod, yerr=yerr)
                chi2_1 += c2
            k1 = 2 + m      # (v0, sig) + m 個振幅
            aic1 = 2*k1 + chi2_1

            
            jout = fit_joint_gaussians(
                packs,
                v0_guess1=v0_guess - 0.2, sig_guess1=0.4,
                v0_guess2=v0_guess + 0.2, sig_guess2=0.6,
                amp_guesses=None, use_bounds=True
            )

            if not jout.get('success', False):
                print("  [fail] joint 2G fit; fallback to 1G")
                jout = {
                    'v0_1': v01_1g, 'sig1': s1_1g,
                    'v0_2': np.nan, 'sig2': np.nan,
                    'A1': [float(theta1[2 + j]) for j in range(m)],
                    'A2': [0.0]*m
                }
                model_used = "1G"; aic2 = np.nan
            else:
                v01_2g, s1_2g = jout['v0_1'], jout['sig1']
                v02_2g, s2_2g = jout['v0_2'], jout['sig2']

                chi2_2 = 0.0
                for j, pk in enumerate(packs):
                    v, y = pk['v'], pk['y']; yerr = pk.get('yerr', None)
                    A1j = jout['A1'][j]
                    A2j = jout['A2'][j]
                    ymod = (A1j*np.exp(-((v - v01_2g)**2)/(2*s1_2g**2))
                        + A2j*np.exp(-((v - v02_2g)**2)/(2*s2_2g**2)))
                    c2, _ = _chi2(y, ymod, yerr=yerr)
                    chi2_2 += c2

                k2 = 4 + 2*m   # (v01,s1,v02,s2) + 每條光譜兩個振幅
                aic2 = 2*k2 + chi2_2

                if (not np.isfinite(aic2)) or (aic2 > aic1 - 6.0):
                    jout = {
                        'v0_1': v01_1g, 'sig1': s1_1g,
                        'v0_2': np.nan, 'sig2': np.nan,
                        'A1': [float(theta1[2 + j]) for j in range(m)],
                        'A2': [0.0]*m
                    }
                    model_used = "1G"
                else:
                    model_used = "2G"


            # 後續共用變數（供繪圖/寫表）
            v01, s1 = jout['v0_1'], jout['sig1']
            v02, s2 = jout.get('v0_2', np.nan), jout.get('sig2', np.nan)
            fwhm1 = 2.35482 * s1
            fwhm2 = (2.35482 * s2) if np.isfinite(s2) else np.nan

            # 寫入結果列（含 AIC 與使用模型）
            row = {
                "molecule": mol, "x": x0, "y": y0,
                "v0_1 (km/s)": v01, "σ1 (km/s)": s1, "FWHM1 (km/s)": fwhm1,
                "v0_2 (km/s)": v02, "σ2 (km/s)": s2, "FWHM2 (km/s)": fwhm2,
                "model": model_used, "AIC1": aic1, "AIC2": aic2,
            }
            for j, base in enumerate(used_bases):
                row[f"A1[{base}]"] = jout['A1'][j]
                row[f"A2[{base}]"] = jout['A2'][j]
            results.append(row)
            # >>> END AIC BLOCK



            if plot_idx == slots:
                fig.tight_layout(); pp.savefig(fig); plt.close(fig)
                fig, axes = new_page(); plot_idx = 0
            ax = axes[plot_idx]; plot_idx += 1

            colors = ["C0", "C1", "C2", "C3"]
            divider = make_axes_locatable(ax)
            rax = divider.append_axes("bottom", size="28%", pad=0.15, sharex=ax)

            resid_max = 0.0
            for j, pk in enumerate(packs):
                vj = pk['v']; yj = pk['y']
                A1j = jout['A1'][j]
                A2j = jout['A2'][j]

                g1 = A1j * np.exp(-((vj - v01)**2) / (2*s1**2))
                g2 = A2j * np.exp(-((vj - v02)**2) / (2*s2**2)) if (np.isfinite(v02) and np.isfinite(s2) and (A2j != 0.0)) else 0.0
                yj_model = g1 + g2



                col = colors[j % len(colors)]
                molname = mol2formula.get(mol, mol)
                short   = used_bases[j].rsplit("_",1)[1] if "_" in used_bases[j] else used_bases[j]

                ax.step(vj, yj, where='mid', lw=1.0, label=f"{molname} {short} data", color=col)
                ax.step(vj, yj_model, where='mid', lw=1.0, ls='--', label=f"{molname} {short} fit", color=col)

                resid = yj - yj_model
                rax.step(vj, resid, where='mid', lw=0.9, color=col, alpha=0.95, label=f"{used_bases[j]} resid")
                resid_max = max(resid_max, float(np.nanmax(np.abs(resid))))

            ax.set_xlim(XMIN, XMAX)
            ax.axvline(v01, ls='--', lw=0.8, color='0.3')
            if np.isfinite(v02):
                ax.axvline(v02, ls='--', lw=0.8, color='0.3')

            ax.set_title(f"{molname}  (x={x0}, y={y0})", fontsize=10)
            ax.set_ylabel("Intensity (K)")
            ax.set_xlabel("")
            ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

            txt = (
            f"v0₁={v01:.2f} km/s  σ₁={s1:.2f}  FWHM₁={fwhm1:.2f}\n"
            f"v0₂={v02:.2f} km/s  σ₂={s2:.2f}  FWHM₂={fwhm2:.2f}\n" +
            "\n".join([
                f"A₁[{mol2formula.get(mol, mol)} {used_bases[j].rsplit('_',1)[1]}]={jout['A1'][j]:.3g}, "
                f"A₂={jout['A2'][j]:.3g}"
                for j in range(len(packs))
            ])
        )



            ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.9))

            rax.axhline(0.0, ls='-', lw=0.8, color='0.5')
            if np.isfinite(resid_max) and resid_max > 0:
                rax.set_ylim(-1.1*resid_max, 1.1*resid_max)
            rax.set_ylabel("resid")
            rax.set_xlabel("Velocity (km/s)")
            rax.tick_params(axis='both', labelsize=8)
            if plot_idx_simple == slots:
                fig_s.tight_layout(); pp_simple.savefig(fig_s); plt.close(fig_s)
                fig_s, axes_s = new_page_simple(); plot_idx_simple = 0

            ax_s = axes_s[plot_idx_simple]; plot_idx_simple += 1
            divider_s = make_axes_locatable(ax_s)
            rax_s = divider_s.append_axes("bottom", size="28%", pad=0.15, sharex=ax_s)
            # 取這個點的溫度標籤
            T_labels = get_T_labels_for_point(mol, x0, y0)

            draw_simple_panel(ax_s, rax_s, packs, jout, v01, v02, s1, s2,
                  xmin=XMIN, xmax=XMAX,
                  T_labels=T_labels, t_fontsize=13)

           

            molname = mol2formula.get(mol, mol)
            ax_s.set_title(f"{molname}  (x={x0}, y={y0})", fontsize=10)
        # --- BEGIN per-point PNG ---
        # 若這個 target 有有效的 packs，為該點輸出一張 PNG，分面顯示每個 transition
            if len(packs) > 0:
                wcel = WCS(banks[used_bases[0]]['header']).celestial
                ra_deg, dec_deg = wcel.all_pix2world(x0, y0, 0)
                ra_str  = Angle(ra_deg, unit=u.deg).to_string(unit=u.hourangle, sep=':', precision=2, pad=True)
                dec_str = Angle(dec_deg, unit=u.deg).to_string(unit=u.deg,       sep=':', precision=2, pad=True, alwayssign=True)

                ntr = len(packs)
                pt_ncols = 2 if ntr >= 2 else 1
                pt_nrows = int(np.ceil(ntr / pt_ncols))

                fig_pt, axes_pt = plt.subplots(nrows=pt_nrows, ncols=pt_ncols,
                                            figsize=(4.6*pt_ncols, 3.8*pt_nrows),
                                            sharex=False)
                axes_pt = np.atleast_1d(axes_pt).ravel()
                fig_pt.subplots_adjust(hspace=0.4)

                # ---- pass 1: 預先計算所有子圖的 y/resid 範圍 ----
                global_ymin, global_ymax = np.inf, -np.inf
                global_resid_abs_max = 0.0
                precomp = []  # 存起來避免重算

                has_comp2 = np.isfinite(v02) and np.isfinite(s2) and (np.nanmax(np.abs(jout['A2'])) > 1e-6)

                for j, pk in enumerate(packs):
                    vj = pk['v']; yj = pk['y']
                    A1j = float(jout['A1'][j])
                    A2j = float(jout['A2'][j]) if has_comp2 else 0.0

                    g1 = A1j * np.exp(-((vj - v01)**2) / (2*s1**2))
                    g2 = (A2j * np.exp(-((vj - v02)**2) / (2*s2**2))) if has_comp2 else 0.0
                    ymod  = g1 + g2
                    resid = yj - ymod

                    # 更新全域極值（保留可能的微小負值）
                    local_min = np.nanmin([np.nanmin(yj), np.nanmin(g1), np.nanmin(g2), np.nanmin(ymod), 0.0])
                    local_max = np.nanmax([np.nanmax(yj), np.nanmax(g1), np.nanmax(g2), np.nanmax(ymod), 0.0])
                    if np.isfinite(local_min): global_ymin = min(global_ymin, local_min)
                    if np.isfinite(local_max): global_ymax = max(global_ymax, local_max)

                    if np.isfinite(resid).any():
                        global_resid_abs_max = max(global_resid_abs_max, float(np.nanmax(np.abs(resid))))

                    precomp.append((vj, yj, g1, g2, ymod, resid))

                # 給一點頭尾空間
                if not np.isfinite(global_ymin): global_ymin = 0.0
                if not np.isfinite(global_ymax): global_ymax = 1.0
                ypad = 0.05*(global_ymax - global_ymin) if global_ymax > global_ymin else 0.1
                yl = global_ymin - ypad
                yu = global_ymax + ypad

                rpad = 0.1*global_resid_abs_max if global_resid_abs_max > 0 else 1.0
                rl, ru = -global_resid_abs_max - rpad, global_resid_abs_max + rpad

                # ---- pass 2: 繪圖，所有子圖用相同的 ylim/resid ylim ----
                for j, (vj, yj, g1, g2, ymod, resid) in enumerate(precomp):
                    axp = axes_pt[j]
                    divider_p = make_axes_locatable(axp)
                    raxp = divider_p.append_axes("bottom", size="28%", pad=0.15, sharex=axp)

                    axp.step(vj, yj, where='mid', lw=1.0, color="0.35", alpha=0.9)
                    if np.any(np.asarray(g1) != 0):
                        axp.fill_between(vj, 0, g1, step='mid', color=(1.0, 0.6, 0.2, 0.35))
                    if np.any(np.asarray(g2) != 0):
                        axp.fill_between(vj, 0, g2, step='mid', color=(0.55, 0.45, 0.95, 0.35))
                    axp.step(vj, ymod, where='mid', lw=1.6, color="crimson")

                    axp.set_xlim(XMIN, XMAX)
                    axp.set_ylim(yl, yu)
                    axp.axvline(v01, ls='--', lw=0.8, color='0.35')
                    if has_comp2: axp.axvline(v02, ls='--', lw=0.8, color='0.35')
                    axp.set_ylabel("T (K)")

                    raxp.step(vj, resid, where='mid', lw=0.9, color="0.35", alpha=0.95)
                    raxp.axhline(0.0, ls='-', lw=0.8, color='0.5')
                    raxp.set_ylim(rl, ru)
                    raxp.set_ylabel("resid"); raxp.set_xlabel("Velocity (km/s)")
                    raxp.tick_params(axis='both', labelsize=8)

                short = used_bases[j].rsplit("_", 1)[1] if "_" in used_bases[j] else used_bases[j]
                axp.set_title(f"{molname} {short}", fontsize=10, pad=8)

                
                for k in range(ntr, len(axes_pt)):
                    fig_pt.delaxes(axes_pt[k])

                fig_pt.suptitle(f"{molname}   (x={x0}, y={y0})   RA={ra_str}  Dec={dec_str}", fontsize=12)
                out_pt_dir = os.path.join(OUT_DIR, "per_point_png", mol)
                os.makedirs(out_pt_dir, exist_ok=True)
                png_pt_path = os.path.join(out_pt_dir, f"{mol}_{x0}_{y0}.png")
                fig_pt.savefig(png_pt_path, dpi=300, bbox_inches="tight", transparent=True)
                plt.close(fig_pt)
                print(f"[{mol}] Saved per-point PNG: {png_pt_path}")
            # --- END per-point PNG ---
   


        if plot_idx > 0:
            for k in range(plot_idx, slots):
                fig.delaxes(axes[k])
            fig.tight_layout(); pp.savefig(fig); 
            

            plt.close(fig)
        if plot_idx_simple > 0:
            for k in range(plot_idx_simple, slots):
                fig_s.delaxes(axes_s[k])
            fig_s.tight_layout(); pp_simple.savefig(fig_s)
            plt.close(fig_s)
                    
            if results:
                try:
                    
                    base_for_map = None
                    tpath = mol_targets.get(mol)
                    if tpath and os.path.exists(tpath):
                        # e.g. "acetaldehyde_2313698_M0_targets.csv" -> "acetaldehyde_2313698"
                        base_for_map = os.path.basename(tpath).replace("_M0_targets.csv", "")
                    if not base_for_map or (base_for_map not in bases):
                        base_for_map = bases[0]

                    m0_fits = os.path.join(M0_DIR, f"{base_for_map}_M0.fits")
                    if not os.path.exists(m0_fits):
                        raise FileNotFoundError(f"M0 FITS not found: {m0_fits}")

                    parts = base_for_map.split("_")
                    molecule = "_".join(parts[:-1])
                    raw_freq = parts[-1]
                    freq_lbl = f"{raw_freq[:3]}.{raw_freq[3:]} GHz" if raw_freq.isdigit() else raw_freq

                    with fits.open(m0_fits) as hdul:
                        w = WCS(hdul[0].header)
                        H, W = hdul[0].data.squeeze().shape[-2:]

                    pix_1g = []
                    pix_2g = []
                    for row in results:
                        x, y = int(row["x"]), int(row["y"])
                        if 0 <= x < W and 0 <= y < H:
                            if row.get("model") == "1G":
                                pix_1g.append((x, y))
                            elif row.get("model") == "2G":
                                pix_2g.append((x, y))

                    def pix_to_world(pairs):
                        if not pairs:
                            return [], []
                        xy = np.array(pairs, dtype=float)
                        ra, dec = w.all_pix2world(xy[:,0], xy[:,1], 0)
                        return ra.tolist(), dec.tolist()

                    ra1, dec1 = pix_to_world(pix_1g)
                    ra2, dec2 = pix_to_world(pix_2g)

                    fig_map = plt.figure(figsize=(5, 4))
                    ff = aplpy.FITSFigure(m0_fits, figure=fig_map)
                    ff.tick_labels.set_font(size=14)   # 座標數字字體大小
                    ff.axis_labels.set_font(size=16)   # 軸標籤字體大小                    
                    ff.recenter(CENTER_RA, CENTER_DEC, FOV_DEG, FOV_DEG)
                    ff.show_colorscale(vmin=M0_VMIN, vmax=M0_VMAX,
                                    cmap="inferno", stretch="arcsinh")

                    # beam
                    ff.add_beam()
                    ff.beam.set_color("white")

                    # scalebar: 0.5"
                    ff.add_scalebar(0.5/3600.0)
                    ff.scalebar.set_label("0.5″ (150 AU)")
                    ff.scalebar.set_color("white")
                    ff.scalebar.set_linewidth(2)
                    ff.scalebar.set_font(size=10)

                    # colorbar
                    ff.add_colorbar()
                    ff.colorbar.set_axis_label_text("Jy/beam·km/s")

                    molname = mol2formula.get(mol, mol)
                    fig_map.suptitle(f"{molname}  {freq_lbl}", fontsize=18, color="black")


                    # VLA markers
                    ff.show_markers([VLA4A_RA, VLA4B_RA], [VLA4A_DEC, VLA4B_DEC],
                                    marker="x", s=50, edgecolor="cyan", facecolor="cyan", linewidth=2)
                    ff.add_label(VLA4A_RA, VLA4A_DEC + 0.00002, "VLA4A", color="cyan", size=10)
                    ff.add_label(VLA4B_RA, VLA4B_DEC + 0.00002, "VLA4B", color="cyan", size=10)

                    # 1G / 2G markers
                    if ra1:
                        ff.show_markers(
                            ra1, dec1,
                            marker="o", s=40,
                            facecolor="none", edgecolor="deepskyblue", linewidth=1.8
                        )

                    if ra2:
                        ff.show_markers(
                            ra2, dec2,
                            marker="*", s=65,
                            facecolor="lime", edgecolor="black", linewidth=0.9
                        )
                    legend_handles = [
                        mlines.Line2D([], [], color='deepskyblue', marker='o', markersize=8,
                                    linestyle='None', markerfacecolor='none',
                                    label=f'1G fit (N={len(ra1)})'),
                        mlines.Line2D([], [], color='lime', marker='*', markersize=12,
                                    linestyle='None', markeredgecolor='black',
                                    label=f'2G fit (N={len(ra2)})'),
                        mlines.Line2D([], [], color='cyan', marker='x', markersize=8,
                                    linestyle='None', label='VLA4A/B'),
                    ]

                    # 把 legend 加到 APLpy 的軸上，定位在圖內左上
                    leg = ff._ax1.legend(
                        handles=legend_handles,
                        loc='upper left',            # 左上角
                        bbox_to_anchor=(0.02, 0.98), # 稍微離邊界一點（軸座標）
                        borderaxespad=0.2,
                        frameon=True,
                        framealpha=0.85,
                        fontsize=11,
                        markerscale=1.0,
                        handlelength=1.2,
                    )
                    leg.get_frame().set_facecolor('white')
                    leg.get_frame().set_edgecolor('0.3')

                    
                    overview_items.append({
                        "mol": mol,
                        "m0_fits": m0_fits,
                        "ra1": ra1, "dec1": dec1,   # 1G（藍空心圈）
                        "ra2": ra2, "dec2": dec2,   # 2G（金色星星）
                        "title": f"{molname} ",
                    })
                    # 存進 PDF
                    fig_map.subplots_adjust(left=0.07, right=0.95, top=0.93, bottom=0.07)

                    pp.savefig(fig_map)
                    png_path = os.path.join(OUT_DIR, f"model_select_{mol}.png")
                    fig_map.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
                    print(f"[{mol}] Saved PNG: {png_path}")
                    plt.close(fig_map)
                    try:
                        ff.close()
                    except Exception:
                        pass

                    print(f"[{mol}] Appended APLpy model-selection M0 map to PDF: {os.path.basename(m0_fits)}")
                except Exception as e:
                    print(f"[{mol}] WARN: failed to add APLpy M0 map: {e}")
        # === 所有分子同一頁總覽（單頁 PDF）===
        if overview_items:
            ALL_PDF_PATH = os.path.join(OUT_DIR, "all_molecules_model_selection_onepage.pdf")

            n = len(overview_items)
            ncols = 3 if n >= 3 else n      # 最多 3 欄
            nrows = int(np.ceil(n / max(1, ncols)))

            bigfig = plt.figure(figsize=(6.2*ncols, 6.2*nrows))
            for idx, item in enumerate(overview_items, start=1):
                try:
                    ff = aplpy.FITSFigure(item["m0_fits"], figure=bigfig, subplot=(nrows, ncols, idx))
                    ff.recenter(CENTER_RA, CENTER_DEC, FOV_DEG, FOV_DEG)
                    ff.show_colorscale(vmin=M0_VMIN, vmax=M0_VMAX, cmap="inferno", stretch="arcsinh")

                    # Beam + Scalebar（總覽頁不加 colorbar，避免擁擠）
                    ff.add_beam(); ff.beam.set_color("white")
                    ff.add_scalebar(0.5/3600.0)
                    ff.scalebar.set_label("0.5″ (150 AU)")
                    ff.scalebar.set_color("white")
                    ff.scalebar.set_linewidth(1.5)
                    ff.scalebar.set_font(size=9)

                    # 子圖標題
                    ff.add_label(0.5, 1.02, item["title"], relative=True, ha="center", size=18, color="black")

                    # VLA 標記
                    ff.show_markers([VLA4A_RA, VLA4B_RA], [VLA4A_DEC, VLA4B_DEC],
                                    marker="x", s=45, edgecolor="cyan", facecolor="cyan", linewidth=1.8)

                    # 1G（藍空心圈）
                    if item["ra1"]:
                        ff.show_markers(item["ra1"], item["dec1"], marker="o", s=38,
                                        facecolor="none", edgecolor="deepskyblue", linewidth=1.6)
                    # 2G（金色星星）
                    if item["ra2"]:
                        ff.show_markers(item["ra2"], item["dec2"], marker="*", s=65,
                                        facecolor="gold", edgecolor="black", linewidth=0.9)

                    try: ff.close()
                    except Exception: pass
                except Exception as e:
                    print(f"[overview] WARN: draw {item.get('mol','?')} failed: {e}")

            bigfig.tight_layout()
            bigfig.savefig(ALL_PDF_PATH)
            ALL_PNG_PATH = os.path.join(OUT_DIR, "all_molecules_model_selection_onepage.png")
            bigfig.savefig(ALL_PNG_PATH, dpi=300, bbox_inches="tight", transparent=True)
            print(f"[ALL] Saved one-page combined PNG: {ALL_PNG_PATH}")
            plt.close(bigfig)
            print(f"[ALL] Saved one-page combined PDF: {ALL_PDF_PATH}")
        else:
            print("[ALL] No overview items to plot.")
                

        pp.close()
        pp_simple.close()
        print(f"[{mol}] Saved multi-page PDF: {pdf_path}")
        print(f"[{mol}] Saved multi-page PDF (simple):   {pdf_simple_path}")

        if results:
            df = pd.DataFrame(results).sort_values(by=["y","x"]).reset_index(drop=True)
            csv_path = os.path.join(OUT_DIR, f"fit_results_{mol}_joint.csv")
            df.to_csv(csv_path, index=False)
            print(f"[{mol}] Saved results to {csv_path}")
        else:
            print(f"[{mol}] No successful fits.")


if __name__=='__main__':
    main()
