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
from scipy.ndimage import maximum_filter
from astropy import units as u
from astropy.coordinates import SpectralCoord
import os
import glob
from scipy.optimize import differential_evolution
from mpl_toolkits.axes_grid1 import make_axes_locatable

OUT_DIR      = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/joint_fitting_1G"
SUBIMAGE_DIR = "/almabeegfs/scratch/ssp202525/data/subimage"
M1_DIR      = "/almabeegfs/scratch/ssp202525/moment_fig/mom1"
SELECTED_DIR = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/select_target"
os.makedirs(OUT_DIR, exist_ok=True)
MAKE_SIMPLE_PDF = True


def draw_simple_panel(ax, rax, packs, jout,
                      xmin=5.0, xmax=10.0,
                      data_color="0.35", fit_color="black"):
    """
    Simple 版（單 Gaussian、對應 joint_fitting_1G）：
      - 上 panel：灰色資料、橘色填滿的單一 Gaussian、黑色總合曲線
      - 下 panel：殘差
    不印數值、不畫虛線/legend。
    packs: list of {'v': ndarray, 'y': ndarray, 'yerr': ndarray or None}
    jout : dict from fit_joint_gaussians_global, keys: 'v0','sig','A', ...
    """
    comp_fill = (1.0, 0.6, 0.2, 0.35)   # 橘色半透明
    lw_data, lw_fit = 0.9, 1.6

    # 取單一 Gaussian 參數
    v0 = float(jout.get('v0', np.nan))
    s  = float(abs(jout.get('sig', np.nan)))

    resid_max = 0.0
    for j, pk in enumerate(packs):
        vj = np.asarray(pk['v'], float)
        yj = np.asarray(pk['y'], float)

        # 對應每個 transition 的振幅（保險起見做長度檢查）
        Aj = 0.0
        A_list = jout.get('A', [])
        if isinstance(A_list, (list, tuple, np.ndarray)) and j < len(A_list):
            Aj = float(A_list[j])

        # 單一 component 與總模型（1G → 就是同一條）
        g  = Aj * np.exp(-((vj - v0)**2)/(2*s**2))
        ymod = g

        # ---- 上 panel：資料 / 填色 / 總曲線 ----
        ax.step(vj, yj, where='mid', lw=lw_data, color=data_color, alpha=0.9)
        if np.any(np.asarray(g) != 0):
            ax.fill_between(vj, 0, g, step='mid', color=comp_fill)
        ax.step(vj, ymod, where='mid', lw=lw_fit, color=fit_color)

        # ---- 下 panel：殘差 ----
        resid = yj - ymod
        rax.step(vj, resid, where='mid', lw=0.9, color=data_color, alpha=0.95)
        if np.isfinite(resid).any():
            resid_max = max(resid_max, float(np.nanmax(np.abs(resid))))

    # ---- 外觀調整 ----
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("T (K)")
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=9)

    rax.axhline(0.0, ls='-', lw=0.8, color='0.5')
    if np.isfinite(resid_max) and resid_max > 0:
        rax.set_ylim(-1.1*resid_max, 1.1*resid_max)
    rax.set_ylabel("resid")
    rax.set_xlabel("Velocity (km/s)")
    rax.tick_params(axis="both", labelsize=9)

def Jy2Tbri(I, bmaj_arcsec, bmin_arcsec, fre_GHz):
    
    I   = np.asarray(I,   float)
    fre = np.asarray(fre_GHz, float)
    fac = 1.222e6 / (bmaj_arcsec * bmin_arcsec)
    return I * (fac / (fre**2))

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

def joint_gaussian_model(v, v0, sig, A):
    return A * np.exp(-((v - v0)**2) / (2*sig**2))

def joint_residual(theta, packs):
    """
    theta = [v0, sig, A_0, A_1, ..., A_{m-1}]
    packs = list of dicts, one per transition:
        {'v': array, 'y': array, 'yerr': array or None}
    回傳：把所有 transition 的殘差串接起來
    """
    v0, sig = theta[0], np.abs(theta[1])  
    m = (len(theta) - 2) // 2
    res_all = []
    for j in range(m):
        A = theta[2 + 2*j]
        v = packs[j]['v']; y = packs[j]['y']
        yerr = packs[j].get('yerr', None)
        mod = joint_gaussian_model(v, v0, sig, A)
        r = y - mod
        if yerr is not None:
            r = r / np.where(yerr>0, yerr, 1.0)
        res_all.append(r)
    return np.concatenate(res_all)

def fit_joint_gaussians(packs, v0_guess=8.0, sig_guess=0.6, amp_guesses=None, use_bounds=True):
    
    m = len(packs)
    if amp_guesses is None:
        amp_guesses = []
        for pk in packs:
            yy = pk['y']
            amp_guesses.append(float(np.nanmax(yy) - np.nanmedian(yy)))
    c0s = [0.0]*m

    theta0 = [float(v0_guess), float(sig_guess)]
    for Aj, Cj in zip(amp_guesses, c0s):
        theta0 += [float(max(Aj, 0.0)), float(Cj)]

    if use_bounds:
        
        lb = [v0_guess - 2.0, 0.01] + sum(([0.0, -np.inf] for _ in range(m)), [])
        ub = [v0_guess + 2.0, (packs[0]['v'].max() - packs[0]['v'].min())] + sum(([np.inf, np.inf] for _ in range(m)), [])
    else:
        lb = -np.inf*np.ones_like(theta0, dtype=float)
        ub =  np.inf*np.ones_like(theta0, dtype=float)

    res = least_squares(joint_residual, x0=np.array(theta0, float),
                        bounds=(np.array(lb), np.array(ub)), args=(packs,),
                        max_nfev=20000)

    out = {'success': res.success, 'cost': res.cost}
    if res.success:
        theta = res.x
        out['v0']  = float(theta[0])
        out['sig'] = float(abs(theta[1]))
        A_list, C_list = [], []
        for j in range(m):
            A_list.append(float(theta[2+2*j]))
            C_list.append(float(theta[2+2*j+1]))
        out['A'] = A_list
        out['C'] = C_list
        out['FWHM'] = 2.35482 * out['sig']
    return out


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
            diffs.append(m1_loc - m1)  # cube基準 - 外部M1
    return float(np.nanmedian(diffs)) if len(diffs) >= 3 else 0.0


def load_cube_and_vels(path):
    hdul = fits.open(path); header = hdul[0].header; raw = hdul[0].data; hdul.close()
    cube = np.transpose(np.squeeze(raw), (1, 2, 0))  # (y, x, chan)
    rest_Hz = header.get('RESTFRQ') or header.get('RESTFREQ')
    if rest_Hz is None:
        raise ValueError(f"[{path}] RESTFRQ/RESTFREQ missing.")
    rest = rest_Hz * u.Hz

    spec_wcs = WCS(header).sub(['spectral'])
    nchan = cube.shape[2]
    pix = np.arange(nchan)
    world = np.asarray(spec_wcs.all_pix2world(pix, 0)).squeeze()

    ctype = (header.get('CTYPE3') or '').upper()
    cunit = (header.get('CUNIT3') or '').strip()
    unit = u.Unit(cunit) if cunit else None

    if 'FREQ' in ctype:
        if unit is None: unit = u.Hz
        freq = (world * unit).to(u.Hz)
        vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value
    elif any(k in ctype for k in ['VRAD','VELO','VOPT']):
        if unit is None: unit = u.m/u.s
        vels = (world * unit).to(u.km/u.s).value
    else:
        freq = (world * (unit or u.Hz)).to(u.Hz)
        vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value

    return cube, header, vels

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

    # 依分子分組（去掉最後一段數字就是分子名）
    groups = {}
    for base in infiles:
        mol = base.rsplit("_", 1)[0]    # e.g. methyl_formate
        groups.setdefault(mol, []).append(base)
    for mol in groups:
        groups[mol] = sorted(groups[mol], key=lambda b: float(b.rsplit("_",1)[1]))

    # 每個分子用哪一份 targets.csv（用你現有檔名）
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

    
    for mol, bases in groups.items():
        # 讀 targets（找不到就跳過）
        tpath = mol_targets.get(mol)
        if (tpath is None) or (not os.path.exists(tpath)):
            print(f"[{mol}] skip: no targets CSV")
            continue
        targets = load_targets_csv(tpath)

        # 建 banks（每條 transition 的 cube/vels/noise/m1）
        banks = {}
        for base in bases:
            cube_jy, header, vels = load_cube_and_vels(infiles[base])

            # 依 header 的 CTYPE3/CUNIT3 推回每 channel 的頻率 freq (Hz) 與 GHz
            ctype = (header.get('CTYPE3') or '').upper()
            cunit = (header.get('CUNIT3') or '').strip()
            rest_Hz = header.get('RESTFRQ') or header.get('RESTFREQ')
            if rest_Hz is None:
                raise ValueError(f"[{infiles[base]}] RESTFRQ/RESTFREQ missing.")
            rest = (rest_Hz * u.Hz)

            spec_wcs = WCS(header).sub(['spectral'])
            nchan = cube_jy.shape[2]
            pix = np.arange(nchan)
            world = np.asarray(spec_wcs.all_pix2world(pix, 0)).squeeze()
            unit = u.Unit(cunit) if cunit else None

            if 'FREQ' in ctype:
                if unit is None: unit = u.Hz
                freq = (world * unit).to(u.Hz)
            elif any(k in ctype for k in ['VRAD','VELO','VOPT']):
                if unit is None: unit = u.m/u.s
                vq = (world * unit).to(u.km/u.s)
                # 用 radio 定義把速度轉回頻率
                freq = vq.to(u.Hz, equivalencies=u.doppler_radio(rest))
            else:
                freq = (world * (unit or u.Hz)).to(u.Hz)

            freq_GHz = freq.to(u.GHz).value  # ndarray, 長度 nchan

            # 讀 beam、把 Jy/beam → K
            bmaj_as = float(header['BMAJ']) * 3600.0
            bmin_as = float(header['BMIN']) * 3600.0
            cube_K = Jy2Tbri(cube_jy, bmaj_as, bmin_as, freq_GHz[None, None, :])  # (y,x,chan)

            # 用 K 單位估噪
            _, _, noise_sigma = calculate_noise(cube_K, pl=False, pr=False)

            m1_locmap = load_moment_map(m1_files.get(base, None))
            banks[base] = dict(
                cube=cube_K, header=header, vels=vels,
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
        def new_page():
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9),
                                    constrained_layout=True, sharex=True)
            return fig, np.atleast_1d(axes).ravel()
        fig, axes = new_page()
        plot_idx = 0
        results = []

        if MAKE_SIMPLE_PDF:
            pdf_simple = os.path.join(OUT_DIR, f"fit_pages_{mol}_joint_SIMPLE.pdf")
            pp_simple = PdfPages(pdf_simple)
            def new_page_simple():
                fig_s, axes_s = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9),
                                            constrained_layout=True, sharex=True)
                return fig_s, np.atleast_1d(axes_s).ravel()
            fig_s, axes_s = new_page_simple()
            idx_s = 0

        for (x0, y0) in targets:
            print(f"\n=== [{mol}] JOINT @ (x={x0}, y={y0}) ===")

            packs = []
            v_guess_candidates = []

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

            if len(packs) == 0:
                print("  [skip] empty packs")
                continue

            v0_guess = float(np.nanmedian(v_guess_candidates)) if v_guess_candidates else 8.0
            sig_guess = 0.6

            jout = fit_joint_gaussians_global(
                packs,
                v0_guess=v0_guess,
                sig_guess=sig_guess,
                allow_absorption=False,
                use_polish=True,
                v0_window=2.0,
                de_maxiter=200,
                de_popsize=15,
                de_seed=None
            )
            if not jout.get('success', False):
                print("  [fail] joint fit")
                continue

            v0 = jout['v0']; sigma = jout['sig']; fwhm = jout['FWHM']

            row = {"molecule": mol, "x": x0, "y": y0, "v0 (km/s)": v0, "σ (km/s)": sigma, "FWHM (km/s)": fwhm}
            for j, base in enumerate(bases[:len(packs)]):
                row[f"Amp[{base}]"] = jout['A'][j]
            results.append(row)

            if plot_idx == slots:
                fig.tight_layout(); pp.savefig(fig); plt.close(fig)
                fig, axes = new_page(); plot_idx = 0
            ax = axes[plot_idx]; plot_idx += 1

            colors = ["C0", "C1", "C2", "C3"]
            divider = make_axes_locatable(ax)
            rax = divider.append_axes("bottom", size="28%", pad=0.15, sharex=ax)

            resid_max = 0.0
            for j, pk in enumerate(packs):
                vj = pk['v']; yj = pk['y']; Aj = jout['A'][j]
                yj_model = joint_gaussian_model(vj, v0, sigma, Aj)
                col = colors[j % len(colors)]

                molname = mol2formula.get(mol, mol)   # 取化學式
                short   = bases[j].rsplit("_",1)[1]   # 頻率尾碼

                ax.step(vj, yj, where='mid', lw=1.0,
                        label=f"{molname} {short} data", color=col)
                ax.step(vj, yj_model, where='mid', lw=1.0, ls='--',
                        label=f"{molname} {short} fit", color=col)

                resid = yj - yj_model
                rax.step(vj, resid, where='mid', lw=0.9, color=col, alpha=0.95, label=f"{bases[j]} resid")
                resid_max = max(resid_max, float(np.nanmax(np.abs(resid))))

            ax.set_xlim(XMIN, XMAX)
            ax.axvline(v0, ls='--', lw=0.8, color='0.3')
            ax.set_title(f"{molname}  (x={x0}, y={y0})", fontsize=10)
            ax.set_ylabel("T_B (K)")

            ax.set_xlabel("")
            ax.legend(loc='upper right', fontsize=7, framealpha=0.85)

            txt = (
            f"v0={v0:.2f} km/s  σ={sigma:.2f}  FWHM={fwhm:.2f}\n"
            + "\n".join([
                f"A[{mol2formula.get(mol, mol)} {bases[j].rsplit('_',1)[1]}]={jout['A'][j]:.3g}"
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
            if MAKE_SIMPLE_PDF:
                if idx_s == slots:
                    fig_s.tight_layout(); pp_simple.savefig(fig_s); plt.close(fig_s)
                    fig_s, axes_s = new_page_simple(); idx_s = 0

                ax_s = axes_s[idx_s]; idx_s += 1
                div_s = make_axes_locatable(ax_s)
                rax_s = div_s.append_axes("bottom", size="28%", pad=0.15, sharex=ax_s)

                # 用 packs + jout 畫極簡 1G 面板
                draw_simple_panel(ax_s, rax_s, packs, jout, xmin=XMIN, xmax=XMAX)

                # 可選的標題（不會擠 legend，因為這裡我們沒放 legend）
                molname = mol2formula.get(mol, mol)
                ax_s.set_title(f"{molname} (x={x0}, y={y0})", fontsize=10)
            


    # === 收尾 ===
        if plot_idx > 0:
            for k in range(plot_idx, slots):
                fig.delaxes(axes[k])
            fig.tight_layout(); pp.savefig(fig); plt.close(fig)
        pp.close()
        print(f"[{mol}] Saved multi-page PDF: {pdf_path}")
        if MAKE_SIMPLE_PDF:
            if idx_s > 0:
                for k in range(idx_s, slots):
                    fig_s.delaxes(axes_s[k])
                fig_s.tight_layout(); pp_simple.savefig(fig_s); plt.close(fig_s)
            pp_simple.close()
            print(f"[{mol}] Saved SIMPLE PDF: {pdf_simple}")


        if results:
            df = pd.DataFrame(results).sort_values(by=["y","x"]).reset_index(drop=True)
            csv_path = os.path.join(OUT_DIR, f"fit_results_{mol}_joint.csv")
            df.to_csv(csv_path, index=False)
            print(f"[{mol}] Saved results to {csv_path}")
        else:
            print(f"[{mol}] No successful fits.")


if __name__=='__main__':
    main()
