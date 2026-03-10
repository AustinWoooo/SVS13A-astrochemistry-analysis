#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
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

SELECTED_DIR = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/select_target"
SUBIMAGE_DIR = "/almabeegfs/scratch/ssp202525/data/subimage"
M1_DIR      = "/almabeegfs/scratch/ssp202525/moment_fig/mom1"
OUT_DIR     = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/2D"
TOP_K = 20
FIT_HALF_WIDTH   = 2.0   
PLOT_HALF_WIDTH  = 5.0  


os.makedirs(OUT_DIR, exist_ok=True)

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

def Jy2Tbri(I, bmaj_arcsec, bmin_arcsec, fre_GHz):
    
    I   = np.asarray(I,   float)
    fre = np.asarray(fre_GHz, float)
    fac = 1.222e6 / (bmaj_arcsec * bmin_arcsec)
    return I * (fac / (fre**2))



def gaussian2(x, a1, m1, s1, a2, m2, s2):
    return (a1*np.exp(-((x-m1)**2)/(2*s1**2))
          + a2*np.exp(-((x-m2)**2)/(2*s2**2)))

def _bic(y, y_model, yerr=None, k_params=3, eps=1e-300):
    
    
    mask = np.isfinite(y) & np.isfinite(y_model)
    if yerr is not None:
        mask &= np.isfinite(yerr)
    y = np.asarray(y)[mask]
    y_model = np.asarray(y_model)[mask]
    if yerr is not None:
        yerr = np.asarray(yerr)[mask]

    n = y.size
    if n == 0:
        return np.nan  

    resid = y - y_model

    if yerr is None:
        rss = np.sum(resid**2)
        sigma2_hat = max(rss / n, eps)  
        
        ll = -0.5 * (n * np.log(2*np.pi*sigma2_hat) + rss / sigma2_hat)
        k_total = k_params + 1  
    else:
        var = np.maximum(yerr**2, eps)
        ll = -0.5 * np.sum((resid**2)/var + np.log(2*np.pi*var))
        k_total = k_params  

    return -2.0 * ll + k_total * np.log(n)

def _fit_one_peak(x, y, yerr=None, v0_hint=None, sig_hint=None,
                  dv_bound=2.0, allow_absorption=False):
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = max(1e-6, np.median(np.abs(np.diff(x))))
    amp0 = (np.nanmax(np.abs(y)) if allow_absorption else np.nanmax(y))
    mu0  = (v0_hint if (v0_hint is not None and np.isfinite(v0_hint)) else x[np.nanargmax(y)])
    sig0 = (sig_hint if (sig_hint is not None and np.isfinite(sig_hint) and sig_hint>0)
            else max(5*dx/2.355, dx))
    amp_lo = -np.inf if allow_absorption else 0.0

    lo = (max(x.min(), mu0 - dv_bound) if v0_hint is not None else x.min())
    hi = (min(x.max(), mu0 + dv_bound) if v0_hint is not None else x.max())

    bounds = ([amp_lo, lo, dx/2], [np.inf, hi, max(dx, 0.5*(hi-lo))])
    p0 = (amp0, mu0, sig0)

    popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=bounds,
                           sigma=yerr, absolute_sigma=(yerr is not None), maxfev=40000)
    return popt, pcov

def fit_two_stage_gaussian(x, y, yerr=None, v0_hint=None, sig_hint=None,
                           dv_bound=2.0, allow_absorption=False,
                           min_delta_mu_frac=1.0,
                           delta_aic_thresh=6.0,   
                           use_aicc=True,          
                           plot=False):
    
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit

    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        m &= np.isfinite(yerr)
    x, y = x[m], y[m]
    yerr = (yerr[m] if yerr is not None else None)

    p1, C1 = _fit_one_peak(x, y, yerr=yerr, v0_hint=v0_hint, sig_hint=sig_hint,
                           dv_bound=dv_bound, allow_absorption=allow_absorption)
    a1, m1, s1 = p1
    y1 = gaussian(x, *p1)
    aic1 = _aic(y, y1, yerr=yerr, k_params=3, use_aicc=use_aicc)

    resid = y - y1
    kmax = np.nanargmax(np.abs(resid))
    sign = 1.0 if resid[kmax] >= 0 else -1.0
    r_use = sign * resid  
    peaks, props = find_peaks(r_use,
                              height=np.nanmax(r_use)*0.3 if np.nanmax(r_use)>0 else None,
                              prominence=np.nanmax(r_use)*0.15 if np.nanmax(r_use)>0 else None,
                              distance=max(2, int(0.5*np.ceil(np.ptp(x)/np.median(np.diff(x))))))

    if peaks.size == 0:
        return {"model":"1g", "params":p1, "cov":C1,
                "extras":{"aic1": aic1, "aic2": np.nan, "second_found":False}}

    j = peaks[np.argmax(props['peak_heights'])]
    m2_init = x[j]
    a2_init = resid[j]  
    dx = max(1e-6, np.median(np.abs(np.diff(x))))
    s2_init = max(s1, 5*dx/2.355)

    
    if abs(m2_init - m1) < min_delta_mu_frac*dx:
        return {"model":"1g", "params":p1, "cov":C1,
                "extras":{"aic1": aic1, "aic2": np.nan, "second_found":False,
                      "reason":"delta_mu_too_small"}}

    
    p0_2 = (a1, m1, s1, a2_init, m2_init, s2_init)

    
    lo = max(x.min(), (v0_hint - dv_bound)) if v0_hint is not None else x.min()
    hi = min(x.max(), (v0_hint + dv_bound)) if v0_hint is not None else x.max()

    
    dx = max(1e-6, np.median(np.abs(np.diff(x))))
    sig_lo = max(dx/2, 0.05)  # 

    sig_hi = max(4.0, 0.75*(x.max() - x.min()), 5.0*dx)

    s1 = np.clip(s1, sig_lo, sig_hi)
    s2_init = np.clip(s2_init, sig_lo, sig_hi)

    
    peak_abs = float(np.nanmax(np.abs(y))) if np.isfinite(np.nanmax(np.abs(y))) else 1.0
    amp_hi = 5.0 * peak_abs

    
    if (not allow_absorption) and (a2_init < 0):
        a2_init = 0.1 * peak_abs  

   
    m1 = float(np.clip(m1,      lo + 1e-9, hi - 1e-9))
    s1 = float(np.clip(s1,  sig_lo * 1.01, sig_hi * 0.99))
    m2_init = float(np.clip(m2_init, lo + 1e-9, hi - 1e-9))
    s2_init = float(np.clip(s2_init, sig_lo * 1.01, sig_hi * 0.99))

    
    amp_lo1 = 0.0
    amp_lo2 = (-np.inf if allow_absorption else 0.0)

    bounds_lo = [amp_lo1, lo, sig_lo, amp_lo2, lo, sig_lo]
    bounds_hi = [amp_hi,   hi, sig_hi, amp_hi,  hi, sig_hi]

    p0_2 = (a1, m1, s1, a2_init, m2_init, s2_init)

    p2, C2 = curve_fit(gaussian2, x, y, p0=p0_2, bounds=(bounds_lo, bounds_hi),
                       sigma=yerr, absolute_sigma=(yerr is not None), maxfev=60000)

    a1_, m1_, s1_, a2_, m2_, s2_ = p2
    if m1_ > m2_:
        p2 = np.array([a2_, m2_, s2_, a1_, m1_, s1_])

    too_close = abs(p2[1] - p2[4]) < 0.5 * max(p2[2], p2[5], dx)
    opposite_sign = (p2[0] * p2[3] < 0)
    amp_blow = (abs(p2[0]) > 3 * peak_abs) or (abs(p2[3]) > 3 * peak_abs)
    if (too_close and opposite_sign) or amp_blow:
        return {"model": "1g", "params": p1, "cov": C1,
                "extras": {"aic1": aic1, "aic2": np.nan,
                       "second_found": True, "reason": "degenerate_2g"}}
    a1_, m1_, s1_, a2_, m2_, s2_ = p2
    if m1_ > m2_:
        p2 = np.array([a2_, m2_, s2_, a1_, m1_, s1_])

    y2 = gaussian2(x, *p2)
    aic2 = _aic(y, y2, yerr=yerr, k_params=6, use_aicc=use_aicc)

    if aic2 <= aic1 - delta_aic_thresh:
        # 接受 2G
        if plot:
            ...
        return {"model":"2g", "params":p2, "cov":C2,
                "extras":{"aic2":aic2, "aic1":aic1, "second_found":True}}
    else:
        # 保留 1G
        return {"model":"1g", "params":p1, "cov":C1,
                "extras":{"aic2":aic2, "aic1":aic1, "second_found":True, "rejected_by_aic":True}}

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
    if 'm/s' in bunit or ('m' in bunit and 's' in bunit and '-1' in bunit):
        m = m / 1000.0
    return m

def local_value(map2d, x, y, box=1):
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
    
    bmaj_deg = header.get('BMAJ')   # deg
    bmin_deg = header.get('BMIN')   # deg
    cdelt1   = abs(header.get('CDELT1'))  # deg/pix
    cdelt2   = abs(header.get('CDELT2'))  # deg/pix
    # 等效圓形 FWHM（幾何平均），轉成像素
    fwhm_eq_x_pix = np.sqrt(bmaj_deg*bmin_deg) / cdelt1
    fwhm_eq_y_pix = np.sqrt(bmaj_deg*bmin_deg) / cdelt2
    # 取平均當作「一個 beam」的像素尺度
    return float(0.5*(fwhm_eq_x_pix + fwhm_eq_y_pix))



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



def pick_window_around_vsys(vels, v_sys=8.0, half_width=3.0):
    # 先做升冪排序來用 searchsorted
    order = np.argsort(vels)           
    vv = vels[order]                   

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




def main():
    csv_list = sorted(glob.glob(os.path.join(SELECTED_DIR, "*_M0_targets.csv")))
    if not csv_list:
        print(f"No target CSV found in {SELECTED_DIR}")
        return

    for csv_path in csv_list:
        base = os.path.basename(csv_path).replace("_M0_targets.csv", "")
        infile  = os.path.join(SUBIMAGE_DIR, f"{base}.fits")
        m1_path = os.path.join(M1_DIR,      f"{base}_M1_kms.fits")

        print(f"\n=== Processing {base} ===")
        if not os.path.exists(infile):
            print(f"[skip] cube not found: {infile}")
            continue

        hdul = fits.open(infile)
        header = hdul[0].header
        raw = hdul[0].data
        hdul.close()

        # 統一 cube 形狀為 (y, x, chan)
        cube = np.transpose(np.squeeze(raw), (1, 2, 0))
        print("Cube shape:", raw.shape, "→", cube.shape)

        nchan = cube.shape[2]

        # 取得 REST 頻率
        rest_Hz = header.get('RESTFRQ') or header.get('RESTFREQ')
        if rest_Hz is None:
            raise ValueError("RESTFRQ/RESTFREQ missing in header.")
        rest = rest_Hz * u.Hz

        # 頻譜座標（WCS 第三軸）
        spec_wcs = WCS(header).sub(['spectral'])
        pix = np.arange(nchan)
        world = np.asarray(spec_wcs.all_pix2world(pix, 0)).squeeze()

        ctype = (header.get('CTYPE3') or '').upper()
        cunit = (header.get('CUNIT3') or '').strip()
        unit = u.Unit(cunit) if cunit else None

        # 構建 vels 與 freq（Hz）
        if 'FREQ' in ctype:
            if unit is None: unit = u.Hz
            freq = (world * unit).to(u.Hz)
            vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value
        elif any(k in ctype for k in ['VRAD', 'VELO', 'VOPT']):
            if unit is None: unit = u.m/u.s
            vels = (world * unit).to(u.km/u.s).value
            # 從速度反推頻率（radio 定義）
            vq = vels * (u.km/u.s)
            freq = vq.to(u.Hz, equivalencies=u.doppler_radio(rest))
        else:
            # 當作頻率處理
            freq = (world * (unit or u.Hz)).to(u.Hz)
            vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value

        freq_GHz = freq.to(u.GHz).value

        # --- Jy/beam → 亮溫 K ---
        bmaj = header.get('BMAJ'); bmin = header.get('BMIN')
        if not (np.isfinite(bmaj) and np.isfinite(bmin) and bmaj > 0 and bmin > 0):
            raise ValueError("BMAJ/BMIN 缺失或為 0，無法轉亮溫。")
        bmaj_as = float(bmaj) * 3600.0
        bmin_as = float(bmin) * 3600.0

        cube_K = Jy2Tbri(cube, bmaj_as, bmin_as, freq_GHz[None, None, :])

        # 用 K 單位估噪（σ_K）
        _, _, noise_sigma = calculate_noise(cube_K, pl=False, pr=True)
        yerr_full = np.full(nchan, noise_sigma)

        print(f"[spec] CTYPE3={ctype}, CUNIT3={cunit}, "
              f"v=({np.nanmin(vels):.2f},{np.nanmax(vels):.2f}) km/s, "
              f"σ_noise≈{noise_sigma:.4g} K")

        #  targets CSV 
        df_t = pd.read_csv(csv_path).sort_values('score', ascending=False).head(TOP_K)
        targets = [(int(r.x), int(r.y)) for r in df_t.itertuples(index=False)]

        #  M1 map
        m1_map = load_moment_map(m1_path) if os.path.exists(m1_path) else None
        dv_m1 = estimate_m1_offset(targets, m1_map, cube, vels)
        print(f"[m1] estimated offset Δv ≈ {dv_m1:.2f} km/s (cube_m1 - moment1)")

        
        results = []
        ncols, nrows = 2, 2
        slots = ncols * nrows
        pdf_path = os.path.join(OUT_DIR, f"fit_pages_{base}.pdf")
        pp = PdfPages(pdf_path)

        def new_page():
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8), sharex=True)
            return fig, np.atleast_1d(axes).ravel()

        fig, axes = new_page()
        plot_idx = 0
        snr_threshold = 0.0

        for x0, y0 in targets:
            print(f"\n--- Pixel (x={x0}, y={y0}) ---")
            spec = cube_K[y0, x0, :]   # 單位：K


            m1_here = local_value(m1_map, x0, y0, box=1) if m1_map is not None else np.nan
            if np.isfinite(m1_here):
                v_guess_here = m1_here + dv_m1
            else:
                v_guess_here = robust_m1(vels, spec, center=8.0, half_width=5.0)
            if not np.isfinite(v_guess_here):
                v_guess_here = 8.0

            lo, hi = pick_window_around_vsys(vels, v_sys=v_guess_here, half_width=FIT_HALF_WIDTH)
            x_fit = vels[lo:hi+1]
            y_fit = spec[lo:hi+1]

            mask = np.isfinite(y_fit)
            if mask.sum() < 4:
                print("  [skip] too few finite points")
                continue

            x_fit_masked = x_fit[mask]
            y_fit_masked = y_fit[mask]
            yerr_masked  = yerr_full[lo:hi+1][mask]

            res = fit_two_stage_gaussian(
                x_fit_masked, y_fit_masked, yerr=yerr_masked,
                v0_hint=v_guess_here, dv_bound=2.0,
                allow_absorption=False,         
                min_delta_mu_frac=1.0,          
                delta_aic_thresh=10.0,            
                plot=False
            )

            if res["model"] == "1g":
                a, m, s = res["params"]
                fwhm = 2.35482 * s
                snr  = (a / noise_sigma) if noise_sigma > 0 else np.nan

                
                results.append({
                    "x": x0, "y": y0, "model": "1g",
                    "amp": a, "v0 (km/s)": m, "σ (km/s)": s,
                    "FWHM (km/s)": fwhm,
                    "SNR_peak": snr,
                    "AIC(1G)": res["extras"].get("aic1", np.nan),
                    "AIC(2G)": res["extras"].get("aic2", np.nan)
                })
                print(f"[1G] Pixel ({x0},{y0}) amp={a:.3f}, v0={m:.2f}, σ={s:.2f}, FWHM={fwhm:.2f}, SNR={snr:.1f}")

                lo_p, hi_p = pick_window_around_vsys(vels, v_sys=m, half_width=PLOT_HALF_WIDTH)
                x_plot = vels[lo_p:hi_p+1]
                y_plot = spec[lo_p:hi_p+1]

                mask_plot = np.isfinite(x_plot) & np.isfinite(y_plot)
                ord_ = np.argsort(x_plot[mask_plot])
                xs   = x_plot[mask_plot][ord_]
                ys   = y_plot[mask_plot][ord_]

                y_model = gaussian(xs, a, m, s)

                ax = axes[plot_idx]; plot_idx += 1
                ax.step(xs, ys, where='mid', linewidth=1.0, label='data')
                ax.step(xs, y_model, where='mid', linestyle='--', linewidth=1.0, label='1G fit')
                ax.axvline(m, linestyle='--', linewidth=0.8)

                txt = (f"(x={x0}, y={y0})\n"
       f"amp={a:.3f} K, v0={m:.2f} km/s\n"
       f"σ={s:.2f} km/s, FWHM={fwhm:.2f} km/s\n"
       f"SNR={snr:.1f}")

                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        va='top', ha='left', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.8))

                ax.set_xlim(3, 12)
                ax.set_xticks([5, 8, 10])
                _cur_idx = plot_idx - 1
                is_bottom_row = (_cur_idx >= ncols * (nrows - 1))
                ax.tick_params(axis='x', which='both', labelbottom=is_bottom_row)
                ax.set_xlabel("Velocity (km/s)" if is_bottom_row else "")
                ax.set_title(f"(x={x0}, y={y0})", fontsize=10)
                ax.set_ylabel("Intensity (Jy/beam)")

                if plot_idx == slots:
                    fig.tight_layout(); pp.savefig(fig); plt.close(fig)
                    fig, axes = new_page(); plot_idx = 0

            else:
                a1, m1, s1, a2, m2, s2 = res["params"]
                fwhm1 = 2.35482 * s1
                fwhm2 = 2.35482 * s2
                snr1  = (a1 / noise_sigma) if noise_sigma > 0 else np.nan
                snr2  = (a2 / noise_sigma) if noise_sigma > 0 else np.nan

                results.append({
                    "x": x0, "y": y0, "model": "2g",
                    "amp1": a1, "v01 (km/s)": m1, "σ1 (km/s)": s1, "FWHM1 (km/s)": fwhm1, "SNR1": snr1,
                    "amp2": a2, "v02 (km/s)": m2, "σ2 (km/s)": s2, "FWHM2 (km/s)": fwhm2, "SNR2": snr2,
                    "AIC(1G)": res["extras"]["aic1"], "AIC(2G)": res["extras"]["aic2"]
                })
                print(f"[2G] Pixel ({x0},{y0}) "
                    f"comp1: amp={a1:.3f}, v0={m1:.2f}, σ={s1:.2f}, FWHM={fwhm1:.2f}, SNR={snr1:.1f} ; "
                    f"comp2: amp={a2:.3f}, v0={m2:.2f}, σ={s2:.2f}, FWHM={fwhm2:.2f}, SNR={snr2:.1f}")

                v_center = 0.5*(m1+m2)
                half = max(PLOT_HALF_WIDTH, 0.5*abs(m2-m1) + PLOT_HALF_WIDTH*0.2)
                lo_p, hi_p = pick_window_around_vsys(vels, v_sys=v_center, half_width=half)
                x_plot = vels[lo_p:hi_p+1]
                y_plot = spec[lo_p:hi_p+1]

                mask_plot = np.isfinite(x_plot) & np.isfinite(y_plot)
                ord_ = np.argsort(x_plot[mask_plot])
                xs   = x_plot[mask_plot][ord_]
                ys   = y_plot[mask_plot][ord_]

                ysum = gaussian2(xs, a1, m1, s1, a2, m2, s2)
                y1   = a1*np.exp(-((xs-m1)**2)/(2*s1**2))
                y2   = a2*np.exp(-((xs-m2)**2)/(2*s2**2))

                ax = axes[plot_idx]; plot_idx += 1
                ax.step(xs, ys, where='mid', linewidth=1.0, label='data')
                ax.step(xs, ysum, where='mid', linewidth=1.0, linestyle='--', label='sum (2G)')
                ax.step(xs, y1, where='mid', linewidth=0.8, linestyle=':', label='comp1')
                ax.step(xs, y2, where='mid', linewidth=0.8, linestyle=':', label='comp2')
                ax.axvline(m1, linestyle='--', linewidth=0.8)
                ax.axvline(m2, linestyle='--', linewidth=0.8)

                txt = (f"(x={x0}, y={y0})\n"
       f"c1: amp={a1:.3f} K, v0={m1:.2f}, σ={s1:.2f}, FWHM={fwhm1:.2f}, SNR={snr1:.1f}\n"
       f"c2: amp={a2:.3f} K, v0={m2:.2f}, σ={s2:.2f}, FWHM={fwhm2:.2f}, SNR={snr2:.1f}")

                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        va='top', ha='left', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.8))

                ax.set_xlim(3, 12)
                ax.set_xticks([5, 8, 10])
                _cur_idx = plot_idx - 1
                is_bottom_row = (_cur_idx >= ncols * (nrows - 1))
                ax.tick_params(axis='x', which='both', labelbottom=is_bottom_row)
                ax.set_xlabel("Velocity (km/s)" if is_bottom_row else "")
                ax.set_title(f"(x={x0}, y={y0})", fontsize=10)
                ax.set_ylabel("T_B (K)")


                if plot_idx == slots:
                    fig.tight_layout(); pp.savefig(fig); plt.close(fig)
                    fig, axes = new_page(); plot_idx = 0

            



        if plot_idx > 0:
            for k in range(plot_idx, slots):
                fig.delaxes(axes[k])
            fig.tight_layout(); pp.savefig(fig); plt.close(fig)
        pp.close()
        print(f"\nSaved multi-page PDF: {pdf_path}")

        # 存 CSV
        if results:
            df = pd.DataFrame(results).sort_values(by=["y","x"]).reset_index(drop=True)
            out_csv = os.path.join(OUT_DIR, f"fit_results_{base}.csv")
            df.to_csv(out_csv, index=False)
            print(f"Saved results to {out_csv}")
        else:
            print("No successful fits.")


if __name__=='__main__':
    main()
