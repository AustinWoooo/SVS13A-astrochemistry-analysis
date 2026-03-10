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
OUT_DIR     = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/1D"
TOP_K = 20
FIT_HALF_WIDTH   = 2.0   # 擬合用的半寬 (km/s)
PLOT_HALF_WIDTH  = 5.0   # 圖上顯示用的半寬 (km/s)


os.makedirs(OUT_DIR, exist_ok=True)



def gaussian_with_c(x, amp, mean, stddev, c):
    return c + amp * np.exp(-((x-mean)**2)/(2*stddev**2))

def fit_gaussian_with_c(x, y, yerr=None, plot=False,
                        v0_guess=None, sig_guess=None, dv_bound=5.0,
                        allow_absorption=False):
    dx = max(1e-6, np.median(np.abs(np.diff(x))))
    amp0 = np.nanmax(y) if not allow_absorption else np.nanmax(np.abs(y))
    sig0_default = max(5*dx/2.355, dx)
    sig0 = sig_guess if (sig_guess is not None and np.isfinite(sig_guess) and sig_guess > 0) else sig0_default
    mu0  = v0_guess if (v0_guess is not None and np.isfinite(v0_guess)) else 8.0
    c0   = float(np.nanmedian(y))

    amp_lo = -np.inf if allow_absorption else 0.0
    bounds = ([amp_lo, mu0 - dv_bound, dx/2,        -np.inf],
              [np.inf, mu0 + dv_bound, (x.max()-x.min())/2,  np.inf])

    p0 = (amp0, mu0, sig0, c0)

    popt, pcov = curve_fit(gaussian_with_c, x, y, p0=p0, bounds=bounds,
                           sigma=yerr, absolute_sigma=(yerr is not None), maxfev=20000)

    if plot:
        plt.figure(figsize=(6,4))
        if yerr is not None: plt.errorbar(x, y, yerr=yerr, fmt='.')
        else:                 plt.plot(x, y, '.')
        plt.plot(x, gaussian_with_c(x, *popt), '-')
        plt.show()
    return popt, pcov


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
            diffs.append(m1_loc - m1)  # cube基準 - 外部M1
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

        # ---------- 讀 cube ----------
        hdul = fits.open(infile)
        header = hdul[0].header
        raw = hdul[0].data
        hdul.close()

        # 修正軸順序：→ (y, x, chan)
        cube = np.transpose(np.squeeze(raw), (1,2,0))
        print("Cube shape:", raw.shape, "→", cube.shape)

        # ---------- 估計雜訊 ----------
        _, _, noise_sigma = calculate_noise(cube, pl=False, pr=True)
        nchan = cube.shape[2]
        yerr_full = np.full(nchan, noise_sigma)

        # ---------- 建立速度軸 ----------
        rest_Hz = header.get('RESTFRQ') or header.get('RESTFREQ')
        if rest_Hz is None:
            raise ValueError("RESTFRQ/RESTFREQ missing in header.")
        rest = rest_Hz * u.Hz

        spec_wcs = WCS(header).sub(['spectral'])
        pix = np.arange(nchan)
        world = np.asarray(spec_wcs.all_pix2world(pix, 0)).squeeze()

        ctype = (header.get('CTYPE3') or '').upper()
        cunit = (header.get('CUNIT3') or '').strip()
        unit = u.Unit(cunit) if cunit else None

        if 'FREQ' in ctype:
            if unit is None: unit = u.Hz
            freq = (world * unit).to(u.Hz)
            vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value
        elif any(k in ctype for k in ['VRAD', 'VELO', 'VOPT']):
            if unit is None: unit = u.m/u.s
            vels = (world * unit).to(u.km/u.s).value
        else:
            freq = (world * (unit or u.Hz)).to(u.Hz)
            vels = freq.to(u.km/u.s, equivalencies=u.doppler_radio(rest)).value

        print(f"[spec] CTYPE3={ctype}, CUNIT3={cunit}, v=({np.nanmin(vels):.2f},{np.nanmax(vels):.2f}) km/s")

        # ---------- 讀 targets CSV ----------
        df_t = pd.read_csv(csv_path).sort_values('score', ascending=False).head(TOP_K)
        targets = [(int(r.x), int(r.y)) for r in df_t.itertuples(index=False)]

        # ---------- 讀 M1（若有） ----------
        m1_map = load_moment_map(m1_path) if os.path.exists(m1_path) else None
        dv_m1 = estimate_m1_offset(targets, m1_map, cube, vels)
        print(f"[m1] estimated offset Δv ≈ {dv_m1:.2f} km/s (cube_m1 - moment1)")

        # ---------- 繪圖+輸出 ----------
        results = []
        ncols, nrows = 5, 4
        slots = ncols * nrows
        pdf_path = os.path.join(OUT_DIR, f"fit_pages_{base}.pdf")
        pp = PdfPages(pdf_path)

        def new_page():
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14,12), sharex=True)
            return fig, np.atleast_1d(axes).ravel()

        fig, axes = new_page()
        plot_idx = 0
        snr_threshold = 0.0

        for x0, y0 in targets:
            print(f"\n--- Pixel (x={x0}, y={y0}) ---")
            spec = cube[y0, x0, :]

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

            popt, pcov = fit_gaussian(
            x_fit_masked, y_fit_masked, yerr=yerr_masked,
            plot=False, v0_guess=v_guess_here, dv_bound=2.0
            )
            amp, v0, sigma = popt
            fwhm = 2.35482 * sigma
            snr_peak = (amp / noise_sigma) if noise_sigma > 0 else np.nan
            if snr_peak < snr_threshold:
                print(f"  [skip] SNR {snr_peak:.1f} < {snr_threshold}")
                continue

            results.append({
            "x": x0, "y": y0,
            "Amplitude": amp,
            "v0 (km/s)": v0,
            "σ (km/s)": sigma,
            "FWHM (km/s)": 2.35482 * sigma,
            "SNR_peak": (amp / noise_sigma) if noise_sigma > 0 else np.nan,
            })

            if plot_idx == slots:
                fig.tight_layout(); pp.savefig(fig); plt.close(fig)
                fig, axes = new_page(); plot_idx = 0

            # ----- 顯示窗口：以 v0 為中心畫 ±PLOT_HALF_WIDTH km/s -----
            lo_p, hi_p = pick_window_around_vsys(vels, v_sys=v0, half_width=PLOT_HALF_WIDTH)
            x_plot = vels[lo_p:hi_p+1]
            y_plot = spec[lo_p:hi_p+1]

            mask_plot = np.isfinite(y_plot) & np.isfinite(x_plot)
            if mask_plot.sum() < 2:
                # 若邊界太極端，退回用擬合視窗
                x_plot = x_fit_masked
                y_plot = y_fit_masked
                mask_plot = np.isfinite(y_plot)

            ax = axes[plot_idx]; plot_idx += 1
            ord_ = np.argsort(x_plot[mask_plot])
            xs = x_plot[mask_plot][ord_]
            ys = y_plot[mask_plot][ord_]

            # 三參數模型在顯示視窗上評估
            y_model = gaussian(xs, *popt)

            ax.step(xs, ys, where='mid', linewidth=1.0, label='data')
            ax.step(xs, y_model, where='mid', linewidth=1.0, linestyle='--', label='fit')
            ax.axvline(v0, linestyle='--', linewidth=0.8)

            
            ax.set_xlim(3, 12)            # 固定顯示範圍，可自行調整
            ax.set_xticks([5, 8, 10])     # 固定絕對刻度：5, 8, 10 km/s
            ax.tick_params(axis='x', which='both', bottom=True, top=False)  # 刻度線每張都有

            # 只在最下面一排顯示數字與標籤
            _cur_idx = plot_idx - 1  # ← 重點：因為上面已經 plot_idx += 1 了
            is_bottom_row = (_cur_idx >= ncols * (nrows - 1))
            ax.tick_params(axis='x', which='both', labelbottom=is_bottom_row)
            ax.set_xlabel("Velocity (km/s)" if is_bottom_row else "")

            # 其他標題/縱軸維持
            ax.set_title(f"(x={x0}, y={y0})", fontsize=10)
            ax.set_ylabel("Intensity (Jy/beam)")

            txt = (f"v0={v0:.2f} km/s\nσ={sigma:.2f} km/s\n"
                f"FWHM={2.35482*sigma:.2f} km/s\n"
                f"amp={amp:.3f} Jy/b\nS/N={(amp / noise_sigma) if noise_sigma>0 else np.nan:.1f}")
            ax.text(0.02, 0.98, txt, transform=ax.axes.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.8))



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
