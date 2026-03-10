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
OUT_DIR     = "/almabeegfs/scratch/ssp202525/results/Gaussian_fitting/multi_G"
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
    if n <= 0:
        return np.inf
    if yerr is None:
        rss = chi2
        aic = n * np.log(max(rss / max(n, 1), eps)) + 2.0 * k_params
    else:
        aic = 2.0 * k_params + chi2
    if use_aicc and (n - k_params - 1) > 0:
        aic += (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
    return aic


def gaussian_n_params(x, *theta):
    
    return gaussian_n(x, np.asarray(theta, float))

def gaussian_n(x, theta):
    
    x = np.asarray(x, float)
    theta = np.asarray(theta, float)
    assert theta.size % 3 == 0
    y = np.zeros_like(x, dtype=float)
    for i in range(0, theta.size, 3):
        a, m, s = theta[i:i+3]
        y += a * np.exp(-0.5 * ((x - m) / s)**2)
    return y

def _k_params_for_gaussians(n_components, per_comp_params=3):
    
    return per_comp_params * n_components

def fit_ng_curve_fit(x, y, yerr, n, v0_hint=None, dv_bound=2.0, allow_absorption=False):
    from scipy.optimize import curve_fit

    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = max(1e-6, np.median(np.abs(np.diff(x))))
    lo = x.min() if v0_hint is None else max(x.min(), v0_hint - dv_bound)
    hi = x.max() if v0_hint is None else min(x.max(), v0_hint + dv_bound)
    sig_lo = max(dx/2, 0.05)
    sig_hi = max(4.0, 0.75*(x.max()-x.min()), 5.0*dx)

    peak_abs = float(np.nanmax(np.abs(y))) if np.isfinite(np.nanmax(np.abs(y))) else 1.0
    amp_hi = 5.0 * peak_abs
    amp_lo = (-np.inf if allow_absorption else 0.0)

    ms = np.linspace(lo + 0.2*(hi-lo), hi - 0.2*(hi-lo), n)
    ss = np.full(n, max(5*dx/2.355, dx))
    a0 = max(peak_abs, 1e-3)
    p0 = np.ravel(np.column_stack([np.full(n, a0), ms, ss]))

    bounds_lo = []; bounds_hi = []
    for _ in range(n):
        bounds_lo += [amp_lo, lo, sig_lo]
        bounds_hi += [amp_hi,  hi, sig_hi]

    popt, pcov = curve_fit(
    gaussian_n_params, x, y, p0=p0, bounds=(bounds_lo, bounds_hi),
    sigma=yerr, absolute_sigma=(yerr is not None), maxfev=100000
)

    popt = popt.reshape(n, 3)
    popt = popt[np.argsort(popt[:,1])]
    popt = popt.ravel()
    return popt, pcov

def select_components_by_aic(x, y, yerr, n_max=5, v0_hint=None,
                             aic_drop_per_new_comp=6.0,  
                             use_aicc=True, per_comp_params=3,
                             dv_bound=2.0, allow_absorption=False):
    
    best = None
    aic_prev = np.inf

    for n in range(1, n_max+1):
        popt, pcov = fit_ng_curve_fit(x, y, yerr, n, v0_hint=v0_hint,
                                      dv_bound=dv_bound, allow_absorption=allow_absorption)
        y_model = gaussian_n(x, popt)
        k_params = _k_params_for_gaussians(n, per_comp_params=per_comp_params)
        aic = _aic(y, y_model, yerr=yerr, k_params=k_params, use_aicc=use_aicc)

        if n == 1:
            best = (n, popt, aic, pcov)
            aic_prev = aic
        else:
            if (aic_prev - aic) >= aic_drop_per_new_comp:
                best = (n, popt, aic, pcov)
                aic_prev = aic
            else:
                break
    return best




def gaussian2(x, a1, m1, s1, a2, m2, s2):
    return (a1*np.exp(-((x-m1)**2)/(2*s1**2))
          + a2*np.exp(-((x-m2)**2)/(2*s2**2)))

def _bic(y, y_model, yerr=None, k_params=3, eps=1e-300):
    mask = np.isfinite(y) & np.isfinite(y_model) & (np.isfinite(yerr) if yerr is not None else True)
    y = np.asarray(y)[mask]; y_model = np.asarray(y_model)[mask]
    if y.size == 0:
        return np.nan
    resid = y - y_model
    n = y.size
    if yerr is None:
        rss = float(np.sum(resid**2))
        sigma2_hat = max(rss / n, eps)
        ll = -0.5 * (n * np.log(2*np.pi*sigma2_hat) + rss / sigma2_hat)
        k_total = k_params + 1  # 另估 σ^2
    else:
        var = np.maximum(np.asarray(yerr)[mask]**2, eps)
        ll = -0.5 * np.sum((resid**2)/var + np.log(2*np.pi*var))
        k_total = k_params
    return -2.0 * ll + k_total * np.log(n)






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




def pick_window_around_vsys(vels, v_sys=8.0, half_width=3.0):
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

        cube = np.transpose(np.squeeze(raw), (1,2,0))
        print("Cube shape:", raw.shape, "→", cube.shape)

        _, _, noise_sigma = calculate_noise(cube, pl=False, pr=True)
        nchan = cube.shape[2]
        yerr_full = np.full(nchan, noise_sigma)

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

            sel = select_components_by_aic(
                x_fit_masked, y_fit_masked, yerr_masked,
                n_max=3,                      
                v0_hint=v_guess_here,
                aic_drop_per_new_comp=6.0,    
                use_aicc=True,
                per_comp_params=3,            
                dv_bound=2.0,
                allow_absorption=False
            )
            if sel is None:
                print("  [skip] model selection failed")
                continue

            n_best, popt_best, aic_best, pcov_best = sel
            print(f"[AIC] n_best={n_best}, AIC(best)={aic_best:.2f}")

            
            mask_full = np.isfinite(vels) & np.isfinite(spec)
            if not np.any(mask_full):
                print("  [skip] full spectrum all NaN")
                continue

            ord_full  = np.argsort(vels[mask_full])
            x_full    = vels[mask_full][ord_full]
            y_full    = spec[mask_full][ord_full]

            x_model   = np.linspace(x_full.min(), x_full.max(), 800)
            y_sum_full = gaussian_n(x_model, popt_best)

            ax = axes[plot_idx]; plot_idx += 1

            ax.step(x_full, y_full, where='mid', linewidth=1.2, label='data')

            ax.plot(x_model, y_sum_full, '-', linewidth=1.6, label=f'sum ({n_best}G)')

            txt_lines = [f"(x={x0}, y={y0})", f"AIC={aic_best:.1f}"]   # ← 先初始化
            for i in range(n_best):
                a, m, s = popt_best[3*i:3*i+3]
                comp_full = a * np.exp(-0.5 * ((x_model - m) / s)**2)
                ax.plot(x_model, comp_full, ':', linewidth=1.2, label=f'c{i+1}')
                ax.axvline(m, linestyle='--', linewidth=0.9)  # ← 留一條就好
                fwhm = 2.35482 * s
                snr  = (a / noise_sigma) if noise_sigma > 0 else np.nan
                txt_lines.append(f"c{i+1}: amp={a:.3f} Jy/b, v0={m:.2f}, σ={s:.2f}, FWHM={fwhm:.2f}, SNR={snr:.1f}")

            
            ax.text(0.02, 0.98, "\n".join(txt_lines), transform=ax.transAxes,
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

            row = {"x": x0, "y": y0, "model": f"{n_best}g", "AIC(best)": aic_best}
            for i in range(n_best):
                a, m, s = popt_best[3*i:3*i+3]
                fwhm = 2.35482 * s
                snr  = (a / noise_sigma) if noise_sigma > 0 else np.nan
                row.update({
                    f"amp{i+1}": a, f"v0{i+1} (km/s)": m, f"σ{i+1} (km/s)": s,
                    f"FWHM{i+1} (km/s)": fwhm, f"SNR{i+1}": snr
                })
            results.append(row)

                

            



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
