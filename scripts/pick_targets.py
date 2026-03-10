#!/usr/bin/env python3
from PIL import Image
import math
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Ellipse
import glob

fits_list = [f for f in glob.glob("/data/ssp202525/moment_fig/mom0/third_carta_channels/*_M0.fits")
             if not f.endswith("_dM0.fits")]
#OUTPUT_CSV  = "propanenitrile_2206609_targets.csv"           # 輸出 CSV
#PLOT_PATH   = "propanenitrile_2206609_test.png"           # 預覽圖檔名；設為 None 則不畫
OUT_DIR = "/data/ssp202525/Gaussian_fitting/select_target"
os.makedirs(OUT_DIR, exist_ok=True)


# （可選）外部遮罩：非零/True 代表保留
MASK_FITS   = None                    # 例如 "pbmask.fits"；沒有就設 None


POLY_PIX = [
    (330, 430),  # 左上
    (430, 430),  # 右上
    (430, 350),  # 右下
    (330, 350),  # 左下
]

K_MAX        = 20       
BEAM_FACTOR  = 1.0      
THR_ABS      = None     
PERC         = 85.0    
NSIG = 5.0 


def beam_ellipse_in_pixels(header):
    
    bmaj = header.get('BMAJ', None)   # deg
    bmin = header.get('BMIN', None)   # deg
    bpa  = float(header.get('BPA', 0.0))  # deg, E of N
    cd1  = abs(header.get('CDELT1', np.nan))  # deg/pix
    cd2  = abs(header.get('CDELT2', np.nan))  # deg/pix
    if any(v is None for v in (bmaj, bmin)) or np.any(np.isnan([cd1, cd2])):
        return None
    major_pix = bmaj / cd1
    minor_pix = bmin / cd2
    angle = 90.0 - bpa  # E of N → matplotlib
    return float(major_pix), float(minor_pix), float(angle)

def beam_in_pixels(header):
    
    bmaj_deg = header.get('BMAJ', None)
    bmin_deg = header.get('BMIN', None)
    cdelt1   = abs(header.get('CDELT1', np.nan))
    cdelt2   = abs(header.get('CDELT2', np.nan))
    if any(v is None for v in (bmaj_deg, bmin_deg)) or np.any(np.isnan([cdelt1, cdelt2])):
        raise ValueError("Header 缺少 BMAJ/BMIN/CDELT1/CDELT2，無法換算 beam 像素尺寸。")
    fwhm_eq_pix_x = np.sqrt(bmaj_deg * bmin_deg) / cdelt1
    fwhm_eq_pix_y = np.sqrt(bmaj_deg * bmin_deg) / cdelt2
    return float(0.5 * (fwhm_eq_pix_x + fwhm_eq_pix_y))

def combine_pngs(png_dir, output_path, cols=3, bg_color=(255,255,255)):
    # 找出資料夾裡所有 _targets.png
    png_files = sorted([os.path.join(png_dir, f) 
                        for f in os.listdir(png_dir) 
                        if f.endswith("_targets.png")])

    if not png_files:
        print("沒有找到任何 _targets.png")
        return

    # 開啟所有圖片
    images = [Image.open(p) for p in png_files]

    # 單張尺寸（假設都一樣大）
    w, h = images[0].size

    # 計算網格行列數
    rows = math.ceil(len(images) / cols)

    # 建立背景大圖
    combined = Image.new("RGB", (cols * w, rows * h), color=bg_color)

    # 貼上圖片
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        combined.paste(img, (x, y))

    combined.save(output_path)
    print(f"[info] 已輸出總覽圖：{output_path}")

def select_peaks_with_minsep(score_map, r_pix, k_max=30, thr=None):
    
    m = np.array(score_map, dtype=float)
    m[~np.isfinite(m)] = -np.inf
    if thr is not None:
        m = np.where(m >= thr, m, -np.inf)

    picked = []
    mask = np.zeros_like(m, dtype=bool)

    R = int(np.ceil(r_pix))
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    disk = (xx*xx + yy*yy) <= (r_pix*r_pix)

    for _ in range(int(k_max)):
        cur = np.where(~mask, m, -np.inf)
        # 若整張都是 -inf，argmax 還是會回 0；需要檢查值
        flat_idx = np.nanargmax(cur)
        y0, x0 = np.unravel_index(flat_idx, cur.shape)
        val = cur[y0, x0]
        if not np.isfinite(val) or (thr is not None and val < thr):
            break

        picked.append((int(x0), int(y0), float(val)))

        ys = slice(max(0, y0-R), min(m.shape[0], y0+R+1))
        xs = slice(max(0, x0-R), min(m.shape[1], x0+R+1))
        submask = mask[ys, xs]

        dy0 = y0 - ys.start
        dx0 = x0 - xs.start
        subdisk = disk[
            (slice(R-dy0, R-dy0+submask.shape[0]),
             slice(R-dx0, R-dx0+submask.shape[1]))
        ]
        submask |= subdisk
        mask[ys, xs] = submask

    return picked

def polygon_mask_from_xy(shape, xy_list):
    """
    由 (x,y) 頂點建立多邊形遮罩；回傳 bool mask：多邊形內為 True
    shape: (ny, nx)
    xy_list: list[(x, y)]，x=col, y=row
    """
    from matplotlib.path import Path
    ny, nx = shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    pts = np.vstack([X.ravel(), Y.ravel()]).T  # (N,2) with (x,y)
    poly = Path(xy_list)
    inside = poly.contains_points(pts)
    return inside.reshape((ny, nx))

def wcs_polygon_to_pixel_xy(header, radec_list):
    """
    將 WCS 多邊形頂點(ra,dec) 轉為像素 (x,y)
    radec_list: list[(ra_str, dec_str)]
    """
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    w = WCS(header)
    xs, ys = [], []
    for ra_str, dec_str in radec_list:
        sc = SkyCoord(ra_str.strip(), dec_str.strip(), unit=(u.hourangle, u.deg))
        px, py = w.world_to_pixel(sc)  # x=col, y=row
        xs.append(float(px)); ys.append(float(py))
    return list(zip(xs, ys))

def main():
    # 讀 FITS
    with fits.open(INPUT_FITS) as hdul:
        header = hdul[0].header
        data = hdul[0].data

    # squeeze 成 2D
    arr = np.squeeze(np.array(data))
    if arr.ndim != 2:
        raise ValueError(f"輸入陣列是 {arr.shape}，不是 2D。請提供 2D 圖。")

    # 外部 mask（交集）
    if MASK_FITS is not None:
        with fits.open(MASK_FITS) as mhdul:
            mask_arr = np.squeeze(np.array(mhdul[0].data))
        if mask_arr.shape != arr.shape:
            raise ValueError("MASK_FITS 與輸入 FITS 的影像大小不一致。")
        arr = np.where(mask_arr, arr, np.nan)

    # ROI：像素多邊形 / WCS 多邊形（可同時給 → 交集）
    roi_mask = None
    if POLY_PIX and len(POLY_PIX) >= 3:
        poly_mask = polygon_mask_from_xy(arr.shape, POLY_PIX)
        roi_mask = poly_mask if roi_mask is None else (roi_mask & poly_mask)

    
    if roi_mask is not None:
        arr = np.where(roi_mask, arr, np.nan)

    # beam 像素與最小間距
    beam_pix = beam_in_pixels(header)
    r_pix = float(BEAM_FACTOR * beam_pix)

    # 決定門檻
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        raise ValueError("ROI 內沒有可用的有限數值。")

    if THR_ABS is not None:
        thr_val = float(THR_ABS)
    elif NSIG is not None:
        med = np.nanmedian(finite_vals)
        mad = np.nanmedian(np.abs(finite_vals - med))
        sigma = 1.4826 * mad
        thr_val = float(med + NSIG * sigma)
    else:
        thr_val = float(np.nanpercentile(finite_vals, float(PERC)))

    # 執行挑峰
    picked = select_peaks_with_minsep(arr, r_pix=r_pix, k_max=int(K_MAX), thr=thr_val)

    # 存 CSV
    df = pd.DataFrame(picked, columns=["x", "y", "score"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[info] beam_pix ≈ {beam_pix:.3f}, 使用 r_pix = {r_pix:.3f} 像素")
    print(f"[info] 門檻 = {thr_val:.6g}（{'絕對值' if THR_ABS is not None else f'{PERC}th 百分位'}）")
    print(f"[info] 挑到 {len(picked)} 點 → {os.path.abspath(OUTPUT_CSV)}")

    # 可選：畫 PNG 預覽
    if PLOT_PATH is not None:
        import matplotlib.pyplot as plt
        from astropy.visualization import ImageNormalize, AsinhStretch

        fig, ax = plt.subplots(figsize=(6,5))

        # ===== 顏色與拉伸設定 =====
        vmin, vmax = 0, 0.11
        cmap = 'inferno'
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())
        # ========================

        # 顯示影像
        im = ax.imshow(arr, origin="lower", aspect="equal", cmap=cmap, norm=norm)

        # 畫挑到的點
        if picked:
            xs = [p[0] for p in picked]
            ys = [p[1] for p in picked]
            ax.scatter(xs, ys, s=15, marker="+", c="cyan", linewidths=1)


        # 設定視窗範圍
        if roi_mask is not None and np.any(roi_mask):
            ys_idx, xs_idx = np.where(roi_mask)
            ax.set_xlim(xs_idx.min(), xs_idx.max())
            ax.set_ylim(ys_idx.min(), ys_idx.max())
        elif picked:
            xs_np = np.asarray(xs)
            ys_np = np.asarray(ys)
            ax.set_xlim(xs_np.min(), xs_np.max())
            ax.set_ylim(ys_np.min(), ys_np.max())

        # Beam
        be = beam_ellipse_in_pixels(header)
        if be is not None:
            major_pix, minor_pix, angle = be
            xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
            x0 = xlo + 0.05*(xhi - xlo)
            y0 = ylo + 0.05*(yhi - ylo)
            e = Ellipse((x0, y0), width=major_pix, height=minor_pix,
                        angle=angle, fill=False, lw=1.5, ec="w")
            ax.add_patch(e)

        # 加顏色條
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Integrated Intensity (Jy/beam km/s)")

        ax.set_title(f"k={len(picked)}, min sep ≈ {r_pix:.1f} px")
        ax.set_xlabel("x (pix)"); ax.set_ylabel("y (pix)")

        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=200, transparent=True)
        print(f"[info] 預覽圖已存：{os.path.abspath(PLOT_PATH)}")

if __name__ == "__main__":
    for f in fits_list:
        base = os.path.splitext(os.path.basename(f))[0]
        INPUT_FITS  = f
        OUTPUT_CSV  = os.path.join(OUT_DIR, f"{base}_targets.csv")
        PLOT_PATH   = os.path.join(OUT_DIR, f"{base}_targets.png")

        print(f"[batch] 處理 {base} …")
        try:
            main()
        except Exception as e:
            print(f"[batch] {base} 失敗：{e}")
    combine_pngs("/data/ssp202525/Gaussian_fitting/select_target",
             "/data/ssp202525/Gaussian_fitting/select_target/all_targets_overview.png",
             cols=3)