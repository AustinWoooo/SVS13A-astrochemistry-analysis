#!/usr/bin/env python3
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch

# 參數
m0_fits = "/almabeegfs/scratch/ssp202525/moment_fig/mom0/third_carta_channels/glycolaldehyde_2178307_M0.fits"
x_pix, y_pix = 398, 402   # 輸入 pixel 座標
zoom_size = 30            # 放大範圍 (pixel)

# 開檔
hdu = fits.open(m0_fits)[0]
data = hdu.data.squeeze()
wcs = WCS(hdu.header).celestial

# arcsinh normalization
norm = ImageNormalize(data, stretch=AsinhStretch(), vmin=0, vmax=0.1)

# 畫圖
fig = plt.figure(figsize=(6,5))
ax = plt.subplot(projection=wcs)
im = ax.imshow(data, origin="lower", cmap="inferno", norm=norm)

# 標點
ax.plot(x_pix, y_pix, marker="o", color="cyan", ms=10, mfc="none", mew=2)
ax.text(x_pix+2, y_pix+2, f"({x_pix},{y_pix})", color="cyan", fontsize=10)

# 放大 (只顯示座標附近 zoom_size 範圍)
ax.set_xlim(x_pix - zoom_size, x_pix + zoom_size)
ax.set_ylim(y_pix - zoom_size, y_pix + zoom_size)

# 加 colorbar 與標籤
cb = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046)
cb.set_label("Jy/beam·km/s")
ax.set_xlabel("RA")
ax.set_ylabel("Dec")
plt.title("Acetaldehyde moment 0 (zoomed, arcsinh stretch)")

# 存檔
out_path = "/almabeegfs/scratch/ssp202525/acetaldehyde_m0_with_point_zoom_arcsinh.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved zoomed figure to {out_path}")
