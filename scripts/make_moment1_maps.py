# /bin/python
import bettermoments as bm
import matplotlib
import pylab as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits as pyfits
import os

def Gaussian(x, mean, amplitude, stddev):
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

def calculate_noise(Map, pl=False, pr=False, start=None, end=None, cell=None):
    # 將Map壓縮為一個 numpy 數組
    Map = np.squeeze(np.array(Map))

    if Map.ndim>=3:
        # 切片處理
        if Map.shape[0] > 200:
            num_slices = 8
            indices = np.linspace(0, Map.shape[0] - 1, num_slices, dtype=int)
            Map = Map[indices, :, :]

        # 均勻取樣處理
        if Map.shape[1] > 500 or Map.shape[2] > 500:
            y_indices = np.linspace(0, Map.shape[1] - 1, min(300, Map.shape[1]), dtype=int)
            x_indices = np.linspace(0, Map.shape[2] - 1, min(300, Map.shape[2]), dtype=int)
            Map = Map[:, y_indices, :][:, :, x_indices]

        if Map.shape[1] > 500:
            x_indices = np.linspace(0, Map.shape[1] - 1, 300, dtype=int)
            y_indices = np.linspace(0, Map.shape[2] - 1, 300, dtype=int)
            Map = Map[:, y_indices, :]
            Map = Map[:, :, x_indices]

    # 設定默認的start, end, cell
    if start is None and end is None and cell is None:
        mid = np.nanmedian(Map)
        disp = np.nanstd(Map)
        start, end = mid - 5 * disp, mid + 5 * disp
        cell = (end - start) / 300

    # 計算 bins
    bins = int((end - start) / cell + 1)

    # 轉換Map為一維
    Map = Map.reshape(-1)
    Map = Map[~np.isnan(Map)]  # 去除NaN值

    # 計算直方圖
    Hist, edges = np.histogram(Map, bins=bins, range=(start - cell / 2, end + cell / 2))

    # 調整Hist[1]長度，使其與Hist[0]匹配
    Hist_x = edges[:-1]  # 使用邊界的左端點作為x軸

    # 設置初始擬合參數
    peak = max(Hist)
    p0 = [Hist_x[np.argmax(Hist)], peak, (end - start) * 0.1]

    # 高斯擬合
    popt, _ = curve_fit(Gaussian, Hist_x, Hist, p0=p0, maxfev=100000)

    # 繪製結果
    if pl:
        x = np.arange(start, end, cell / 5.)
        plt.step(Hist_x, Hist, where='mid')
        plt.plot(x, Gaussian(x, *popt))
        plt.xlim(start, end)
        plt.show()

    # 計算噪聲和均值
    Noi = popt[2]
    mean = popt[0]

    if pr:
        print("Noise = ", Noi, "Mean = ", mean)

    return [mean, popt[1], abs(Noi)]


img_name = '/data/ssp202525/moment_fig/subimage/propanenitrile_2329676.fits'

data, velax = bm.load_cube(img_name)
smoothed_data = bm.smooth_data(data=data, smooth=3, polyorder=0)

rms = calculate_noise(data)[2]
rms_smoothed = calculate_noise(data)[2]


mom0 = pyfits.getdata('propanenitrile_2329676_M0.fits')
noi_mom0 = calculate_noise(mom0)[2]
user_mask = mom0/noi_mom0   
plt.imshow(user_mask, origin='lower')
plt.contour(user_mask, levels=[3, 5, 7], colors='k')
#plt.show()
user_mask = user_mask[None,:,:]


threshold_mask = bm.get_threshold_mask(data=data,
                                       clip=5.0,
                                       smooth_threshold_mask=3.0)

channel_mask = bm.get_channel_mask(data=data,
                                   firstchannel=136,
                                   lastchannel=175)

mask = bm.get_combined_mask(user_mask=user_mask,
                            threshold_mask=threshold_mask,
                            channel_mask=channel_mask,
                            combine='and')

def mask_stats(name, m):
    print(f'{name:15s}  shape={m.shape}  True/1 像素 = {np.nansum(m>0):7d}')

mask_stats('user_mask',        user_mask)
mask_stats('threshold_mask',   threshold_mask)
mask_stats('channel_mask',     channel_mask)
mask_stats('AND combo',        mask)

masked_data = smoothed_data * mask

bm.available_collapse_methods()
moments = bm.collapse_first(velax=velax, data=masked_data, rms=rms)

bm.save_to_FITS(moments=moments, method='first', path=img_name)
os.system('mv '+'/'.join(img_name.split('/')[:-1])+'/*M?.fits .')

