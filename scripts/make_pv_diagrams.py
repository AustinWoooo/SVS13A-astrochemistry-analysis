#! bin/python
from numpy import *
from pylab import *
from astropy.io import fits as pyfits
from astropy import constants as con
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import defaultdict
import os

mol_dict = defaultdict(list)

#Jy2Tbri
def Jy2Tbri(I='', bmaj='', bmin='', fre=''):
    I = np.asarray(I)
    fre = np.asarray(fre)

    # 如果 fre 是 scalar，就轉成 shape (1,)
    if fre.ndim == 0:
        fre = fre[np.newaxis]

    # 進行 brightness temperature 計算
    T = 1.222e6 * I / (fre[:, np.newaxis, np.newaxis]**2 * bmaj * bmin)
    return T

#header
def read_wave(header,lineFre=''):
    names=list(header.keys())
    NAXIS3=header['NAXIS3']
    CRVAL3=header['CRVAL3']
    CDELT3=header['CDELT3']
    CRPIX3=header['CRPIX3']
    if 'RESTFREQ' in names:
        RESTFREQ=header['RESTFREQ']
    elif 'RESTFRQ' in names:
        RESTFREQ=header['RESTFRQ']
    else:
        print('Error: check rest frequ')
    chan=arange(1,NAXIS3+1,1.)

    if header['CTYPE3']=='FREQ':
        freq_li=(chan-CRPIX3)*CDELT3+CRVAL3
        vel_li=-(freq_li-RESTFREQ)/RESTFREQ*con.c.to('km/s').value
    if header['CTYPE3'].strip()=='VRAD':
        vel_li=(chan-CRPIX3)*CDELT3+CRVAL3
        freq_li=RESTFREQ*(1-vel_li/con.c.to('m/s').value)

    if lineFre!='':
        vel_li=-(freq_li-lineFre)/lineFre*con.c.to('km/s').value
    return freq_li,chan,vel_li

def read_wcs_header(header):
    NAXIS1=header['NAXIS1']
    CRVAL1=header['CRVAL1']
    CDELT1=header['CDELT1']
    CRPIX1=header['CRPIX1']

    NAXIS2=header['NAXIS2']
    CRVAL2=header['CRVAL2']
    CDELT2=header['CDELT2']
    CRPIX2=header['CRPIX2']
    return NAXIS1, CRVAL1, CDELT1,CRPIX1,NAXIS2,CRVAL2,CDELT2,CRPIX2


def read_wcs_coord(header):
    NAXIS1, CRVAL1, CDELT1,CRPIX1,NAXIS2,CRVAL2,CDELT2,CRPIX2=read_wcs_header(header)

    bins_ra=arange(0,NAXIS1,1.)+1
    CDELT1=CDELT1/(cos(deg2rad(CRVAL2)))
    RA=(bins_ra-CRPIX1)*CDELT1+CRVAL1

    bins_dec=arange(1,NAXIS2+1,1.)
    Dec=(bins_dec-CRPIX2)*CDELT2+CRVAL2

    extent=[RA[0]+abs(CDELT1/2),RA[-1]-abs(CDELT1/2),Dec[0]-abs(CDELT2/2),Dec[-1]+abs(CDELT2/2)]
    return extent

def read_pv_coord(header):
    NAXIS1, CRVAL1, CDELT1,CRPIX1,NAXIS2,CRVAL2,CDELT2,CRPIX2=read_wcs_header(header)

    bins_ra=arange(1,NAXIS1+1,1.)
    RA=(bins_ra-CRPIX1)*CDELT1+CRVAL1

    names=list(header.keys())
    if 'RESTFREQ' in names:
        RESTFREQ=header['RESTFREQ']
    elif 'RESTFRQ' in names:
        RESTFREQ=header['RESTFRQ']
    else:
        print('Error: check rest frequ')
    lineFre=RESTFREQ
    if header['CTYPE2'] == 'VRAD':  # make sure the velocity is in km/s
        bins_vel=arange(1,NAXIS2+1,1.)
        v = (bins_vel-CRPIX2)*CDELT2+CRVAL2
        VDELT = CDELT2
        v /=1000    # to km/s
        VDELT /=1000    #to km/s
    else:
        bins_freq=arange(1,NAXIS2+1,1.)
        freq=(bins_freq-CRPIX2)*CDELT2+CRVAL2
        v=(lineFre-freq)/lineFre*con.c.to('km/s').value
        VDELT=average(v[1:]-v[:-1])

    extent=[RA[0]+abs(CDELT1/2),RA[-1]-abs(CDELT1/2),v[0]-abs(VDELT/2),v[-1]+abs(VDELT/2)]
    return extent


def read_wcs_coord2(header):
    NAXIS1, CRVAL1, CDELT1,CRPIX1,NAXIS2,CRVAL2,CDELT2,CRPIX2=read_wcs_header(header)

    bins_ra=arange(0,NAXIS1,1.)
    CDELT1=CDELT1/(cos(deg2rad(CRVAL2)))
    RA=(bins_ra-CRPIX1)*CDELT1+CRVAL1

    bins_dec=arange(1,NAXIS2+1,1.)
    Dec=(bins_dec-CRPIX2)*CDELT2+CRVAL2

    return RA,Dec

def read_wcs_coord3(header,precise=1):
    NAXIS1, CRVAL1, CDELT1,CRPIX1,NAXIS2,CRVAL2,CDELT2,CRPIX2=read_wcs_header(header)

    bins_ra=arange(0,NAXIS1,precise)+1
    CDELT1=CDELT1/(cos(deg2rad(CRVAL2)))
    RA=(bins_ra-CRPIX1)*CDELT1+CRVAL1

    bins_dec=arange(1,NAXIS2+1,precise)
    Dec=(bins_dec-CRPIX2)*CDELT2+CRVAL2

    return RA,Dec


#Noise
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

fits_dir = '/almabeegfs/scratch/ssp202525/data/PV_diagram'

mol_formula = {
    'acetaldehyde':    r'$\boldsymbol{CH_3CHO}$',
    'propanenitrile':  r'$\boldsymbol{C_2H_5CN}$',
    'methyl_formate':  r'$\boldsymbol{HCOOCH_3}$',
    'glycolaldehyde':  r'$\boldsymbol{HOCH_2CHO}$',
    'ethylene_glycol': r"aGg'-$\boldsymbol{HOCH_2CH_2OH}$",
}


rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

#sou_coords=array([[52.26562,31.26771],[52.26571,31.26772],[52.26413,31.26711],[52.26283,31.26436]])[:2] # from Hsieh 2019
sou_coords=array([[52.2656318,31.2676934],[52.2657302,31.2676879]]) # from continuum
c1 = SkyCoord(sou_coords[0][0]*u.deg, sou_coords[0][1]*u.deg)
c2 = SkyCoord(sou_coords[1][0]*u.deg, sou_coords[1][1]*u.deg)
dis=c1.separation(c2).arcsec
print(dis)

Vlsr_vla4a=7.36
Vlsr_vla4b=9.33
color_vla4a = 'deepskyblue'
color_vla4b = 'orange'
def KepRot(ax, Mstar, inc = 0, center=[8.1,0], dis=293, c='k',ls='-'):
    Mstar = Mstar*con.M_sun
    R = np.arange(0.0,0.8,0.01)   # arcsec
    R_au = R*dis*con.au
    V=(con.G*Mstar/R_au)**0.5
    V_incCor = V*np.sin(deg2rad(inc))  # 0 for pole-on
    V_incCor = V_incCor.to('km/s').value
    ax.plot(center[0]+V_incCor,center[1]+R,c=c,ls=ls)
    ax.plot(center[0]-V_incCor,center[1]-R,c=c,ls=ls)


# --- 讀清單並依「分子」分組 ---------------------------------------------------
# --- 讀清單並依「分子」分組 ---------------------------------------------------
line_li = np.genfromtxt('/almabeegfs/scratch/ssp202525/data/PV_diagram/Line_spw.txt', dtype=str, delimiter=',')

from collections import defaultdict
grouped = defaultdict(list)

records = []
for row in line_li:
    lineFre = float(row[0])   # GHz
    tran    = row[1]          # k1, k2, ...
    mol_key = row[2].strip()  # 分子名：ethylene_glycol / methyl_formate ...
    Eu      = float(row[3])   # K
    band    = row[4]

    rec = dict(freq=lineFre, tran=tran, mol=mol_key, Eu=Eu, band=band)
    records.append(rec)
    grouped[mol_key].append(rec)

# 欄數＝單一分子含的最大轉線數；列數＝分子數
rows = max(len(v) for v in grouped.values())  # 一分子有幾條轉線 → 高度
cols = len(grouped)                           # 分子數 → 寬度

# 分子列的排序：先照 mol_formula 的鍵順序，其餘再字母序
order = [k for k in mol_formula.keys() if k in grouped] + sorted([k for k in grouped.keys() if k not in mol_formula])

mins, maxs = [], []
for mol in order:
    for rec in grouped[mol]:
        lineFre = rec['freq']
        mol_key = rec['mol']
        freq_str = str(int(round(lineFre * 1e4)))
        path = os.path.join(fits_dir, f"pv_{mol_key}_{freq_str}GHz_impv.fits")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到檔案：{path}")

        with pyfits.open(path) as hdul:
            header = hdul[0].header
            arr = np.squeeze(hdul[0].data)
            arr = np.rot90(arr)[::-1]

        # Jy/beam -> K（跟你畫圖時一致）
        lineFre_GHz = header['RESTFREQ']/1e9 if header.get('RESTFREQ', 0) > 1e9 else lineFre
        bmaj = header['BMAJ'] * 3600.0
        bmin = header['BMIN'] * 3600.0
        data_K = np.squeeze(Jy2Tbri(I=arr, bmaj=bmaj, bmin=bmin, fre=lineFre_GHz))

        if np.isfinite(data_K).any():
            mins.append(np.nanmin(data_K))
            maxs.append(np.nanmax(data_K))

vmin_global = float(np.nanmin(mins)) if mins else 0.0
vmax_global = float(np.nanmax(maxs)) if maxs else 1.0



panel_size = 2.5   
xgap_inch  = 0.30  
ygap_inch  = 0.30  


fig_w = panel_size * cols + (cols - 1) * xgap_inch
fig_h = panel_size * rows + (rows - 1) * ygap_inch
fig = plt.figure(figsize=(fig_w, fig_h))


xwid = panel_size / fig_w
ywid = panel_size / fig_h
xgap = xgap_inch / fig_w
ygap = ygap_inch / fig_h

left = 0.08   # 邊界留白(相對比例，可調)
top  = 0.95

last_img = None
for c, mol in enumerate(order):
    entries = sorted(grouped[mol], key=lambda rec: rec['Eu'], reverse=True)

    m = len(entries)
    start_row = rows - m         # 從第幾列開始放，前面留空白 → 圖片貼底

    for r in range(rows):
        ax = plt.axes([left + c*(xwid+xgap), top-(r+1)*(ywid+ygap), xwid, ywid])

        # r < start_row → 這些列是用來「墊高」的空白 (在最上面)
        if r < start_row:
            ax.axis('off')
            continue

        # 從底部開始放 entries，index 要扣掉 start_row
        rec = entries[r - start_row]
        lineFre = rec['freq']
        tran    = rec['tran']
        Eu      = rec['Eu']
        mol_key = rec['mol']

        freq_str = str(int(round(lineFre * 1e4)))
        fits_fi  = os.path.join(fits_dir, f"pv_{mol_key}_{freq_str}GHz_impv.fits")
        if not os.path.exists(fits_fi):
            raise FileNotFoundError(f"找不到檔案：{fits_fi}")

        with pyfits.open(fits_fi) as hdul:
            header = hdul[0].header
            data   = np.squeeze(hdul[0].data)
        data = np.rot90(data)[::-1]

        lineFre_GHz = header['RESTFREQ']/1e9 if header.get('RESTFREQ', 0) > 1e9 else lineFre
        bmaj = header['BMAJ'] * 3600.0
        bmin = header['BMIN'] * 3600.0
        data = np.squeeze(Jy2Tbri(I=data, bmaj=bmaj, bmin=bmin, fre=lineFre_GHz))

        extent_li = read_pv_coord(header)
        extent_li = np.array(extent_li)[[2, 3, 0, 1]]
        extent_li[2:] = -extent_li[2:]

        img = ax.imshow(
            data, extent=extent_li, origin='lower', interpolation='nearest',
            cmap='magma', vmin=vmin_global, vmax=vmax_global
        )
        last_img = img

        chem_title = mol_formula.get(mol_key, mol_key)
        # 只在「這欄最上面的那張有效圖」放標題（也就是 r == start_row）
        if r == start_row:
            ax.set_title(chem_title, fontsize=12, color='black')
        else:
            ax.set_title("")

        ax.plot(np.ones(2)*Vlsr_vla4a, [-1,20], c=color_vla4a, ls='--', zorder=10)
        ax.plot(np.ones(2)*Vlsr_vla4b, [-1,20], c=color_vla4b, ls='--', zorder=10)
        ax.plot([-1,20],[0,0],         c=color_vla4a, ls='--', zorder=10)
        ax.plot([-1,20],[dis,dis],     c=color_vla4b, ls='--', zorder=10)

        ax.set_xlim(2.9, 12.7)
        ax.set_ylim(-0.3, 0.55)
        ax.annotate(r'$E_{\rm u}$=' + f'{int(Eu)} K', xycoords='axes fraction',
                    xy=(0.02,0.8), c='w')
        ax.set_aspect('auto')
        ax.tick_params(axis='both', direction='in', labelbottom=False, labelleft=False, colors='w')

        if r == rows-1:
            ax.tick_params(axis='both', labelbottom=True, labelcolor='k')
            ax.set_xlabel(r'$V$ (km s$^{-1}$)')
        if c == 0:
            ax.tick_params(axis='both', labelleft=True, labelcolor='k')
            ax.set_ylabel(r'$\delta$X (arcsec)')

        KepRot(ax, 0.27, inc=22, center=[Vlsr_vla4a,0],  dis=293, c=color_vla4a)
        KepRot(ax, 0.15, inc=22, center=[Vlsr_vla4a,0],  dis=293, c=color_vla4a)
        KepRot(ax, 0.10, inc=22, center=[Vlsr_vla4a,0],  dis=293, c=color_vla4a)
        KepRot(ax, 0.60, inc=40, center=[Vlsr_vla4b,dis], dis=293, c=color_vla4b)



# 共用 colorbar
cax = plt.axes([
    left,                        
    top - rows*(ywid+ygap) - 0.08,   
    cols*(xwid+xgap) - xgap,     # 寬度涵蓋整個 panel
    0.025                        # colorbar 高度 (很薄一條)
])
cbar = plt.colorbar(last_img, cax=cax, orientation='horizontal')
cbar.set_label(r'(K)')

plt.subplots_adjust(
    bottom=top - rows*(ywid+ygap) + ygap,
    left=left,
    right=left + cols*(xwid+xgap) - xgap,
    top=top,
    wspace=xgap,
    hspace=ygap
)

plt.savefig('PV_diagram_by_molecule.pdf', dpi=300, bbox_inches='tight')
plt.savefig('PV_diagram_by_molecule.png', dpi=300, bbox_inches='tight')
plt.show()
