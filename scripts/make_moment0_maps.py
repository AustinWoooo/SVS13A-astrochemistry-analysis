importfits(
    fitsimage='propanenitrile_2206609.fits',
    imagename='propanenitrile_2206609.image',
    overwrite=True
)

# === User inputs ===
cube = 'propanenitrile_2206609.image'     # cube name
chans_noise = '0~50'            # channel range for noise estimation
chans_moment = '115~181'        # channel range for moment map
sigma_clip = 3.0                # threshold: N * rms
masked_cube = 'masked.image'   # temporary cube with masked data
mom0_out = 'propanenitrile_2206609_mom0_masked.image'
fits_out = 'propanenitrile_2206609_mom0_masked.fits'

# === 1. Estimate RMS ===
print(">> Estimating noise from channel range:", chans_noise)
rms = imstat(imagename=cube, chans=chans_noise)['rms'][0]
print(">> Measured RMS =", rms)

# === 2. Create masked cube: keep only pixels > Nσ ===
expr = f'iif(IM0 > {sigma_clip}*{rms}, IM0, 0.0)'
print(">> Creating masked cube with expr:", expr)

immath(
    imagename=cube,
    expr=expr,
    outfile=masked_cube
)

# === 3. Compute moment 0 on masked cube ===
print(">> Calculating moment 0 map...")
immoments(
    imagename=masked_cube,
    moments=[0],
    axis='spectral',
    chans=chans_moment,
    outfile=mom0_out
)

# === 4. Export to FITS ===
print(">> Exporting to FITS:", fits_out)
exportfits(
    imagename=mom0_out,
    fitsimage=fits_out,
    velocity=True
)

print(">> Done.")
