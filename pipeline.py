# Now presenting Eleanor's pipeline to identify outliers in the AKARI L and S maps and fill them with scaled IRAS data
# Updated: July 14, 2025


# loading packages

import numpy as np
import healpy as hp
import pickle
from scipy.stats import linregress
import functions as fun


# defining path to the pipeline

path = '/Users/eleanorgallay/SULIcmb/pipe/'


#  loading our data

AKARI_L = hp.read_map(path + 'akari_WideL_1_4096.fits')
AKARI_S = hp.read_map(path + 'akari_WideS_1_4096.fits')
IRAS = hp.read_map(path + 'IRIS_combined_SFD_really_nohole_nosource_4_2048.fits')
ps = hp.read_map(path + 'irps_mask_for_akari.fits')   # point source mask from Jacques
AKARI_65 = hp.read_map('/Users/eleanorgallay/SULIcmb/code/old/akari_65_1_4096.fits')
AKARI_160 = hp.read_map('/Users/eleanorgallay/SULIcmb/code/old/akari_160_1_4096.fits')



# step 1 - compute difference maps, mask negative pixels and point sources

fun.diff_map(akari_data=AKARI_L, iras_data=IRAS, ps_data=ps, dmap_filename='diff_L.fits', ma_filename='ma_diff_L.npz', path=path, smooth_nside=8)
fun.diff_map(akari_data=AKARI_S, iras_data=IRAS, ps_data=ps, dmap_filename='diff_S.fits', ma_filename='ma_diff_S.npz', path=path, smooth_nside=8)
fun.diff_map(akari_data=AKARI_65, iras_data=IRAS, ps_data=ps, dmap_filename='diff_65.fits', ma_filename='ma_diff_65.npz', path=path, smooth_nside=8)
fun.diff_map(akari_data=AKARI_160, iras_data=IRAS, ps_data=ps, dmap_filename='diff_160.fits', ma_filename='ma_diff_160.npz', path=path, smooth_nside=8)


# step 2 - find outliers

L_loaded = np.load('ma_diff_L.npz')
diff_L_masked = np.ma.masked_array(L_loaded['data'], mask=L_loaded['mask'])
S_loaded = np.load('ma_diff_S.npz')
diff_S_masked = np.ma.masked_array(S_loaded['data'], mask=S_loaded['mask'])   # Reading in masked difference maps

a65_loaded = np.load('ma_diff_65.npz')
diff_65_masked = np.ma.masked_array(a65_loaded['data'], mask=a65_loaded['mask'])
a160_loaded = np.load('ma_diff_160.npz')
diff_160_masked = np.ma.masked_array(a160_loaded['data'], mask=a160_loaded['mask'])

fun.findoutliers(diffmap=diff_L_masked, filename='L_outliers_s3.pkl', path = path, super_nside=32)
fun.findoutliers(diffmap=diff_S_masked, filename='S_outliers_s3.pkl', path = path)
fun.findoutliers(diffmap=diff_65_masked, filename='a65_outliers_s3.pkl', path = path)
fun.findoutliers(diffmap=diff_160_masked, filename='a160_outliers_s3.pkl', path = path)


# step 3 - mask outliers, negative pixels, and neighbors

with open(path + 'L_outliers_s3.pkl', 'rb') as f:
    L_sig3 = pickle.load(f)
with open(path + 'S_outliers_s3.pkl', 'rb') as f:
    S_sig3 = pickle.load(f)   # outlier pixel indices

with open(path + 'a65_outliers_s3.pkl', 'rb') as f:
    a65_sig3 = pickle.load(f)
with open(path + 'a160_outliers_s3.pkl', 'rb') as f:
    a160_sig3 = pickle.load(f)


L_negs = np.where(AKARI_L <= 0)[0]
S_negs = np.where(AKARI_S <= 0)[0]   # negative pixel indices
a65_negs = np.where(AKARI_65 <= 0)[0]
a160_negs = np.where(AKARI_160 <= 0)[0]


L_outliers = []
L_outliers.extend(L_sig3)
L_outliers.extend(L_negs)
S_outliers = []
S_outliers.extend(S_sig3)
S_outliers.extend(S_negs)   # combining these two arrays of pixel indices

a65_outliers = []
a65_outliers.extend(a65_sig3)
a65_outliers.extend(a65_negs)
a160_outliers = []
a160_outliers.extend(a160_sig3)
a160_outliers.extend(a160_negs)

fun.maskoutliers(data=AKARI_L, outliers=L_outliers, nsideout=2048, fwhmin=1.5, fwhmout=5, filename='L_outliers_masked', path=path)
fun.maskoutliers(data=AKARI_S, outliers=S_outliers, nsideout=2048, fwhmin=1.3, fwhmout=5, filename='S_outliers_masked', path=path)
fun.maskoutliers(data=AKARI_65, outliers=a65_outliers, nsideout=2048, fwhmin=1.05, fwhmout=5, filename='a65_outliers_masked', path=path)
fun.maskoutliers(data=AKARI_160, outliers=a160_outliers, nsideout=2048, fwhmin=1.5, fwhmout=5, filename='a160_outliers_masked', path=path)


# step 4 - filling in outliers

with np.load('L_outliers_masked.npz') as npz:
    L_outlier_ma = np.ma.MaskedArray(**npz)
with np.load('S_outliers_masked.npz') as npz:
    S_outlier_ma = np.ma.MaskedArray(**npz)
with np.load('a65_outliers_masked.npz') as npz:
    a65_outlier_ma = np.ma.MaskedArray(**npz)
with np.load('a160_outliers_masked.npz') as npz:
    a160_outlier_ma = np.ma.MaskedArray(**npz)

with np.load('L_outliers_masked_smooth.npz') as npz:
    L_outlier_ma_smooth = np.ma.MaskedArray(**npz)
with np.load('S_outliers_masked_smooth.npz') as npz:
    S_outlier_ma_smooth = np.ma.MaskedArray(**npz)
with np.load('a65_outliers_masked_smooth.npz') as npz:
    a65_outlier_ma_smooth = np.ma.MaskedArray(**npz)
with np.load('a160_outliers_masked_smooth.npz') as npz:
    a160_outlier_ma_smooth = np.ma.MaskedArray(**npz)

fun.fill(akari_data=L_outlier_ma, smooth_akari_data= L_outlier_ma_smooth, iras_data=IRAS, fwhmin=1.5, fwhmout=5,  filename='filled_L_s3_n32', path=path, super_nside=32)
fun.fill(akari_data=S_outlier_ma, smooth_akari_data= S_outlier_ma_smooth, iras_data=IRAS, fwhmin=1.3, fwhmout=5, filename='filled_S_s3_n16', path=path)
fun.fill(akari_data=a65_outlier_ma, smooth_akari_data= a65_outlier_ma_smooth, iras_data=IRAS, fwhmin=1.05, fwhmout=5, filename='filled_65_s3_n32', path=path)
fun.fill(akari_data=a160_outlier_ma, smooth_akari_data= a160_outlier_ma_smooth, iras_data=IRAS, fwhmin=1.5, fwhmout=5, filename='filled_160_s3_n16', path=path)




# GREAT CIRCLE SHENANAGINS

# r = hp.Rotator(coord = ['G', 'E'])
# nside = hp.get_nside(IRAS)
# npix = hp.nside2npix(nside)
# ipix = np.arange(npix)

# lon, lat = hp.pix2ang(nside, ipix, lonlat=True) 

# lon_bins = np.arange(0, 180, 0.25)
# lat_bins = np.arange(-10, 10, 0.25)

# shape = (len(lat_bins), len(lon_bins))

# slopes = np.full(shape, np.nan , dtype=float)
# intercepts = np.full(shape, np.nan , dtype=float)
# errors = np.full(shape, np.nan, dtype=float)

# slope_map = np.zeros_like(ipix, dtype=float)
# intercept_map = np.zeros_like(ipix, dtype=float)
# error_map = np.zeros_like(ipix, dtype=float)

# for i, lon_val in enumerate(lon_bins):
#     for j, lat_val in enumerate(lat_bins):
        
#         # Define conditions 
        
#         lon_cond = (lon >= lon_val) & (lon < lon_val + 0.25)
#         lat_cond = (lat >= lat_val) & (lat < lat_val + 0.5)
#         pixels = ipix[lon_cond & lat_cond]

#         if len(pixels) < 2:
#             continue  # Not enough data points to regress

#         ak = akarifilled[pixels]
#         ir = IRAS[pixels]

#         slope, intercept, r_value, p_value, std_err = linregress(ir, ak)

#         slopes[j, i] = slope
#         intercepts[j, i] = intercept
#         errors[j, i] = std_err

#         slope_map[pixels] = slope
#         intercept_map[pixels] = intercept
#         error_map[pixels] = std_err

# with open(path + 'great_circle_slopes.pkl', 'wb') as f:
#     pickle.dump(slope_map, f)

# with open(path + 'great_circle_intercepts.pkl', 'wb') as f:
#     pickle.dump(intercept_map, f)

# with open(path + 'great_circle_errors.pkl', 'wb') as f:
#     pickle.dump(error_map, f)