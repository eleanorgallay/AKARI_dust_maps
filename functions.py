# Here are all of the functions I use in my pipeline to identify and then fill outliers in the AKARI maps with scaled IRAS data
# Updated: July 9, 2025

# imports

import numpy as np
import healpy as hp
import skytools as st
import pickle 
from scipy.stats import linregress


# Shamik's median filter function, which I adapted to account for masked arrays with UNSEEN values

def median_filter(map_in, nside_super, fwhm_smooth=None, mask=None, theshold=0.):
    """
    Applies a median filter to a HEALPix map at a degraded resolution and optionally smooths the result.

    Parameters
    ----------
    map_in : np.ndarray
        Input HEALPix map.
    nside_super : int
        Target NSIDE for superpixel binning (coarser resolution).
    fwhm_smooth : float, optional
        Full-width-half-maximum (FWHM) for Gaussian smoothing in arcminutes.
        If None, a default value based on the pixel area of `nside_super` is used.
    mask : np.ndarray, optional
        Binary or float mask (same shape as `map_in`) where values < 0.9 will be masked.
    theshold : float, optional
        Value to assign to masked pixels before filtering. Default is 0.

    Returns
    -------
    np.ndarray
        Smoothed and upsampled HEALPix map at original NSIDE.
    """
    
    nside_in = hp.get_nside(map_in)
    order_diff = int(np.log2(nside_in / nside_super))
    npix_super = hp.nside2npix(nside_super)

    nest_slice = 1
    for ord in range(order_diff):
        nest_slice += 4**ord * 3

    nest_slice = np.int32(nest_slice)
    
    if isinstance(mask, np.ndarray):
        map_in[mask < 0.9] = theshold

    map_in_masked = np.ma.masked_where(map_in == hp.UNSEEN, map_in)
    
    map_nest_super = np.reshape(hp.reorder(map_in_masked, r2n=True), (npix_super, nest_slice))
    median_super = hp.reorder(np.ma.median(map_nest_super, axis=1).filled(hp.UNSEEN), n2r=True)

    if fwhm_smooth == None:
        fwhm_smooth = 3 * np.sqrt(hp.nside2pixarea(nside_super, degrees=True)) * 60

    return st.change_resolution(median_super, nside_out=nside_in, fwhm_out=fwhm_smooth)





# Here comes a function to take in an AKARI and IRAS map and compute a difference map (AKARI - scaled IRAS)


def diff_map(akari_data, iras_data, ps_data, dmap_filename, ma_filename, path, smooth_nside):
    """
    Computes the difference map between AKARI and IRAS maps, 
    where is IRAS data is scaled by a ratio 
    determined by computing the median of the maps at a lower (smoothed) nside.

    Parameters
    ----------
    akari_data : np.ndarray
        HEALPix AKARI map (nside 4096)
    iras_data : np.ndarray
        HEALPix IRAS map (nside 2048)
    ps_data : np.ndarray
        Point source mask (1 = not a point source, unmasked, 0 = point source, masked).
    dmap_filename : str
        Filename to save the resulting difference map (FITS).
    ma_filename : str
        Filename to save the masked array (NPZ).
    path : str
        Directory to save output files.
    smooth_nside : int
        NSIDE used for median filtering to compute scaling ratio.

    Returns
    -------
    None
    """
    
    nside = hp.get_nside(akari_data)
    iras = hp.pixelfunc.ud_grade(iras_data, nside) # matching nside of akari data
    
    # this part is Shamik's mask
    sp_mask = np.zeros(hp.nside2npix(nside))
    eps = 0.05
    l,b = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    sp_mask = eps + (1 - eps) * np.sin(np.deg2rad(np.abs(b)))

    # masking negative pixels in AKARI maps

    negmask = akari_data <= 0
    akari_pos = np.copy(akari_data)
    akari_pos[negmask] = hp.UNSEEN # setting negative values in AKARI map to UNSEEN

    # Applying Shamik's mask to AKARI and IRAS data

    akari_sp_masked = akari_pos * sp_mask
    iras_sp_masked = iras * sp_mask

    # Using Shamik's median filter function to calculate the median of AKARI and IRIS data at nside 8. Then taking the ratio to determine what factor to scale the IRIS map

    smooth_akari = median_filter(akari_sp_masked, smooth_nside)
    smooth_iras = median_filter(iras_sp_masked, smooth_nside)

    ratio = smooth_akari/smooth_iras

    # Taking the difference between the AKARI map and the scaled IRIS map

    diff = (akari_data - (iras * ratio))

    # Saving our difference map

    hp.write_map(path + str(dmap_filename), diff, overwrite=True)

    # Masking negative pixels and point sources, saving masked array

    neg_cond = akari_data <= 0
    ps_cond = ps_data == 0 
    
    diff_masked = np.ma.masked_where((neg_cond) | (ps_cond), diff)
    np.savez(path + str(ma_filename), data=diff_masked.data, mask=diff_masked.mask)





# My function to find outliers in data that has already been masked for negative pixels and point sources

def findoutliers(diffmap, filename, path, super_nside=16, nsigma=3, threshold=3):
    """
    Identifies statistical outliers in a HEALPix map within superpixels using an iterative sigma-clipping algorithm.

    Parameters
    ----------
    diffmap : np.ma.MaskedArray
        Difference map where original negative pixels and point sources are masked. 
    filename : str
        Filename to save the list of outlier pixel indices (pickle).
    path : str
        Directory to save the output file.
    super_nside : int, optional
        NSIDE for superpixel binning. Default is 16.
    nsigma : float, optional
        Number of standard deviations for sigma clipping. Default is 3.
    threshold : float, optional
        Convergence threshold on standard deviation percent change. Default is 3%.

    Returns
    -------
    None
    """

    nside = hp.get_nside(diffmap)
    npix = hp.nside2npix(nside)
    super_npix = hp.nside2npix(super_nside)
    finegrid = np.arange(npix)
    vecgrid = hp.pix2vec(nside, finegrid)
    supergrid = hp.vec2pix(super_nside, *vecgrid)
    out_mask = diffmap.mask
    data = diffmap.data

    globaloutliers = []

    for superpixel in np.arange(super_npix):

        pix_mask = (supergrid != superpixel) # Condition: pixel index in nside 4096 does not fall in superpixel at nside 16 (or whatever you set)
        mask = pix_mask | (out_mask).copy() # both pixels outside our superpixel and negative pixels are masked 

        medians = []
        stds = []
        percentchanges = []
        outlierindices = []

        i = 0

        while True:

            # 1. Get indices of unmasked (valid) values (positive pixels in superpixel)
            valid_indices = np.where(~mask)[0]

            # 3. Valid pixel values
            values = data[valid_indices]


            med = np.ma.median(values)
            sigma = np.ma.std(values)
            sig = (values > med - nsigma * sigma) & (values < med + nsigma * sigma)

            medians.append(med)
            stds.append(sigma)

            new_outlier_indices = valid_indices[~sig]
            outlierindices = np.concatenate((outlierindices, new_outlier_indices))
            mask[new_outlier_indices] = True

            if i > 0:
                prev_std = stds[i - 1]
                percentchange = (sigma - prev_std) / prev_std * 100
                percentchanges.append(percentchange)

                if np.abs(percentchange) < threshold:
                    print(f"Superpixel {superpixel} Converged at iteration {i}")
                    print(str(len(np.unique(outlierindices)))+" outliers found")
                    break


            i += 1
        
        
        globaloutliers.extend(np.unique(outlierindices).astype(int))
    
    globaloutliers = np.array(globaloutliers)
    
    with open(path + str(filename), 'wb') as f:
        pickle.dump(globaloutliers, f)





# This function will remove isolated pixels from our list of outliers, and then mask outliers, negative pixels, and neighbors

def maskoutliers(data, outliers, nsideout, fwhmin, fwhmout, filename, path):
    """
    Masks outlier pixels and their neighbors, performs smoothing with an SHT, and saves the result at a new resolution. 
    Also saves the mask at the original resolution (4096 for AKARI data).

    Parameters
    ----------
    data : np.ndarray
        Input HEALPix map.
    outliers : np.ndarray
        Indices of outlier pixels.
    nsideout : int
        NSIDE of the output map after smoothing.
    fwhmin : float
        Input beam FWHM in arcminutes.
    fwhmout : float
        Output beam FWHM in arcminutes.
    filename : str
        Filename to save the smoothed, masked map (NPZ).
    oldres_filename : str
        Filename to save the original-resolution masked map (NPZ).
    path : str
        Directory to save the output files.

    Returns
    -------
    None
    """

    nside = hp.get_nside(data)
    npix = hp.nside2npix(nside)
    
    # creating a mask with our current list of outliers
    outliermask = np.ones((npix, ))
    outliermask[outliers] = 0
 
    # removing isolated pixels
    true_outliers = []

    for index in outliers:
        neighbors = hp.get_all_neighbours(nside, index)
        if np.any(outliermask[neighbors] == 0):
            true_outliers.append(index)
        else:
            outliermask[index] = 1 

    # creating a masked array of our data where outliers are masked
    ma = np.ma.array(data, mask=False)
    ma.mask[true_outliers] = True   

    # masking neighbors
        
    for i in true_outliers:
        neighbors = hp.get_all_neighbours(nside, i)
        ma.mask[neighbors] = True

    # saving mask (nside 4096)


    np.savez(str(path) + str(filename), data=ma.data, mask=ma.mask)

    
    # Setting masked values to unseen 

    ma_unseen = ma.filled(hp.UNSEEN)
    
    # performing a spherical harmonic transform on the masked data to smooth to beam width of 5 arcminutes and nside 2048
    sht = st.change_resolution(ma_unseen, nside_out= nsideout, lmax_sht= 3* nsideout -1, fwhm_in= fwhmin, fwhm_out= fwhmout)

    # masking outliers at nside 2048

    maskout = st.mask_udgrade(ma.mask, nsideout)
    ma_smooth = np.ma.masked_where(maskout, sht)


    # saving our smoothed masked array

    np.savez(str(path) + str(filename) + '_smooth', data=ma_smooth.data, mask=ma_smooth.mask)

    




# My function to fill in outliers with scaled IRAS data in superpixels of a given nside

def fill(akari_data, smooth_akari_data, iras_data, fwhmin, fwhmout, filename, path, super_nside=16):
    """
    Fills masked regions in the AKARI map using scaled IRAS data via linear regression in superpixels. 
    First fills in at nside 4096, then smooths to 5' beam, then refills at 5' resolution.

    Parameters
    ----------
    akari_data : np.ma.MaskedArray
        AKARI map with outliers and negative pixels masked. This map should match NSIDE of IRAS data.
    iras_data : np.ndarray
        IRAS map.
    filename : str
        Filename to save the filled AKARI map (FITS).
    path : str
        Directory to save the output file.
    super_nside : int, optional
        NSIDE for defining superpixels used for regression. Default is 16.

    Returns
    -------
    None
    """

    filled = np.ma.copy(akari_data)

    nside = hp.get_nside(smooth_akari_data)
    nsideout = hp.get_nside(iras_data)
    npix = hp.nside2npix(nside)
    super_npix = hp.nside2npix(super_nside)
    finegrid = np.arange(npix)
    vecgrid = hp.pix2vec(nside, finegrid)
    supergrid = hp.vec2pix(super_nside, vecgrid[0], vecgrid[1], vecgrid[2])

    slopes = []
    intercepts = []
    slope_errors = []
    intercept_errors = []

    for pixel in np.arange(super_npix):

        mask = (supergrid == pixel)
        test_mask = smooth_akari_data.mask | ~mask  # outliers or data outside of superpixel
        goodidx = (~test_mask).nonzero()[0] # valid data inside superpixel
        goodpix = akari_data.data[goodidx]

        if len(goodpix) > 10:
            lower, upper = np.percentile(goodpix, [10, 90])
            extmask = (goodpix < lower) | (goodpix > upper)
            test_mask[goodidx[extmask]] = True # masking extreme pixel values

            akari = smooth_akari_data[~test_mask] 
            iras = iras_data[~test_mask]

            slope, intercept, _, _, std_err, intercept_stderr = linregress(iras, akari)
            slopes.append(slope)
            intercepts.append(intercept)
            slope_errors.append(std_err)
            intercept_errors.append(intercept_stderr)

            if pixel % 100 == 0:
                print(f"{pixel} done")


    with open(path + str(filename) + '_slopes.pkl', 'wb') as f:
        pickle.dump(slopes, f)

    with open(path + str(filename) + '_intercepts.pkl', 'wb') as f:
        pickle.dump(intercepts, f)

    with open(path + str(filename) + '_slope_errors.pkl', 'wb') as f:
        pickle.dump(slope_errors, f)

    with open(path + str(filename) + '_intercept_errors.pkl', 'wb') as f:
        pickle.dump(intercept_errors, f)

    # filling akari data at original resolution, with scaled iras data at 5' resolution

    slopes_4096 = hp.ud_grade(slopes, 4096)
    intercepts_4096 = hp.ud_grade(intercepts, 4096)
    iras_4096 = hp.ud_grade(iras_data, 4096)

    fill_iras_4096 = slopes_4096 * iras_4096 + intercepts_4096
    filled.data[akari_data.mask] = fill_iras_4096[akari_data.mask]
    filled.mask[akari_data.mask] = False

    # smoothing filled map to 5' resolution

    sht = st.change_resolution(filled, nside_out= nsideout, lmax_sht= 3* nsideout -1, fwhm_in= fwhmin, fwhm_out= fwhmout)

    smooth_ma = np.ma.array(sht, mask = smooth_akari_data.mask)
    filled_smooth = np.ma.copy(smooth_ma)

    # refilling smoothed map with scaled iras data at 5' resolution

    slopes_2048 = hp.ud_grade(slopes, 2048)
    intercepts_2048 = hp.ud_grade(intercepts, 2048)

    fill_iras = slopes_2048 * iras_data + intercepts_2048
    filled_smooth.data[smooth_akari_data.mask] = fill_iras[smooth_akari_data.mask]
    filled_smooth.mask[smooth_akari_data.mask] = False

    hp.write_map(path + str(filename) + '.fits', filled_smooth, overwrite=True)