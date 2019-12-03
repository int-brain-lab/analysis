import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compute_rfs(spike_times, spike_clusters, stimulus_times, stimulus, lags=8, binsize=0.025):
    """
    Compute receptive fields from locally sparse noise stimulus for all recorded neurons; uses a
    PSTH-like approach that averages responses from each neuron for each pixel flip

    Parameters
    ----------
    spike_times : array-like
        array of spike times
    spike_clusters : array-like
        array of cluster ids associated with each entry of `spike_times`
    stimulus_times : array-like
        array of stimulus presentation times with shape (M,)
    stimulus : np.ndarray
        array of pixel values wtih shape (M, y_pix, x_pix)
    lags : int, optional
        temporal dimension of receptive field
    binsize : float, optional
        length of each lag (seconds)

    Returns
    -------
    dict
        "on" and "off" receptive fields (values are lists); each rf is shape (lags, y_pix, x_pix)

    """

    from brainbox.processing import bincount2D

    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)
    _, y_pix, x_pix = stimulus.shape
    stimulus = stimulus.astype('float')
    subs = ['on', 'off']
    rfs = {sub: np.zeros(shape=(n_clusters, y_pix, x_pix, lags + 1)) for sub in subs}
    flips = {sub: np.zeros(shape=(y_pix, x_pix)) for sub in subs}

    gray = np.median(stimulus)
    # loop over time points
    for i, t in enumerate(stimulus_times):
        # skip first frame since we're looking for pixels that flipped
        if i == 0:
            continue
        # find pixels that flipped
        frame_change = stimulus[i, :, :] - gray
        ys, xs = np.where((frame_change != 0) & (stimulus[i - 1, :, :] == gray))
        # loop over changed pixels
        for y, x in zip(ys, xs):
            if frame_change[y, x] > 0:  # gray -> white
                sub = 'on'
            else:  # black -> white
                sub = 'off'
            # bin spikes in the binsize*lags seconds following this flip
            t_beg = t
            t_end = t + binsize * lags
            idxs_t = (spike_times >= t_beg) & (spike_times < t_end)
            binned_spikes, _, cluster_idxs = bincount2D(
                spike_times[idxs_t], spike_clusters[idxs_t], xbin=binsize, xlim=[t_beg, t_end])
            # insert these binned spikes into the rfs
            _, cluster_idxs, _ = np.intersect1d(cluster_ids, cluster_idxs, return_indices=True)
            rfs[sub][cluster_idxs, y, x, :] += binned_spikes
            # record flip
            flips[sub][y, x] += 1

    # normalize spikes by number of flips
    for sub in rfs:
        for y in range(y_pix):
            for x in range(x_pix):
                if flips[sub][y, x] != 0:
                    rfs[sub][:, y, x, :] /= flips[sub][y, x]

    # turn into list
    rfs_list = {}
    for sub in subs:
        rfs_list[sub] = [np.transpose(rfs[sub][i, :, :, :], (2, 0, 1)) for i in range(n_clusters)]
    return rfs_list


def compute_rfs_corr(spike_times, spike_clusters, stimulus_times, stimulus, lags=8, binsize=0.025):
    """
    Compute receptive fields from locally sparse noise stimulus for all recorded neurons; uses a
    reverse correlation approach.

    Parameters
    ----------
    spike_times : array-like
        array of spike times
    spike_clusters : array-like
        array of cluster ids associated with each entry of `spike_times`
    stimulus_times : array-like
        array of stimulus presentation times with shape (M,)
    stimulus : np.ndarray
        array of pixel values wtih shape (M, y_pix, x_pix)
    lags : int, optional
        temporal dimension of receptive field
    binsize : float, optional
        length of each lag (seconds)

    Returns
    -------
    dict
        "on" and "off" receptive fields (values are lists); each rf is shape (lags, y_pix, x_pix)

    """

    from brainbox.processing import bincount2D
    from scipy.signal import correlate

    # bin spikes
    indx_t = (spike_times > np.min(stimulus_times)) & \
             (spike_times < np.max(stimulus_times))
    binned_spikes, ts_binned_spikes, cluster_ids = bincount2D(
        spike_times[indx_t], spike_clusters[indx_t], xbin=binsize)
    n_clusters = len(cluster_ids)

    _, y_pix, x_pix = stimulus.shape
    stimulus = stimulus.astype('float')
    gray = np.median(stimulus)

    subs = ['on', 'off']
    rfs = {sub: np.zeros(shape=(n_clusters, y_pix, x_pix, lags + 1)) for sub in subs}

    # for indexing output of convolution
    i_end = binned_spikes.shape[1]
    i_beg = i_end - lags

    for sub in subs:
        for y in range(y_pix):
            for x in range(x_pix):
                # find times that pixels flipped
                diffs = np.concatenate([np.diff(stimulus[:, y, x]), [0]])
                if sub == 'on':  # gray -> white
                    changes = (diffs > 0) & (stimulus[:, y, x] == gray)
                else:
                    changes = (diffs < 0) & (stimulus[:, y, x] == gray)
                t_change = np.where(changes)[0]

                # put on same timescale as neural activity
                binned_stim = np.zeros(shape=ts_binned_spikes.shape)
                for t in t_change:
                    stim_time = stimulus_times[t]
                    # find nearest time in binned spikes
                    idx = np.argmin((ts_binned_spikes - stim_time) ** 2)
                    binned_stim[idx] = 1

                for n in range(n_clusters):
                    # cross correlate signal with spiking activity
                    # NOTE: scipy's correlate function is appx two orders of
                    # magnitude faster than numpy's correlate function on
                    # relevant data size; perhaps scipy uses FFT? Not in docs.
                    cc = correlate(binned_stim, binned_spikes[n, :], mode='full')
                    rfs[sub][n, y, x, :] = cc[i_beg:i_end + 1]

    # turn into list
    rfs_list = {}
    for sub in subs:
        rfs_list[sub] = [np.transpose(rfs[sub][i, :, :, :], (2, 0, 1)) for i in range(n_clusters)]
    return rfs_list


def compute_rf_svds(rfs, scale='none'):
    """
    Perform SVD on the spatiotemporal rfs and return the first spatial and first temporal
    components. Used for denoising purposes.

    Parameters
    ----------
    rfs : dict
        dictionary of "on" and "off" receptive fields (values are lists); each rf is of shape
        (n_bins, y_pix, x_pix) - output of `compute_rfs` or `compute_rfs_corr`
    scale : str, optional
        scale either the spatial or temporal component (or neither) by the singular value
        'spatial' | 'temporal' | 'none'

    Returns
    -------
    dict
        dict with 'spatial' and 'temporal' keys; the values are lists of the specified
        components, on for each input rf

    """

    from scipy.linalg import svd
    rfs_svd = {key1: {key2: [] for key2 in rfs.keys()} for key1 in ['spatial', 'temporal']}
    n_bins, y_pix, x_pix = rfs['on'][0].shape
    # loop over rf type
    for sub_type, subs in rfs.items():
        # loop over clusters
        for sub in subs:
            # reshape take PSTH and rearrange into n_pixels x n_bins
            sub_reshaped = np.reshape(sub, (n_bins, y_pix * x_pix))
            # svd
            u, s, v = svd(sub_reshaped.T)
            # keep first spatial dim and temporal trace
            sign = -1 if np.median(v[0, :]) < 0 else 1
            rfs_svd['spatial'][sub_type].append(sign * np.reshape(u[:, 0], (y_pix, x_pix)))
            if scale == 'spatial':
                rfs_svd['spatial'][sub_type][-1] *= s[0]
            rfs_svd['temporal'][sub_type].append(sign * v[0, :])
            if scale == 'temporal':
                rfs_svd['temporal'][sub_type][-1] *= s[0]
    return rfs_svd


def find_peak_responses(rfs):
    """
    Find peak response across time, space, and receptive field type ("on" and "off")

    Parameters
    ----------
    rfs : dict
        dictionary of receptive fields (output of `compute_rfs`); each rf is of size
        (lags, y_pix, y_pix)

    Returns
    -------
    dict
        dictionary containing peak rf time slice for both "on" and "off" rfs (values are lists)

    """
    rfs_peak = {key: [] for key in rfs.keys()}
    # loop over rf type
    for sub_type, subs in rfs.items():
        # loop over clusters
        for sub in subs:
            # max over space for each time point
            s_max = np.max(sub, axis=(1, 2))
            # take time point with largest max
            rfs_peak[sub_type].append(sub[np.argmax(s_max), :, :])
    return rfs_peak


def interpolate_rfs(rfs, bin_scale=0.5):
    """
    Bilinear interpolation of receptive fields

    Parameters
    ----------
    rfs : dict
        dictionary of receptive fields (single time slice)
    bin_scale : float, optional
        scaling factor to determine number of bins for interpolation; e.g. bin_scale=0.5 doubles the
        number of bins in both directions

    Returns
    -------
    dict
        dictionary of interpolated receptive fields (values are lists)

    """
    from scipy.interpolate import interp2d
    rfs_interp = {'on': [], 'off': []}
    y_pix, x_pix = rfs['on'][0].shape
    x_grid = np.arange(x_pix)
    y_grid = np.arange(y_pix)
    x_grid_new = np.arange(-0.5, x_pix, bin_scale)
    y_grid_new = np.arange(-0.5, y_pix, bin_scale)
    # loop over rf type
    for sub_type, subs in rfs.items():
        # loop over clusters
        for sub in subs:
            f = interp2d(y_grid, x_grid, sub)
            rfs_interp[sub_type].append(f(y_grid_new, x_grid_new))
    return rfs_interp


def extract_blob(array, y, x):
    """
    Extract contiguous blob of `True` values in a boolean array starting at the point (y, x)

    Parameters
    ----------
    array : np.ndarray
        2D boolean array
    y : int
        initial y pixel
    x : int
        initial x pixel

    Return
    list
        list of blob indices

    """
    y_pix, x_pix = array.shape
    blob = []
    pixels_to_check = [[y, x]]
    while len(pixels_to_check) != 0:
        pixel = pixels_to_check.pop(0)
        blob.append(pixel)
        y = pixel[0]
        x = pixel[1]
        # check N, S, E, W
        for idx in [[y - 1, x], [y + 1, x], [y, x + 1], [y, x - 1]]:
            if idx in blob:
                continue
            # check boundaries
            xi = idx[0]
            yi = idx[1]
            if (0 <= yi < y_pix) and (0 <= xi < x_pix):
                if array[xi, yi] and idx not in pixels_to_check:
                    pixels_to_check.append(idx)
    return blob


def extract_blobs(array):
    """
    Extract contiguous blobs of `True` values in a boolean array

    Parameters
    ----------
    array : np.ndarray
        2D boolean array

    Returns
    -------
    list
        list of lists of blob indices

    """
    y_pix, x_pix = array.shape
    processed_pix = []
    blobs = []
    for y in range(y_pix):
        for x in range(x_pix):
            # find a blob starting at this point
            if not array[y, x]:
                continue
            if [y, x] in processed_pix:
                continue
            blob = extract_blob(array, y, x)
            for pixel in blob:
                processed_pix.append(pixel)
            blobs.append(blob)
    return blobs


def find_contiguous_pixels(rfs, threshold=0.35):
    """
    Calculate number of contiguous pixels in a thresholded version of the receptive field

    Parameters
    ----------
    rfs : dict
        dictionary of receptive fields (single time slice)
    threshold : float, optional
        pixels below this fraction of the maximum firing are set to zero before contiguous pixels
        are calculated

    Returns
    -------
    dict
        dictionary of contiguous pixels for each rf type ("on and "off")

    """

    # store results
    n_clusters = len(rfs['on'])
    max_fr = np.zeros(n_clusters)
    contig_pixels = {'on': np.zeros(n_clusters), 'off': np.zeros(n_clusters)}

    # compute max firing rate for each cluster
    for sub_type, subs in rfs.items():
        for n, sub in enumerate(subs):
            max_fr[n] = np.max([max_fr[n], np.max(sub)])

    # compute max number of contiguous pixels
    for sub_type, subs in rfs.items():
        for n, sub in enumerate(subs):
            # compute rf mask using threshold
            rf_mask = sub > (threshold * max_fr[n])
            # extract contiguous pixels (blobs)
            blobs = extract_blobs(rf_mask)
            # save size of largest blob
            if len(blobs) != 0:
                contig_pixels[sub_type][n] = np.max(
                    [len(blob) for blob in blobs])

    return contig_pixels


def compute_rf_areas(rfs, bin_scale=0.5, threshold=0.35):
    """
    Compute receptive field areas as described in:
    Durand et al. 2016
    "A Comparison of Visual Response Properties in the Lateral Geniculate Nucleus
    and Primary Visual Cortex of Awake and Anesthetized Mice"

    Parameters
    ----------
    rfs : dict
        dictionary of receptive fields (dict keys are 'on' and 'off'); output of `compute_rfs`
    bin_scale : float, optional
        scaling for interpolation (e.g. 0.5 doubles bins)
    threshold : float, optional
        pixels below this fraction of the maximum firing are set to zero before contiguous pixels
        are calculated

    Returns
    -------
    dict
        dictionary of receptive field areas for each type ("on" and "off")

    """

    # "the larger of the ON and OFF peak responses was taken to be the maximum
    # firing rate of the cluster"
    peaks = find_peak_responses(rfs)

    # "the trial-averaged mean firing rates within the peak bins were then used
    # to estimate the sizes of the ON and OFF subfields...we interpolated each
    # subfield using a 2D bilinear interpolation."
    peaks_interp = interpolate_rfs(peaks, bin_scale=bin_scale)

    # "All pixels in the interpolated grids that were <35% of the cluster's
    # maximum firing rate were set to zero and a contiguous non-zero set of
    # pixels, including the peak pixel, were isolated"
    contig_pixels = find_contiguous_pixels(peaks_interp, threshold=threshold)

    return contig_pixels


def plot_rf_distributions(rf_areas, plot_type='box'):
    """

    Parameters
    ----------
    rf_areas : dict
    plot_type : str, optional
        'box' | 'hist'

    Returns
    -------
    matplotlib.figure.Figure
        figure handle

    """

    # put results into dataframe for easier plotting
    results = []
    for sub_type, areas in rf_areas.items():
        for i, area in enumerate(areas):
            results.append(pd.DataFrame({
                'cluster_id': i,
                'area': area,
                'Subfield': sub_type.upper()}, index=[0]))
    results_pd = pd.concat(results, ignore_index=True)
    # leave out non-responsive clusters
    data_queried = results_pd[results_pd.area != 0]

    if plot_type == 'box':

        splt = sns.catplot(
            x='Subfield', y='area', kind='box', data=data_queried)
        splt.fig.axes[0].set_yscale('log')
        splt.fig.axes[0].set_ylabel('RF Area (pixels$^2$)')
        splt.fig.axes[0].set_ylim([1e-1, 1e4])

    elif plot_type == 'hist':
        bins = 10 ** (np.arange(-1, 4.25, 0.25))
        xmin = 1e-1
        xmax = 1e4
        ymin = 1e0
        ymax = 1e3
        splt = plt.figure(figsize=(12, 4))

        plt.subplot(121)
        plt.hist(data_queried[data_queried.Subfield == 'ON']['area'], bins=bins, log=True)
        plt.xlabel('RF Area (pixels)')
        plt.xscale('log')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.ylabel('Cluster count')
        plt.title('ON Subfield')

        plt.subplot(122)
        plt.hist(data_queried[data_queried.Subfield == 'OFF']['area'], bins=bins, log=True)
        plt.xlabel('RF Area (pixels)')
        plt.xscale('log')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.title('OFF Subfield')

    plt.show()

    return splt


def plot_rfs_by_depth_wrapper(
        alf_path, axes=None, cluster_ids=[], method='corr', binsize=0.025, lags=8, n_depths=30,
        use_svd=False):
    """
    Wrapper function to load spikes and rf stimulus info, aggregate clusters over depths, compute
    rfs, and plot spatial components as a function of linear depth on probe. Must have ibllib
    package in python path in order to use alf loaders.

    Parameters
    ----------
    alf_path : str
        absolute path to experimental session directory
    axes : array-like object of matplotlib axes or `NoneType`, optional
        axes in which to plot the rfs; if `NoneType`, a figure with appropriate axes is created
    cluster_ids : array-like, optional
        clusters to use in rf calculation; if empty, all clusters are used
    method : str, optional
        method for calculating receptive fields
        'sta': method used in Durand et al 2016
        'corr': reverse correlation method; uses convolution and is therefor faster than `'sta'`
    binsize : float, optional
        width of bins in seconds for rf computation
    lags : int, optional
        number of bins after pixel flip to use for rf computation
    n_depths : int, optional
        number of bins to divide probe depth into for aggregating clusters
    use_svd : bool, optional
        `True` plots 1st spatial SVD component of rf; `False` plots time lag with peak response

    Returns
    -------
    dict
        depths and associated receptive fields

    """

    import alf.io as ioalf

    # load objects
    spikes = ioalf.load_object(alf_path, 'spikes')
    clusters = ioalf.load_object(alf_path, 'clusters')
    rfmap = ioalf.load_object(alf_path, '_iblcertif_.rfmap')
    rf_stim_times = rfmap['rfmap.times.00']
    rf_stim = rfmap['rfmap.stims.00'].astype('float')

    # combine clusters across similar depths
    min_depth = np.min(clusters['depths'])
    max_depth = np.max(clusters['depths'])
    depth_limits = np.linspace(min_depth - 1, max_depth, n_depths + 1)
    if len(cluster_ids) == 0:
        times_agg = spikes.times
        depths_agg = spikes.depths
    else:
        clust_mask = np.isin(spikes.clusters, np.array(cluster_ids))
        times_agg = spikes.times[clust_mask]
        depths_agg = spikes.depths[clust_mask]
    clusters_agg = np.full(times_agg.shape, fill_value=np.nan)
    for d in range(n_depths):
        lo_limit = depth_limits[d]
        up_limit = depth_limits[d + 1]
        clusters_agg[(lo_limit < depths_agg) & (depths_agg <= up_limit)] = d

    print('computing receptive fields...', end='')
    if method == 'sta':
        # method in Durand et al 2016
        rfs = compute_rfs(
            times_agg, clusters_agg, rf_stim_times, rf_stim, lags=lags, binsize=binsize)
    elif method == 'corr':
        # reverse correlation method
        rfs = compute_rfs_corr(
            times_agg, clusters_agg, rf_stim_times, rf_stim, lags=lags, binsize=binsize)
    else:
        raise NotImplementedError

    # get single spatial footprint of rf
    if use_svd:
        rfs_spatial = compute_rf_svds(rfs, scale='spatial')['spatial']
    else:
        rfs_spatial = find_peak_responses(rfs)
    rfs_interp = interpolate_rfs(rfs_spatial, bin_scale=0.5)
    print('done')

    plot_rfs_by_depth(rfs_interp, axes=axes)

    return {'depths': depths_agg, 'rfs': rfs_interp}


def plot_rfs_by_depth(rfs, axes=None):
    """

    Parameters
    ----------
    rfs : dict
        dict of "on" and "off" rfs (values are lists); each rf is of shape `(ypix, xpix)`
    axes : array of matplotlib axes or NoneType, optional
        matplotlib axes to plot into; if `NoneType`, a figure will be created and returned

    Returns
    -------
    matplotlib.figure.Figure
        matplotlib figure handle if `axes=None`

    """

    if axes is None:
        fig, axes = plt.subplots(1, len(rfs.keys()), figsize=(3, 12))

    for i, sub_type in enumerate(rfs.keys()):
        rf_array = np.vstack(rfs[sub_type])
        n_rfs = len(rfs[sub_type])
        ypix, xpix = rfs[sub_type][0].shape
        vmin = np.min(rf_array)
        vmax = np.max(rf_array)
        imshow_kwargs = {'vmin': vmin, 'vmax': vmax, 'aspect': 'auto', 'cmap': 'Greys_r'}
        axes[i].imshow(rf_array, **imshow_kwargs)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title('%s subfield' % sub_type.upper())
        for n in range(1, n_rfs):
            axes[i].axhline(n * ypix - 1, 0, xpix, color=[0, 0, 0], linewidth=0.5)
    plt.tight_layout()

    if axes is None:
        return fig


if __name__ == '__main__':

    from pathlib import Path
    from oneibl.one import ONE
    import alf.io as ioalf

    # user options
    BINSIZE = 0.05  # sec
    LAGS = 4  # number of bins for calculating RF
    METHOD = 'corr'  # 'corr' | 'sta'

    # get the data from flatiron and the current folder
    one = ONE()
    eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)
    D = one.load(eid[0], clobber=False, download_only=True)
    session_path = Path(D.local_path[0]).parent

    # load objects
    spikes = ioalf.load_object(session_path, 'spikes')
    rfmap = ioalf.load_object(session_path, '_iblcertif_.rfmap')
    rf_stim_times = rfmap['rfmap.times.00']
    rf_stim = rfmap['rfmap.stims.00'].astype('float')

    # compute receptive fields
    if METHOD == 'sta':
        # method in Durand et al 2016; ~9 min for 700 units on a single cpu core
        print('computing receptive fields...', end='')
        rfs = compute_rfs(
            spikes.times, spikes.clusters, rf_stim_times, rf_stim, lags=LAGS, binsize=BINSIZE)
        print('done')
    elif METHOD == 'corr':
        # reverse correlation method; ~3 min for 700 units on a single cpu core
        print('computing receptive fields...', end='')
        rfs = compute_rfs_corr(
            spikes.times, spikes.clusters, rf_stim_times, rf_stim, lags=LAGS, binsize=BINSIZE)
        print('done')
    else:
        raise NotImplementedError('The %s method to compute rfs is not implemented' % METHOD)

    print('computing receptive field areas...', end='')
    rf_areas = compute_rf_areas(rfs)
    print('done')

    fig = plot_rf_distributions(rf_areas, plot_type='hist')
