import numpy as np
import matplotlib.pyplot as plt
import os
import alf.io as ioalf
from brainbox.processing import bincount2D
from brainbox.singlecell import calculate_peths
try:
    from responsive import are_neurons_responsive
except:
    from v1_protocol.responsive import are_neurons_responsive

def bin_responses(spike_times, spike_clusters, stim_times, stim_values, output_fr=True):
    """
    Compute firing rates during grating presentation

    :param spike_times: array of spike times
    :type spike_times: array-like
    :param spike_clusters: array of cluster ids associated with each entry in `spike_times`
    :type spike_clusters: array-like
    :param stim_times: stimulus presentation times; array of size (M, 2) where M is the number of
        stimuli; column 0 is stim onset, column 1 is stim offset
    :type stim_times: array-like
    :param stim_values: grating orientations in radians
    :type stim_values: array-like
    :param output_fr: True to output firing rate, False to output spike count over stim presentation
    :type: bool
    :return: number of spikes for each clusterduring stimulus presentation
    :rtype: array of shape `(n_clusters, n_stims, n_stim_reps)`
    """
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)
    stim_ids = np.unique(stim_values)
    n_stims = len(stim_ids)
    n_reps = len(np.where(stim_values == stim_values[0])[0])
    responses = np.zeros(shape=(n_clusters, n_stims, n_reps))
    stim_reps = np.zeros(shape=n_stims, dtype=np.int)
    for stim_time, stim_val in zip(stim_times, stim_values):
        i_stim = np.where(stim_ids == stim_val)[0][0]
        i_rep = stim_reps[i_stim]
        stim_reps[i_stim] += 1
        # filter spikes
        idxs = (spike_times > stim_time[0]) & (spike_times <= stim_time[1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]
        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        bin_size = np.diff(stim_time)[0]
        xscale = stim_time
        xind = (np.floor((i_spikes - stim_time[0]) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)
        # store
        bs_idxs = np.isin(cluster_ids, yscale)
        scale = bin_size if output_fr else 1.0
        responses[bs_idxs, i_stim, i_rep] = r[:, 0] / scale
    return responses


def compute_selectivity(means, thetas, measure):
    """
    Compute direction or orientation selectivity index measure, as well as preferred direction/ori
    OSI/DSI calculated as in Mazurek et al 2014:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4123790/

    :param means: mean responses over multiple stimulus presentations; array of shape
        `(n_clusters, n_stims)`
    :type means: array-like
    :param thetas: grating direction in radians (corresponds to dimension 1 of `means`)
    :type thetas: array-like
    :param measure: 'ori' | 'dir' - compute orientation or direction selectivity
    :type measure: str
    :return: tuple (index, preference)
    """
    if measure == 'dir' or measure == 'direction':
        factor = 1
    elif measure == 'ori' or measure == 'orientation':
        factor = 2
    else:
        raise ValueError('"%s" is an invalid measure; must be "dir" or "ori"' % measure)
    vector_norm = means / np.sum(means, axis=1, keepdims=True)
    vector_sum = np.sum(vector_norm * np.exp(factor * 1.j * thetas), axis=1)
    index = np.abs(vector_sum)
    preference = np.angle(vector_sum) / factor
    return index, preference


def scatterplot(xs, ys, xlabel, ylabel, id_line=False, linewidth=1, ax=None):
    """
    General scatterplot function

    :param xs:
    :type xs: array-like
    :param ys:
    :type ys: array-like
    :param xlabel:
    :param ylabel:
    :param id_line: boolean, whether or not to plot identity line
    :param linewidth:
    :param ax:
    :return: figure handle or Axes object
    """
    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        return_fig = True
    if id_line:
        lmin = np.nanmin([np.nanquantile(xs, 0.01), np.nanquantile(ys, 0.01)])
        lmax = np.nanmax([np.nanquantile(xs, 0.99), np.nanquantile(ys, 0.99)])
        ax.plot([lmin, lmax], [lmin, lmax], '-', color=[0.7, 0.7, 0.7], linewidth=linewidth)
    ax.scatter(xs, ys, marker='.', s=150, edgecolors=[1, 1, 1], alpha=1.0, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if return_fig:
        plt.show()
        return fig
    else:
        return ax


def plot_cdfs(prefs, xlabel, ax=None):
    """

    :param prefs: dict with keys `beg` and `end`, each of which is an array of values
    :param xlabel:
    :param ax:
    :return:
    """
    from scipy.stats import ks_2samp
    _, p = ks_2samp(prefs['beg'], prefs['end'])
    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        return_fig = True
    for epoch in list(prefs.keys()):
        vals = prefs[epoch]
        ax.hist(
            vals[~np.isnan(vals)], bins=20, histtype='step', density=True, cumulative=True,
            linewidth=2, label='%s epoch' % epoch.capitalize())
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability')
    ax.set_title('KS p-value = %1.2e' % p)
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if return_fig:
        plt.show()
        return fig
    else:
        return ax


def plot_value_by_depth(values, depths, xlabel, ylabel, window=25, linewidth=1, ax=None):
    """
    Plot a given value by depth along with a line indicating the running average of the value if
    `window` is greater than 0.

    :param values: array-like
    :param depths: array-like
    :param xlabel:
    :param ylabel:
    :param window: window for running average
    :param linewidth:
    :param ax:
    :return:
    """
    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))
        return_fig = True

    if window > 0:
        x = np.linspace(np.min(depths), np.max(depths), 100)
        f = np.interp(x, xp=depths[~np.isnan(values)], fp=values[~np.isnan(values)])
        running_average = np.convolve(f, np.ones(window) / window, mode='same')
    else:
        x = None
        running_average = None

    ax.scatter(values, depths, marker='.', c=[[0.1, 0.1, 0.1]], s=5)
    # plot center line at 1
    ax.axvline(x=1, ymin=0.02, ymax=0.98, color=[0.7, 0.7, 0.7], linewidth=linewidth)
    # plot running average
    if window > 0:
        ax.plot(running_average, x, 'r', linewidth=linewidth)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if return_fig:
        plt.show()
        return fig
    else:
        return ax


def plot_histograms_by_epoch_depth(locations, values, val_label=None, axes=None):
    """

    :param locations: array-like
    :param values: dict of values corresponding to `locations`; values will be plotted for the first
        two dict keys
    :param val_label:
    :param axes:
    :return:
    """
    return_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(6, 6))
        return_fig = True

    bar_thickness = 0.95 * np.mean(np.diff(locations))
    epochs = values.keys()
    xmax = np.max(np.array([np.nanmax(values[epoch]) for epoch in epochs]))
    for i, epoch in enumerate(epochs):
        axes[i].barh(locations, values[epoch], height=bar_thickness, facecolor='k')
        axes[i].set_title('%s epoch' % epoch.capitalize())
        axes[i].set_xlabel(val_label)
        axes[i].set_xlim(0, xmax)
        axes[i].invert_yaxis()
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if i == 0:
            axes[i].set_ylabel('Depth (mm)')
        else:
            axes[i].set_yticks([])
    if return_fig:
        plt.tight_layout()
        plt.show()
        return fig
    else:
        return axes


def plot_average_psths(means, stds, on_idx, off_idx, bin_size, ax=None, tick_freq=1):

    from matplotlib.ticker import FuncFormatter, FixedLocator

    return_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        return_fig = True

    n_bins = len(means['beg'])

    if tick_freq is None:
        tick_freq = 0.25  # seconds (10 * bin_size)
    tick_locs = [on_idx]
    bins_per_tick = int(tick_freq / bin_size)
    pre_time = on_idx * bin_size
    post_time = (n_bins - on_idx) * bin_size
    for i in range(1, int(np.floor(pre_time / tick_freq)) + 1):
        tick_locs.append(on_idx - i * bins_per_tick)
    for i in range(1, int(np.floor(post_time / tick_freq)) + 1):
        tick_locs.append(on_idx + i * bins_per_tick)
    xtick_locs = FixedLocator(tick_locs)
    if tick_freq > 1:
        xtick_labs = FuncFormatter(lambda x, p: '%i' % ((x - on_idx) * bin_size))
    elif tick_freq > 0.25:
        xtick_labs = FuncFormatter(lambda x, p: '%1.1f' % ((x - on_idx) * bin_size))
    else:
        xtick_labs = FuncFormatter(lambda x, p: '%1.2f' % ((x - on_idx) * bin_size))

    epochs = means.keys()
    for epoch in epochs:
        m = means[epoch]
        s = stds[epoch]
        ax.fill_between(np.arange(n_bins), m - s, m + s, alpha=0.5)
        ax.plot(np.arange(n_bins), m, label='%s epoch' % epoch.capitalize())
    ax.axvline(x=on_idx, ymin=0.02, ymax=0.98, color='k', linestyle='--')
    ax.axvline(x=off_idx, ymin=0.02, ymax=0.98, color='k', linestyle='--')

    ax.get_xaxis().set_major_locator(xtick_locs)
    ax.get_xaxis().set_major_formatter(xtick_labs)
    ax.set_xlabel('Time from stim onset (s)')
    ax.set_ylabel('Population rate (Hz)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', frameon=False)

    if return_fig:
        plt.show()
        return fig
    else:
        return ax


def plot_polar_psth_and_rasters(
        mean_responses, binned_responses, osi, grating_vals, on_idx, off_idx, bin_size,
        tick_freq=1, cluster=None, grid_spec=None, fig=None):

    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter, FixedLocator

    n_trials, n_clusters, n_bins = binned_responses['beg'][0].shape
    n_rows = 2
    n_cols = 2
    if grid_spec is None:
        fig = plt.figure(figsize=(2 * n_cols, 2.5 * n_rows))
        gs0 = gridspec.GridSpec(
            n_rows, n_cols + 1, height_ratios=[1, 3], width_ratios=[0.1, 4, 4])
    else:
        assert fig is not None
        gs0 = grid_spec
    gs = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # ticks
    if tick_freq is None:
        tick_freq = 0.25  # seconds (10 * bin_size)
    tick_locs = [on_idx]
    bins_per_tick = int(tick_freq / bin_size)
    pre_time = on_idx * bin_size
    post_time = (n_bins - on_idx) * bin_size
    for i in range(1, int(np.floor(pre_time / tick_freq)) + 1):
        tick_locs.append(on_idx - i * bins_per_tick)
    for i in range(1, int(np.floor(post_time / tick_freq)) + 1):
        tick_locs.append(on_idx + i * bins_per_tick)
    xtick_locs = FixedLocator(tick_locs)
    xtick_labs = FuncFormatter(lambda x, p: '%i' % ((x - on_idx) * bin_size))

    for i, epoch in enumerate(mean_responses.keys()):

        stim_ids = np.unique(grating_vals[epoch])
        n_stims = len(stim_ids)

        # plot polar
        gs[0][i] = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs0[0, i + 1], hspace=0.0, height_ratios=[0, 1, 0])
        ax0 = fig.add_subplot(gs[0][i][1], projection='polar')
        if grid_spec is None:
            # making new fig; cluster number will be in plot title
            title_str = str(
                '%s OSI = %1.2f\n' % (epoch.capitalize(), osi[epoch]))
        else:
            title_str = str(
                '\n Cluster %i\n%s OSI = %1.2f\n \n' % (cluster, epoch.capitalize(), osi[epoch]))
        ax0.set_title(title_str)

        m = mean_responses[epoch]
        ax0.plot(np.concatenate([stim_ids, [stim_ids[0]]]), np.concatenate([m, [m[0]]]))

        # plot rasters
        gs[1][i] = gridspec.GridSpecFromSubplotSpec(
            n_stims, 1, subplot_spec=gs0[1, i + 1], hspace=0.0)

        for j, stim_id in enumerate(stim_ids):
            ax1 = fig.add_subplot(gs[1][i][j])
            ax1.imshow(
                binned_responses[epoch][j][:, 0, :], cmap='Greys', origin='lower', aspect='auto')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.axvline(x=on_idx, ymin=0, ymax=n_trials, color='r')
            ax1.axvline(x=off_idx, ymin=0.02, ymax=0.98, linestyle='-', color='r')
            ax1.get_xaxis().set_major_locator(xtick_locs)
            ax1.get_xaxis().set_major_formatter(xtick_labs)
            if j == n_stims - 1:
                ax1.set_xlabel('Time (s)')
            else:
                ax1.set_xticklabels([])
                ax1.set_xlabel('')
                # make stim delineators thicker
                ax1.spines['bottom'].set_visible(False)
                ax1.axhline(y=0, xmin=0, xmax=1, linestyle='-', color='k')
            if epoch == 'beg':
                ax1.set_ylabel('%i' % int(stim_id * 180 / np.pi), rotation=0, labelpad=10)
                ax1.set_yticks([])
            else:
                ax1.set_yticks([])
                ax1.set_ylabel('')

        # ylabel needs its own plot
        if epoch == 'beg':
            gs_label = gridspec.GridSpecFromSubplotSpec(
                1, 1, subplot_spec=gs0[1, 0], hspace=0.0, wspace=0.0)
            ax = fig.add_subplot(gs_label[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel('Orientation (deg)')

    if grid_spec is None:
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle('Cluster %i' % cluster, x=0.57)
        plt.show()
    else:
        pass

    return fig


def plot_grating_figures(
    session_path, cluster_ids_summary, cluster_ids_selected, save_dir=None, format='png',
        pre_time=0.5, post_time=2.5, bin_size=0.005, smoothing=0.025, n_rand_clusters=20,
        plot_summary=True, plot_selected=True):
    """
    Produces two summary figures for the oriented grating protocol; the first summary figure
    contains plots that compare different measures during the first and second grating protocols,
    such as orientation selectivity index (OSI), orientation preference, fraction of visual
    clusters, PSTHs, firing rate histograms, etc. The second summary figure contains plots of polar
    PSTHs and corresponding rasters for a random subset of visually responsive clusters.

    Parameters
    ----------
    session_path : str
        absolute path to experimental session directory
    cluster_ids_summary : list
        the clusters for which to plot summary psths/rasters; if empty, all clusters with responses
        during the grating presentations are used
    cluster_ids_selected : list
        the clusters for which to plot individual psths/rasters; if empty, `n_rand_clusters` are
        randomly chosen
    save_dir : str or NoneType
        if NoneType, figures are displayed; else a string defining the absolute filepath to the
        directory in which figures will be saved
    format : str
        file format, i.e. 'png' | 'pdf' | 'jpg'
    pre_time : float
        time (sec) to plot before grating presentation onset
    post_time : float
        time (sec) to plot after grating presentation onset (should include length of stimulus)
    bin_size : float
        size of bins for raster plots/psths
    smoothing : float
        size of smoothing kernel (sec)
    n_rand_clusters : int
        the number of random clusters to choose for which to plot psths/rasters if
        `cluster_ids_slected` is empty
    plot_summary : bool
        a flag for plotting the summary figure
    plot_selected : bool
        a flag for plotting the selected units figure
        
    Returns
    -------
    metrics : dict
        - 'osi' (dict): keys 'beg', 'end' point to arrays of osis during these epochs
        - 'orientation_pref' (dict): keys 'beg', 'end' point to arrays of orientation preference
        - 'frac_resp_by_depth' (dict): fraction of responsive clusters by depth
    
    fig_dict : dict
        A dict whose values are handles to one or both figures generated.
    """

    fig_dict = {}
    cluster_ids = cluster_ids_summary
    cluster_idxs = cluster_ids_selected
    epochs = ['beg', 'end']
    
    # -------------------------
    # load required alf objects
    # -------------------------  
    print('loading alf objects...', end='', flush=True)
    spikes = ioalf.load_object(session_path, 'spikes')
    clusters = ioalf.load_object(session_path, 'clusters')
    gratings = ioalf.load_object(session_path, '_iblcertif_.odsgratings')
    spontaneous = ioalf.load_object(session_path, '_iblcertif_.spontaneous')
    grating_times = {
        'beg': gratings['odsgratings.times.00'],
        'end': gratings['odsgratings.times.01']}
    grating_vals = {
        'beg': gratings['odsgratings.stims.00'],
        'end': gratings['odsgratings.stims.01']}
    spont_times = {
        'beg': spontaneous['spontaneous.times.00'],
        'end': spontaneous['spontaneous.times.01']}

    # --------------------------
    # calculate relevant metrics
    # --------------------------
    print('calcuating mean responses to gratings...', end='', flush=True)
    # calculate mean responses to gratings
    mask_clust = np.isin(spikes.clusters, cluster_ids)  # update mask for responsive clusters
    mask_times = np.full(spikes.times.shape, fill_value=False)
    for epoch in epochs:
        mask_times |= (spikes.times >= grating_times[epoch].min()) & \
                      (spikes.times <= grating_times[epoch].max())
    resp = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        resp[epoch] = are_neurons_responsive(
            spikes.times[mask_clust], spikes.clusters[mask_clust], grating_times[epoch],
            grating_vals[epoch], spont_times[epoch])
    responses = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        responses[epoch] = bin_responses(
            spikes.times[mask_clust], spikes.clusters[mask_clust], grating_times[epoch],
            grating_vals[epoch])
    responses_mean = {epoch: np.mean(responses[epoch], axis=2) for epoch in epochs}
    # responses_se = {epoch: np.std(responses[epoch], axis=2) / np.sqrt(responses[epoch].shape[2])
    #                 for epoch in responses.keys()}
    print('done')

    # calculate osi and orientation preference
    print('calcuating osi/orientation preference...', end='', flush=True)
    ori_pref = {epoch: [] for epoch in epochs}
    osi = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        osi[epoch], ori_pref[epoch] = compute_selectivity(
            responses_mean[epoch], np.unique(grating_vals[epoch]), 'ori')
    print('done')

    # calculate depth vs osi ratio (osi_beg/osi_end)
    print('calcuating osi ratio as a function of depth...', end='', flush=True)
    depths = np.array([clusters.depths[c] for c in cluster_ids])
    ratios = np.array([osi['beg'][c] / osi['end'][c] for c in range(len(cluster_ids))])
    print('done')

    # calculate fraction of visual neurons by depth
    print('calcuating fraction of visual clusters by depth...', end='', flush=True)
    n_bins = 10
    min_depth = np.min(clusters['depths'])
    max_depth = np.max(clusters['depths'])
    depth_limits = np.linspace(min_depth - 1, max_depth, n_bins + 1)
    depth_avg = (depth_limits[:-1] + depth_limits[1:]) / 2
    # aggregate clusters
    clusters_binned = {epoch: [] for epoch in epochs}
    frac_responsive = {epoch: [] for epoch in epochs}
    cids = cluster_ids
    for epoch in epochs:
        # just look at responsive clusters during this epoch
        cids_tmp = cids[resp[epoch]]
        for d in range(n_bins):
            lo_limit = depth_limits[d]
            up_limit = depth_limits[d + 1]
            # clusters.depth index is cluster id
            cids_curr_depth = np.where(
                (lo_limit < clusters.depths) & (clusters.depths <= up_limit))[0]
            clusters_binned[epoch].append(cids_curr_depth)
            frac_responsive[epoch].append(np.sum(
                np.isin(cids_tmp, cids_curr_depth)) / len(cids_curr_depth))
    # package for plotting
    responsive = {'fraction': frac_responsive, 'depth': depth_avg}
    print('done')

    # calculate PSTH averaged over all clusters/orientations
    print('calcuating average PSTH...', end='', flush=True)
    peths = {epoch: [] for epoch in epochs}
    peths_avg = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        stim_ids = np.unique(grating_vals[epoch])
        peths[epoch] = {i: None for i in range(len(stim_ids))}
        peths_avg_tmp = []
        for i, stim_id in enumerate(stim_ids):
            curr_stim_idxs = np.where(grating_vals[epoch] == stim_id)
            align_times = grating_times[epoch][curr_stim_idxs, 0][0]
            peths[epoch][i], _ = calculate_peths(
                spikes.times[mask_times], spikes.clusters[mask_times], cluster_ids,
                align_times, pre_time=pre_time, post_time=post_time, bin_size=bin_size,
                smoothing=smoothing, return_fr=True)
            peths_avg_tmp.append(
                np.mean(peths[epoch][i]['means'], axis=0, keepdims=True))
        peths_avg_tmp = np.vstack(peths_avg_tmp)
        peths_avg[epoch] = {
            'mean': np.mean(peths_avg_tmp, axis=0),
            'std': np.std(peths_avg_tmp, axis=0) / np.sqrt(peths_avg_tmp.shape[0])}
    peths_avg['bin_size'] = bin_size
    peths_avg['on_idx'] = int(pre_time / bin_size)
    peths_avg['off_idx'] = peths_avg['on_idx'] + int(2 / bin_size)
    print('done')
    
    # compute rasters for entire orientation sequence at beg/end epoch
    if plot_summary:
        print('computing rasters for example stimulus sequences...', end='', flush=True)
        r = {epoch: None for epoch in epochs}
        r_times = {epoch: None for epoch in epochs}
        r_clusters = {epoch: None for epoch in epochs}
        for epoch in epochs:
            # restrict activity to a single stim series; assumes each possible grating direction
            # is displayed before repeating
            n_stims = len(np.unique(grating_vals[epoch]))
            mask_idxs_e = (spikes.times >= grating_times[epoch][:n_stims].min()) & \
                          (spikes.times <= grating_times[epoch][:n_stims].max())
            r_tmp, r_times[epoch], r_clusters[epoch] = bincount2D(
                spikes.times[mask_idxs_e], spikes.clusters[mask_idxs_e], bin_size)
            # order activity by anatomical depth of neurons
            d = dict(zip(spikes.clusters[mask_idxs_e], spikes.depths[mask_idxs_e]))
            y = sorted([[i, d[i]] for i in d])
            isort = np.argsort([x[1] for x in y])
            r[epoch] = r_tmp[isort, :]
        # package for plotting
        rasters = {'spikes': r, 'times': r_times, 'clusters': r_clusters, 'bin_size': bin_size}
        print('done')

    # -------------------------------------------------
    # compute psths and rasters for individual clusters
    # -------------------------------------------------
    if plot_selected:
        print('computing psths and rasters for clusters...', end='', flush=True)
        if len(cluster_ids_selected) == 0:
            if (n_rand_clusters < len(cluster_ids)):
                cluster_idxs = np.random.choice(cluster_ids, size=n_rand_clusters, replace=False)
            else:
                cluster_idxs = cluster_ids
        else:
            cluster_idxs = cluster_ids_selected
        mean_responses = {cluster: {epoch: [] for epoch in epochs} for cluster in cluster_idxs}
        osis = {cluster: {epoch: [] for epoch in epochs} for cluster in cluster_idxs}
        binned = {cluster: {epoch: [] for epoch in epochs} for cluster in cluster_idxs}
        for cluster_idx in cluster_idxs:
            cluster = np.where(cluster_ids == cluster_idx)[0]
            for epoch in epochs:
                mean_responses[cluster_idx][epoch] = responses_mean[epoch][cluster, :][0]
                osis[cluster_idx][epoch] = osi[epoch][cluster]
                stim_ids = np.unique(grating_vals[epoch])
                binned[cluster_idx][epoch] = {j: None for j in range(len(stim_ids))}
                for j, stim_id in enumerate(stim_ids):
                    curr_stim_idxs = np.where(grating_vals[epoch] == stim_id)
                    align_times = grating_times[epoch][curr_stim_idxs, 0][0]
                    _, binned[cluster_idx][epoch][j] = calculate_peths(
                        spikes.times[mask_times], spikes.clusters[mask_times], [cluster_idx],
                        align_times, pre_time=pre_time, post_time=post_time, bin_size=bin_size)
        print('done')

    # --------------
    # output figures
    # --------------
    print('producing figures...', end='')
    if plot_summary:
        if save_dir is None:
            save_file = None
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file = os.path.join(save_dir, 'grating_summary_figure.' + format)
        fig_gr_summary = plot_summary_figure(
            ratios=ratios, depths=depths, responsive=responsive, peths_avg=peths_avg, osi=osi,
            ori_pref=ori_pref, responses_mean=responses_mean, rasters=rasters, save_file=save_file)
        fig_gr_summary.suptitle('Summary Grating Responses')
        fig_dict['fig_gr_summary'] = fig_gr_summary

    if plot_selected:
        if save_dir is None:
            save_file = None
        else:
            save_file = os.path.join(save_dir, 'grating_random_responses.' + format)
        fig_gr_selected = plot_psths_and_rasters(
            mean_responses, binned, osis, grating_vals, on_idx=peths_avg['on_idx'],
            off_idx=peths_avg['off_idx'], bin_size=bin_size, save_file=save_file)
        fig_gr_selected.suptitle('Selected Units Grating Responses')
        print('done')
        fig_dict['fig_gr_selected'] = fig_gr_selected
    
    # -----------------------------
    # package up and return metrics
    # -----------------------------
    metrics = {
        'osi': osi,
        'orientation_pref': ori_pref,
        'frac_resp_by_depth': responsive,
    }
    return fig_dict, metrics

def plot_summary_figure(
        depths, ratios, responsive, peths_avg, osi, ori_pref, responses_mean, rasters,
        save_file=None):
    """
    Produce summary figure for responses to orientated gratings. See code in calling function
    `plot_grating_figures` to see how these inputs are created.

    :param depths:
    :param ratios:
    :param responsive:
    :param peths_avg:
    :param osi:
    :param ori_pref:
    :param responses_mean:
    :param rasters:
    :param save_file:
    :return fig:
    """

    import seaborn as sns
    import matplotlib.gridspec as gridspec

    sns.set_context('paper')
    epochs = osi.keys()

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 3])

    # right side
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.5)

    # ---------------------------
    # right side, top row
    # ---------------------------
    gs1a = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs1[0], wspace=0.5, width_ratios=[1, 0.5, 0.5, 1.5])

    # osi ratio as a function of depth
    ax = fig.add_subplot(gs1a[0])
    ax = plot_value_by_depth(
        ratios, depths, xlabel='OSI ratio (beg/end)', ylabel='Depth (mm)', window=25,
        linewidth=2, ax=ax)

    # fraction of visual clusters by depth
    ax0 = fig.add_subplot(gs1a[1])
    ax1 = fig.add_subplot(gs1a[2])
    axes = plot_histograms_by_epoch_depth(
        responsive['depth'], responsive['fraction'], val_label='Fraction of\nvisual clusters',
        axes=[ax0, ax1])

    # average psth
    ax = fig.add_subplot(gs1a[3])
    ax = plot_average_psths(
        {epoch: peths_avg[epoch]['mean'] for epoch in epochs},
        {epoch: peths_avg[epoch]['std'] for epoch in epochs},
        on_idx=peths_avg['on_idx'], off_idx=peths_avg['off_idx'], bin_size=peths_avg['bin_size'],
        tick_freq=1, ax=ax)

    # ---------------------------
    # right side, bottom row
    # ---------------------------
    gs1b = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs1[1], wspace=0.5, width_ratios=[1, 1, 1])

    # osi end vs beg
    ax = fig.add_subplot(gs1b[0])
    ax = scatterplot(
        osi['beg'], osi['end'], 'OSI (beg epoch)', 'OSI (end epoch)', id_line=True,
        linewidth=2, ax=ax)

    # ori pref end vs beg
    ax = fig.add_subplot(gs1b[1])
    ax = scatterplot(
        ori_pref['beg'], ori_pref['end'], 'Ori Pref (beg epoch)', 'Ori Pref (end epoch)',
        id_line=True, linewidth=2, ax=ax)

    # ori pref cdf
    ax = fig.add_subplot(gs1b[2])
    ax = plot_cdfs(ori_pref, 'Orientation preference', ax=ax)

    # ---------------------------
    # left side, top row
    # ---------------------------
    gs0 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0], wspace=0.1, hspace=0.3,
        width_ratios=[1, 1], height_ratios=[3, 1])

    # plot binned spikes
    for i, epoch in enumerate(epochs):
        ax = fig.add_subplot(gs0[0, i])
        ax.imshow(
            rasters['spikes'][epoch], aspect='auto', cmap='binary',
            vmax=rasters['bin_size'] / 0.001 / 4, origin='upper',
            extent=np.r_[rasters['times'][epoch][[0, -1]], rasters['clusters'][epoch][[0, -1]]])
        ax.set_title('%s epoch\nFirst trial sequence' % epoch.capitalize())
        ax.set_xlabel('Time (s)')
        if ax.is_first_col():
            ax.set_ylabel('Depth')
        else:
            ax.set_yticks([])

    # plot histogram of firing rates
    for i, epoch in enumerate(epochs):
        ax = fig.add_subplot(gs0[1, i])
        means_tmp = np.mean(responses_mean[epoch], axis=1)
        # get rid of outliers
        means_tmp = means_tmp[means_tmp < np.quantile(means_tmp, 0.95)]
        ax.hist(means_tmp, bins=20, facecolor='k')
        ax.set_title('%s firing rates' % epoch.capitalize())
        ax.set_xlabel('Firing rate (Hz)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.is_first_col():
            ax.set_ylabel('Count')

    plt.tight_layout()
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=300)
    
    return fig


def plot_psths_and_rasters(
        mean_responses, binned_spikes, osis, grating_vals, on_idx, off_idx, bin_size,
        save_file=None):

    import matplotlib.gridspec as gridspec
    import seaborn as sns
    sns.set_context('paper')

    fig = plt.figure(figsize=(15, 16))
    cluster_idxs = mean_responses.keys()
    n_clusters = len(cluster_idxs)
    n_cols = int(np.ceil(np.sqrt(n_clusters)))
    n_rows = int(np.ceil(n_clusters / n_cols))
    gs = fig.add_gridspec(n_rows, n_cols)
    for i, cluster_idx in enumerate(cluster_idxs):
        r = int(np.floor(i / n_cols))
        c = int(i % n_cols)
        gs0 = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs[r, c], height_ratios=[1, 3], width_ratios=[0.05, 4, 4])
        plot_polar_psth_and_rasters(
            mean_responses[cluster_idx], binned_spikes[cluster_idx], osis[cluster_idx],
            grating_vals, on_idx, off_idx, bin_size, cluster=cluster_idx, grid_spec=gs0, fig=fig)
    gs.tight_layout(fig)
    plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file, dpi=300)
    return fig


def get_vr_clusters(session_path, clusters=None, n_selected_cl=4):
    '''
    Gets visually responsive clusters
    
    Parameters
    ----------
    session_path : str
        The path to to the appropriate 'alf/probe' directory.
    clusters : ndarray
        The clusters to use to get a subset of visually responsive clusters. (if `None`, take
        visually response subset from all clusters from recording session.)
    n_selected_cl : int
        The number of clusters to return in `vr_clusters_selected`
    
    Returns
    -------
    clusters_vr : ndarray
        The visually responsive clusters.
    clusters_selected_vr : ndarray
        A subset of `n_selected_cl` `vr_clusters`
    '''
    
    print('finding visually responsive clusters...', end='', flush=True)

    # -------------------------
    # load required alf objects
    # -------------------------
    spikes = ioalf.load_object(session_path, 'spikes')
    gratings = ioalf.load_object(session_path, '_iblcertif_.odsgratings')
    spontaneous = ioalf.load_object(session_path, '_iblcertif_.spontaneous')
    grating_times = {
        'beg': gratings['odsgratings.times.00'],
        'end': gratings['odsgratings.times.01']}
    grating_vals = {
        'beg': gratings['odsgratings.stims.00'],
        'end': gratings['odsgratings.stims.01']}
    spont_times = {
        'beg': spontaneous['spontaneous.times.00'],
        'end': spontaneous['spontaneous.times.01']}

    # ---------------------------------
    # find visually responsive clusters
    # ---------------------------------
    epochs = ['beg', 'end']
    if clusters is None:  # use all clusters
        # speed up downstream computations by restricting data to relevant time periods
        mask_times = np.full(spikes.times.shape, fill_value=False)
        for epoch in epochs:
            mask_times |= (spikes.times >= grating_times[epoch].min()) & \
                          (spikes.times <= grating_times[epoch].max())
        clusters = np.unique(spikes.clusters[mask_times])

    # only calculate responsiveness for clusters that were active during gratings
    mask_clust = np.isin(spikes.clusters, clusters)
    resp = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        resp[epoch] = are_neurons_responsive(
            spikes.times[mask_clust], spikes.clusters[mask_clust], grating_times[epoch],
            grating_vals[epoch], spont_times[epoch])
    resp_agg = resp['beg'] & resp['end']
    # remove non-responsive clusters
    clusters_vr = clusters[resp_agg]
    print('done')
    if n_selected_cl < len(clusters_vr):
        clusters_selected_vr = np.random.choice(clusters_vr, size=n_selected_cl, replace=False)
    else:
        clusters_selected_vr = clusters_vr
    return clusters_vr, clusters_selected_vr


if __name__ == '__main__':

    from pathlib import Path
    from oneibl.one import ONE

    # user params for rasters
    PRE_TIME = 0.5     # time (sec) to use before grating onset
    POST_TIME = 2.5    # time (sec) to use after grating onset (grating duration 2 sec)
    BIN_SIZE = 0.005   # sec
    SMOOTHING = 0.025  # width of smoothing kernel (sec)

    # get the data from flatiron and the current folder
    one = ONE()
    eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)
    D = one.load(eid[0], clobber=False, download_only=True)
    session_path = Path(D.local_path[0]).parent

    # example for how to auto output figures to screen; for saving, include the argument `save_dir`
    plot_grating_figures(
        session_path, save_dir=None, pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE,
        smoothing=SMOOTHING, n_rand_clusters=20)
