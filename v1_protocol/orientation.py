import numpy as np
import matplotlib.pyplot as plt


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


def scatterplot(xs, ys, xlabel, ylabel, id_line=False, fontsize=15):
    """
    General scatterplot function

    :param xs:
    :param ys:
    :param xlabel:
    :param ylabel:
    :param id_line: boolean, whether or not to plot identity line
    :param fontsize:
    :return: figure handle
    """
    fig = plt.figure(figsize=(6, 6))
    if id_line:
        lmin = np.nanmin([np.nanquantile(xs, 0.01), np.nanquantile(ys, 0.01)])
        lmax = np.nanmax([np.nanquantile(xs, 0.99), np.nanquantile(ys, 0.99)])
        plt.plot([lmin, lmax], [lmin, lmax], '-', color='k')
    plt.scatter(xs, ys, marker='.', s=150, edgecolors=[1, 1, 1], alpha=1.0)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    return fig


def plot_cdfs(prefs, xlabel, fontsize=15):
    from scipy.stats import ks_2samp
    _, p = ks_2samp(prefs['beg'], prefs['end'])
    fig = plt.figure(figsize=(6, 6))
    for epoch in list(prefs.keys()):
        vals = prefs[epoch]
        plt.hist(
            vals[~np.isnan(vals)], bins=20, histtype='step', density=True, cumulative=True,
            linewidth=2, label=epoch)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('KS p-value = %1.2e' % p)
    plt.legend(fontsize=fontsize, loc='upper left')
    plt.show()
    return fig


def plot_polar_psth_and_rasters(
        mean_responses, binned_responses, osi, grating_vals, on_idx, off_idx, bin_size, tick_freq=1,
        cluster=None):

    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import FuncFormatter, FixedLocator

    n_trials, n_clusters, n_bins = binned_responses['beg'][0].shape
    n_rows = 2
    n_cols = 2
    fig = plt.figure(figsize=(4 * n_cols, 5 * n_rows))
    gs0 = gridspec.GridSpec(n_rows, n_cols + 1, height_ratios=[1, 3], width_ratios=[0.05, 4, 4])
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
    xtick_labs = FuncFormatter(lambda x, p: '%1.2f' % ((x - on_idx) * bin_size))

    for i, epoch in enumerate(mean_responses.keys()):

        stim_ids = np.unique(grating_vals[epoch])
        n_stims = len(stim_ids)

        # plot polar
        gs[0][i] = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=gs0[0, i + 1], hspace=0.0, height_ratios=[0, 1, 0])
        ax0 = fig.add_subplot(gs[0][i][1], projection='polar')
        title_str = str('%s OSI = %1.2f\n' % (epoch.capitalize(), osi[epoch]))
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
            ax1.set_ylabel('%i' % int(stim_id * 180 / np.pi), rotation=0, labelpad=25)
            ax1.set_yticks([])

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
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Cluster %i' % cluster, x=0.57)
    plt.show()

    return fig


if __name__ == '__main__':

    from pathlib import Path
    from oneibl.one import ONE
    import alf.io as ioalf
    from responsive import are_neurons_responsive
    from brainbox.singlecell import calculate_peths

    # user params for rasters
    PRE_TIME = 0.5    # time (sec) to plot before grating onset
    POST_TIME = 2.5   # time (sec) to plot after grating onset (grating duration 2 sec)
    BIN_SIZE = 0.005  # sec

    # get the data from flatiron and the current folder (note: this dataset doesn't work! none do)
    one = ONE()
    eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)
    D = one.load(eid[0], clobber=False, download_only=True)
    session_path = Path(D.local_path[0]).parent

    # load objects
    spikes = ioalf.load_object(session_path, 'spikes')
    gratings = ioalf.load_object(session_path, '_iblcertif_.odsgratings')
    grating_times = {
        'beg': gratings['odsgratings.times.00'],
        'end': gratings['odsgratings.times.01']}
    grating_vals = {
        'beg': gratings['odsgratings.stims.00'],
        'end': gratings['odsgratings.stims.01']}

    # calculate mean responses to gratings
    epochs = ['beg', 'end']
    responses = {epoch: [] for epoch in epochs}
    # speed up computation by only binned relevant spike times; combine spike times across
    # presentations to avoid different cluster indices across grating presentations
    mask_idxs = np.full(spikes.times.shape, fill_value=False)
    for epoch in epochs:
        mask_idxs |= (spikes.times >= grating_times[epoch].min()) & \
                     (spikes.times <= grating_times[epoch].max())
    cluster_ids = np.unique(spikes.clusters[mask_idxs])
    for epoch in epochs:
        responses[epoch] = bin_responses(
            spikes.times[mask_idxs], spikes.clusters[mask_idxs], grating_times[epoch],
            grating_vals[epoch])
    responses_mean = {epoch: np.mean(responses[epoch], axis=2) for epoch in epochs}
    responses_se = {
        epoch: np.std(responses[epoch], axis=2) / np.sqrt(responses[epoch].shape[2])
        for epoch in responses.keys()}

    # calculate osi/ori pref
    ori_pref = {epoch: [] for epoch in epochs}
    osi = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        osi[epoch], ori_pref[epoch] = compute_selectivity(
            responses_mean[epoch], np.unique(grating_vals[epoch]), 'ori')

    # find visually responsive clusters
    spontaneous = ioalf.load_object(session_path, '_iblcertif_.spontaneous')
    spont_times = {
        'beg': spontaneous['spontaneous.times.00'],
        'end': spontaneous['spontaneous.times.01']}

    # only calculate responsiveness for neurons that were active during gratings
    mask_clust = np.isin(spikes.clusters, cluster_ids)
    resp_0 = are_neurons_responsive(
        spikes.times[mask_clust], spikes.clusters[mask_clust], grating_times['beg'],
        grating_vals['beg'], spont_times['beg'])
    resp_1 = are_neurons_responsive(
        spikes.times[mask_clust], spikes.clusters[mask_clust], grating_times['end'],
        grating_vals['end'], spont_times['end'])
    resp = resp_0 & resp_1

    # remove non-responsive clusters
    cluster_ids = cluster_ids[resp]
    for epoch in epochs:
        responses[epoch] = responses[epoch][resp]
        responses_mean[epoch] = responses_mean[epoch][resp]
        responses_se[epoch] = responses_se[epoch][resp]
        osi[epoch] = osi[epoch][resp]
        ori_pref[epoch] = ori_pref[epoch][resp]

    # compare OSI at beginning/end of session
    fig0 = scatterplot(osi['beg'], osi['end'], 'OSI (beginning)', 'OSI (end)', id_line=True)

    # compare orientation preference at beginning/end of session
    fig1 = scatterplot(
        ori_pref['beg'], ori_pref['end'], 'Ori pref (beginning)', 'Ori pref (end)', id_line=True)

    # cdf of orientation preference
    fig2 = plot_cdfs(ori_pref, 'Orientation preference')

    # polar psth and rasters for example cluster
    cluster = 420
    cluster_idx = np.where(cluster_ids == cluster)[0][0]
    mean_responses = {epoch: [] for epoch in epochs}
    osis = {epoch: [] for epoch in epochs}
    binned = {epoch: [] for epoch in epochs}
    for epoch in epochs:
        mean_responses[epoch] = responses_mean[epoch][cluster_idx, :]
        osis[epoch] = osi[epoch][cluster_idx]
        stim_ids = np.unique(grating_vals[epoch])
        binned[epoch] = {j: None for j in range(len(stim_ids))}
        for j, stim_id in enumerate(stim_ids):
            curr_stim_idxs = np.where(grating_vals[epoch] == stim_id)
            align_times = grating_times[epoch][curr_stim_idxs, 0][0]
            _, binned[epoch][j] = calculate_peths(
                spikes.times[mask_idxs], spikes.clusters[mask_idxs], [cluster], align_times,
                pre_time=PRE_TIME, post_time=POST_TIME, bin_size=BIN_SIZE)
    on_idx = int(PRE_TIME / BIN_SIZE)
    off_idx = on_idx + int(2 / BIN_SIZE)  # grating stims on for 2 seconds
    fig3 = plot_polar_psth_and_rasters(
        mean_responses, binned, osis, grating_vals, on_idx, off_idx, BIN_SIZE, cluster=cluster)
