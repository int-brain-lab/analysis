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
        idxs = (spike_times > stim_time[0]) & \
               (spike_times <= stim_time[1]) & \
               np.isin(spike_clusters, cluster_ids)
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


if __name__ == '__main__':

    from pathlib import Path
    from oneibl.one import ONE
    import alf.io as ioalf

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
    for epoch in epochs:
        responses[epoch] = bin_responses(
            spikes.times, spikes.clusters, grating_times[epoch], grating_vals[epoch])
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

    # compare OSI at beginning/end of session
    fig0 = scatterplot(osi['beg'], osi['end'], 'OSI (beginning)', 'OSI (end)', id_line=True)

    # compare orientation preference at beginning/end of session
    fig1 = scatterplot(
        ori_pref['beg'], ori_pref['end'], 'Ori pref (beginning)', 'Ori pref (end)', id_line=True)

    # cdf of orientation preference
    fig2 = plot_cdfs(ori_pref, 'Orientation preference')
