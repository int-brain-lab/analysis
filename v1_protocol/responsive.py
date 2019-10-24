import numpy as np
import scipy.stats


def _get_spike_counts_in_bins(spike_times, spike_clusters, intervals=None):
    """Return the number of spikes in a sequence of time intervals, for each neuron.

    :param spike_times: times of spikes, in seconds
    :type spike_times: 1D array
    :param spike_clusters: spike neurons
    :type spike_clusters: 1D array, same length as spike_times
    :type intervals: the times of the events onsets and offsets
    :param interval: 2D array
    :rtype: 2D array of shape `(n_neurons, n_intervals)`

    """
    # Check inputs.
    assert spike_times.ndim == spike_clusters.ndim == 1
    assert spike_times.shape == spike_clusters.shape
    intervals = np.atleast_2d(intervals)
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    n_intervals = intervals.shape[0]

    # For each neuron and each interval, the number of spikes in the interval.
    neuron_ids = np.unique(spike_clusters)
    n_neurons = len(neuron_ids)
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=neuron_ids.max() + 1)
        counts[:, j] = x[neuron_ids]
    return counts  # value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j


def are_neurons_responsive(
        spike_times, spike_clusters,
        stimulus_intervals=None, spontaneous_period=None, p_value_threshold=.05):
    """Return which neurons are responsive after specific stimulus events, compared to
    spontaneous activity, according to a Wilcoxon test.

    :param spike_times: times of spikes, in seconds
    :type spike_times: 1D array
    :param spike_clusters: spike neurons
    :type spike_clusters: 1D array, same length as spike_times
    :type stimulus_intervals: the times of the stimulus events onsets and offsets
    :param stimulus_intervals: 2D array
    :type spontaneous_period: the period of spontaneous activity
    :param spontaneous_period: 1D array with 2 elements
    :param p_value_threshold: the threshold for the p value in the Wilcoxon test.
    :type p_value_threshold: float
    :rtype: 1D boolean array with `n_neurons` elements (clusters are sorted by increasing cluster
    id as appearing in spike_clusters).
    """
    stimulus_counts = _get_spike_counts_in_bins(spike_times, spike_clusters, stimulus_intervals)
    # Find spontaneous intervals.
    stimulus_durations = np.diff(stimulus_intervals, axis=1).squeeze()
    t0, t1 = spontaneous_period
    spontaneous_starts = np.linspace(t0, t1 - stimulus_durations.max(), len(stimulus_intervals))
    spontaneous_intervals = np.c_[spontaneous_starts, spontaneous_starts + stimulus_durations]
    # Count the spontaneous counts.
    spontaneous_counts = _get_spike_counts_in_bins(
        spike_times, spike_clusters, spontaneous_intervals)
    assert stimulus_counts.shape == stimulus_counts.shape
    # Generate the responsive vector (for every neuron, whether it is responsive).
    responsive = np.zeros(stimulus_counts.shape[0], dtype=np.bool)
    n_neurons = stimulus_counts.shape[0]
    for i in range(n_neurons):
        x = stimulus_counts[i, :]
        y = spontaneous_counts[i, :]
        try:
            _, p = scipy.stats.wilcoxon(x, y)
        except ValueError:
            pass
        responsive[i] = p < p_value_threshold
    return responsive


if __name__ == '__main__':
    from pathlib import Path
    path = Path("~/Downloads/FlatIron/mainenlab/Subjects/ZM_2104/2019-09-19/001/alf/").expanduser()
    odsgratings = np.load(path / '_iblcertif_.odsgratings.times.00.npy')
    spontaneous = np.load(path / '_iblcertif_.spontaneous.times.00.npy')
    spike_times = np.load(path / 'spikes.times.npy')
    spike_clusters = np.load(path / 'spikes.clusters.npy')

    resp = are_neurons_responsive(spike_times, spike_clusters, odsgratings, spontaneous)
    print(resp)
    print(resp.mean())
