import matplotlib.pyplot as plt
import numpy as np
import brainbox.behavior.training as training
from oneibl.one import ONE
import seaborn as sns
import pickle
"""
Code by Sebastian Bruijns
"""

np.set_printoptions(suppress=True)

one = ONE()

download_data = True

sess = one.alyx.rest('sessions', 'list', task_protocol='ephys',
                     django='project__name__'
                     'icontains,ibl_neuropixel_brainwide_01')

if download_data:
    perfs = []
    n_trialses = []
    ps20s = []
    ps80s = []
    rts = []
    not_found = 0
    for i, s in enumerate(sess):
        print(i)
        try:
            trials_all = one.load_object(s['url'][-36:], 'trials')
            trials = dict()
            trials['temp_key'] = trials_all
            perf_easy, n_trials, ps20, ps80, rt = training.compute_bias_info(trials, trials_all)
            perfs.append(perf_easy[0])
            n_trialses.append(n_trials[0])
            ps20s.append(ps20)
            ps80s.append(ps80)
            rts.append(rt)
        except Exception as e:
            print(e)
            not_found += 1
    total_n = i

    pickle.dump(perfs, open('perfs', 'wb'))
    pickle.dump(n_trialses, open('n_trialses', 'wb'))
    pickle.dump(ps20s, open('ps20s', 'wb'))
    pickle.dump(ps80s, open('ps80s', 'wb'))
    pickle.dump(rts, open('rts', 'wb'))
    pickle.dump(not_found, open('not_found', 'wb'))
    pickle.dump(total_n, open('total_n', 'wb'))
else:
    perfs = np.array(pickle.load(open('perfs', 'rb')))
    n_trialses = np.array(pickle.load(open('n_trialses', 'rb')))
    # ps20 and ps80 contain: bias, threshold, lapse high, lapse low for two blocks
    ps20s = np.array(pickle.load(open('ps20s', 'rb')))
    ps80s = np.array(pickle.load(open('ps80s', 'rb')))
    rts = np.array(pickle.load(open('rts', 'rb')))
    not_found = pickle.load(open('not_found', 'rb'))
    total_n = pickle.load(open('total_n', 'rb')) + 1


sns.set_style("whitegrid", {'axes.grid': False})

ax = sns.jointplot(x=n_trialses, y=perfs, marginal_kws=dict(bins=35), height=12)
ax.set_axis_labels('# of trials', 'Performance on easy', fontsize=20)
plt.tight_layout()
plt.savefig('hist')
plt.show()

plt.figure(figsize=(16, 9))
plt.scatter(n_trialses, perfs)
plt.axhline(0.9, color='k')
plt.axvline(400, color='k')
plt.title("{}/{} downloadable ephys sessions".format(total_n - not_found, total_n), fontsize=20)
plt.xlabel("# of trials", fontsize=20)
plt.ylabel("Performance on easy", fontsize=20)


temp = np.sum(np.logical_and(perfs > 0.9, ~(n_trialses > 400)))
plt.annotate("n={} ({:.1f}%)".format(temp, temp / (total_n - not_found) * 100), (-30, 1), fontsize=20)
temp = np.sum(np.logical_and(perfs > 0.9, n_trialses > 400))
plt.annotate("n={} ({:.1f}%)".format(temp, temp / (total_n - not_found) * 100), (1310, 1), fontsize=20)
temp = np.sum(np.logical_and(~(perfs > 0.9), n_trialses > 400))
plt.annotate("n={} ({:.1f}%)".format(temp, temp / (total_n - not_found) * 100), (1310, 0.1), fontsize=20)
temp = np.sum(np.logical_and(~(perfs > 0.9), ~(n_trialses > 400)))
plt.annotate("n={} ({:.1f}%)".format(temp, temp / (total_n - not_found) * 100), (-30, 0.1), fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig('scatter')
plt.show()


previously_valid = np.logical_and(perfs > 0.9, n_trialses > 400)


rt_loss = np.sum(~(rts[previously_valid] < 2))
plt.figure(figsize=(16, 9))

rt_mask = rts < 2
plt.scatter(n_trialses[rt_mask], perfs[rt_mask], color='g')
plt.scatter(n_trialses[~rt_mask], perfs[~rt_mask], color='r')

plt.axhline(0.9, color='k')
plt.axvline(400, color='k')
plt.title("RT < 2s criterion".format(total_n - not_found, total_n), fontsize=20)
plt.xlabel("# of trials", fontsize=20)
plt.ylabel("Performance on easy", fontsize=20)

plt.annotate("-{} (-{:.1f}%)".format(rt_loss, rt_loss / np.sum(previously_valid) * 100), (1310, 1), fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig('scatter_rts')
plt.show()


###
lapse_loss = np.sum(~(np.logical_and.reduce((ps20s[previously_valid][:, 3] < 0.1, ps20s[previously_valid][:, 2] < 0.1,
                                             ps80s[previously_valid][:, 3] < 0.1, ps80s[previously_valid][:, 2] < 0.1))))
plt.figure(figsize=(16, 9))

lapse_mask = np.logical_and.reduce((ps20s[:, 3] < 0.1, ps20s[:, 2] < 0.1,
                                    ps80s[:, 3] < 0.1, ps80s[:, 2] < 0.1))
plt.scatter(n_trialses[lapse_mask], perfs[lapse_mask], color='g')
plt.scatter(n_trialses[~lapse_mask], perfs[~lapse_mask], color='r')

plt.axhline(0.9, color='k')
plt.axvline(400, color='k')
plt.title("Lapse < 0.1 criterion".format(total_n - not_found, total_n), fontsize=20)
plt.xlabel("# of trials", fontsize=20)
plt.ylabel("Performance on easy", fontsize=20)

plt.annotate("-{} (-{:.1f}%)".format(lapse_loss, lapse_loss / np.sum(previously_valid) * 100), (1310, 1), fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig('scatter_lapse')
plt.show()


###
bias_loss = np.sum(~(ps80s[previously_valid][:, 0] - ps20s[previously_valid][:, 0] > 5))
plt.figure(figsize=(16, 9))

bias_mask = ps80s[:, 0] - ps20s[:, 0] > 5
plt.scatter(n_trialses[bias_mask], perfs[bias_mask], color='g')
plt.scatter(n_trialses[~bias_mask], perfs[~bias_mask], color='r')

plt.axhline(0.9, color='k')
plt.axvline(400, color='k')
plt.title("Bias > 5% criterion".format(total_n - not_found, total_n), fontsize=20)
plt.xlabel("# of trials", fontsize=20)
plt.ylabel("Performance on easy", fontsize=20)

plt.annotate("-{} (-{:.1f}%)".format(bias_loss, bias_loss / np.sum(previously_valid) * 100), (1310, 1), fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig('scatter_bias')
plt.show()


###
all_loss = np.sum(~(np.logical_and.reduce((bias_mask[previously_valid], lapse_mask[previously_valid], rt_mask[previously_valid]))))
plt.figure(figsize=(16, 9))

all_mask = np.logical_and.reduce((previously_valid, bias_mask, lapse_mask, rt_mask))
plt.scatter(n_trialses[all_mask], perfs[all_mask], color='g')
plt.scatter(n_trialses[~all_mask], perfs[~all_mask], color='r')

plt.axhline(0.9, color='k')
plt.axvline(400, color='k')
plt.title("All criteria".format(total_n - not_found, total_n), fontsize=20)
plt.xlabel("# of trials", fontsize=20)
plt.ylabel("Performance on easy", fontsize=20)

plt.annotate("-{} (-{:.1f}%)".format(all_loss, all_loss / np.sum(previously_valid) * 100), (1310, 1), fontsize=20)

sns.despine()
plt.tight_layout()
plt.savefig('scatter_all_crit')
plt.show()
