import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


folder = '/home/sebastian/Programs/iblrig/tasks/_iblrig_tasks_ephysChoiceWorld/sessions/' # location of datasets
show = 0 # whether or not to show plots (they will be saved anyway)
plot = 0 # whether or not to compute the (somewhat) time-consuming all-trial plots
plt.rcParams.update({'font.size': 12})

all_blocks = np.zeros(0)

for file in os.listdir(folder):
    if file.endswith('len_blocks.npy'):
        continue

    # Load data and bring into form for Anne's plotting procedure
    data = np.load(folder + file)
    block_lens = np.load(folder + file[:-4] + '_len_blocks.npy')
    all_blocks = np.concatenate((all_blocks, block_lens[1:]))

    behav = pd.DataFrame({'probabilityLeft':50, 'signed_contrast': 100 * np.sign(data[:,0]) * data[:,1], 'trial_id':range(np.sum(block_lens))})
    i = 90
    prob = 80 if np.mean(data[90 : 90 + block_lens[1], 0]) > 0 else 20

    for block in block_lens[1:]:
        behav.at[i:i+block, 'probabilityLeft'] = prob
        i += block
        prob = 100 - prob

    behav['probability_left_block'] = (behav.probabilityLeft - 50) * 2


    # Count streaks of 0's interlude
    current = 0
    streak = 0
    print(file)
    for i, n in enumerate(data[:,1]):
        if n == 0:
            current += 1
        else:
            current = 0
        streak = max(current, streak)
    print('Longest sequence of 0\'s is {}'.format(streak))

    # Plot contrast distributions
    plt.figure(figsize=(20, 14))
    sns.countplot(x='signed_contrast', data=behav, color='b')
    print('Proportion of stimuli on one side is {}'.format(np.sum(np.sign(behav['signed_contrast']) == 1) / (len(behav) - np.sum(np.sign(behav['signed_contrast']) == 0))))
    plt.savefig('./counts_' + file[:-4] + '.png')
    if show:
        plt.show()
    else:
        plt.close()


    # Plot contrast level of all trials + underlying block
    if plot:
        cmap = sns.diverging_palette(20, 220, n=len(behav['probabilityLeft'].unique()), center="dark")

        plt.figure(figsize=(20, 14))
        plt.gcf().suptitle(file, fontsize=16)
        for i in range(4):
            plt.subplot(4, 1, 1 + i)
            sns.lineplot(x="trial_id", y="probability_left_block", data=behav, color='k', legend=0)
            sns.lineplot(x="trial_id", y="signed_contrast", data=behav, hue=np.sign(behav.signed_contrast), palette=cmap, linewidth=0, marker='.', mec=None, legend=0)

            left, right = i * 500, (i+1) * 500
            if i == 3:
                plt.xlabel('Trial number', fontsize=16)
                right = np.sum(block_lens)
            else:
                plt.xlabel(None)

            plt.xlim(left, right)
            plt.ylabel('Signed contrast (%)', fontsize=16)
        plt.tight_layout()
        plt.savefig('./' + file[:-4] + '.png')
        if show:
            plt.show()
        else:
            plt.close()


print(min(all_blocks))
print(max(all_blocks))

n_bins = 20
n = 1000000
# Plot empirical block dist. (from n simulation) vs block lengths in given sequences

# block length draw function
def draw_block_len():
    x = np.random.exponential(50.)
    if 10 <= x <= 91: # effectively never picks 90 if you put 90 here
        return int(x)  + 10
    else:
        return draw_block_len()

# simulate block lengths
emp_blocks = np.zeros(n, dtype=np.int32)
for i in range(n):
    emp_blocks[i] = draw_block_len()

counts = np.bincount(emp_blocks)

plt.hist(all_blocks, n_bins, density=True, label='block length counts')
plt.plot(range(20, 101), counts[20:101] / n, 'r', label='Monte Carlo estimates')

plt.legend()
plt.xlabel('Block length')
plt.ylabel('Normalized frequency')
plt.savefig('./block_length_dist.png')
plt.show()
