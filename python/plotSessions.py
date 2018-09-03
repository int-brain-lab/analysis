"""
Created on Wed Aug 22 16:03:05 2018
Plot behavioral results per session
@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from funcGeneral import getPaths, readAlf
from os import listdir
from os.path import join, isdir

# Settings
#subjects = ['6722','6723','6724','6725','6726']
subjects = ['6722']
root_path, plot_path = getPaths()

# Loop through subjects
for i in subjects:
    perf_df = pd.DataFrame()
    psy_curve = pd.DataFrame()
    sessions = [s for s in listdir(join(root_path, i)) if s[0:4] == '2018']
    for j in sessions:
        # Load in data
        if isdir(join(root_path, i, j, '2')):
            ses_dir = '2'
        else:
            ses_dir = '1'
        
        # Read in data to dataframe
        b_df = readAlf(join(root_path, i, j, ses_dir))
        
        # Get signed presented contrasts
        contrast = np.unique([-b_df['contrast_L'], b_df['contrast_R']])
        contrast = contrast[contrast != 0]
                
        # Get psychometric curves
        for c in contrast:
            psy_curve.loc[c,j[8:10]+'/'+j[5:7]] = np.sum(b_df['choice'][(-b_df['contrast_L'] == c) | (b_df['contrast_R'] == c)] == 2) / len(b_df['choice'][(-b_df['contrast_L'] == c) | (b_df['contrast_R'] == c)] == 2)
        
        # Get performance
        perf_df.loc['correct',j[8:10]+'/'+j[5:7]] = np.sum(b_df['correct'] == 1) / len(b_df['correct'])
        perf_df.loc['num_trials',j[8:10]+'/'+j[5:7]] = len(b_df['correct'])
        perf_df.loc['bias',j[8:10]+'/'+j[5:7]] = (np.sum(b_df['choice'] == 2) - np.sum(b_df['choice'] == 1)) / (np.sum(b_df['choice'] == 2) + np.sum(b_df['choice'] == 1))
        
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)    
    sns.lineplot(data=psy_curve, ax=ax1, palette="YlGnBu_d", dashes=False, marker="o")
    ax1.plot([-1, 1], [0.5, 0.5], '--k')
    ax1.set_ylabel('P(right)')
    ax1.set_xlabel('Signed contrast')
    ax1.set_title(i)
    ax1.legend().set_visible(False)
    
    ax2.plot(perf_df.loc['correct'], '-ok')
    ax2.set_ylabel('Percentage correct')    
    ax3.plot(perf_df.loc['num_trials'], '-ok')
    ax3.set_ylabel('Number of trials')
    ax4.plot(perf_df.loc['bias'], '-ok')
    ax4.set_ylabel('Bias')
    
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    fig = plt.gcf()
    fig.set_size_inches((10, 8), forward=False)
    fig.savefig(plot_path + i, dpi=300)
    
    plt.show()

        
        
        
        
       
       