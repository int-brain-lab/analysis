"""
Created on Wed Aug 22 16:03:05 2018
Plot behavioral results per session
@author: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from funcGeneral import getPaths, pullDrive
from os import listdir
from os.path import join

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
        resp = np.load(join(root_path, i, j, '1', 'cwResponse.choice.npy'))
        contr_left = np.load(join(root_path, i, j, '1', 'cwStimOn.contrastLeft.npy'))
        contr_right = np.load(join(root_path, i, j, '1', 'cwStimOn.contrastRight.npy'))
        correct = np.load(join(root_path, i, j, '1', 'cwFeedback.type.npy'))
        contrasts = np.unique([-contr_left, contr_right])
        contrasts = contrasts[contrasts != 0]        
        
        # Correct unequal length vectors
        if len(correct) == len(contr_left)-1:
            contr_left = contr_left[0:len(contr_left)-1]
            contr_right = contr_right[0:len(contr_right)-1]
        
        # Get psychometric curves
        for c in contrasts:
            #psy_curve.loc[0,c] = np.sum(correct[(-contr_left == c) | (contr_right == c)] == 1) / len(correct[(-contr_left == c) | (contr_right == c)])
            psy_curve.loc[c,j[8:10]+'/'+j[5:7]] = np.sum(resp[(-contr_left == c) | (contr_right == c)] == 2) / len(resp[(-contr_left == c) | (contr_right == c)] == 2)
        
        # Get performance
        perf_df.loc['correct',j[8:10]+'/'+j[5:7]] = np.sum(correct == 1) / len(correct)
        perf_df.loc['num_trials',j[8:10]+'/'+j[5:7]] = len(correct)
        perf_df.loc['bias',j[8:10]+'/'+j[5:7]] = (np.sum(resp == 2) - np.sum(resp == 1)) / (np.sum(resp == 2) + np.sum(resp == 1))
        
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)    
    sns.lineplot(data=psy_curve, ax=ax1, palette="YlGnBu_d", dashes=False, marker="o")
    ax1.plot([-1, 1], [0.5, 0.5], '--k')
    ax1.set_ylabel('P(right)')
    ax1.set_xlabel('Signed contrast')
    
    ax2.plot(perf_df.loc['correct'], '-ok')
    ax2.set_ylabel('Percentage correct')    
    ax3.plot(perf_df.loc['num_trials'], '-ok')
    ax3.set_ylabel('Number of trials')
    ax4.plot(perf_df.loc['bias'], '-ok')
    ax4.set_ylabel('Bias')
    
    plt.tight_layout()
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(plot_path + i, dpi=300)
    plt.show()
    

        
        
        
        
       
       