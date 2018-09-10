"""
Created on Wed Aug 22 16:08:04 2018
General behavioral functions
@author: Guido Meijer
"""

from os.path import join
import numpy as np
import pandas as pd

def getPaths():
    root_path = '/media/guido/data/GDrive/IBL-Mainen/Rigbox_repository/'
    plot_path = '/home/guido/Projects/SteeringWheelBehavior/Plots/'
    return root_path, plot_path

def readAlf(alf_path):
    # Load in data
    choice = np.load(join(alf_path, 'cwResponse.choice.npy'))
    contr_L = np.load(join(alf_path, 'cwStimOn.contrastLeft.npy'))
    contr_R = np.load(join(alf_path, 'cwStimOn.contrastRight.npy'))
    correct = np.load(join(alf_path, 'cwFeedback.type.npy'))
    
    # Correct unequal length vectors (can happen if time runs out during a trial)
    if len(choice) == len(contr_L)-1:
        contr_L = contr_L[0:len(contr_L)-1]
        contr_R = contr_R[0:len(contr_R)-1]
    
    # Construct dataframe
    behavior_df = pd.DataFrame(index=range(len(choice)))
    behavior_df['choice'] = choice
    behavior_df['contrast_L'] = contr_L
    behavior_df['contrast_R'] = contr_R
    behavior_df['correct'] = correct
    return behavior_df

