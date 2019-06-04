"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell # for debugging
from scipy.special import erf # for psychometric functions
import datetime

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Arial')
sns.set(style="darkgrid", context="paper")
sns.set(style="darkgrid", context="paper", font_scale=1.3)

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

b =  behavioral_analyses.PsychResults * behavior.TrialSet * subject.Subject * subject.SubjectLab
behav = pd.DataFrame(b.proj('subject_nickname', 'n_trials',
	'lab_name', 'session_start_time', 'performance_easy', 'threshold', 'bias', 'lapse_low', 'lapse_high').fetch(as_dict=True))

shell()