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

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

subj_query = (subject.Subject & 'subject_nickname LIKE "%human%"') \
             * (acquisition.Session.proj(session_date='date(session_start_time)') & 'session_date = "2019-10-15"')
behav_all = (subj_query * behavior.TrialSet.Trial).fetch(format='frame').reset_index()
assert(not behav_all.empty)

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

for sidx, behav in behav_all.groupby(['session_start_time']):

    print(sidx)

    # 1. recode timings
    session_duration = (behav.trial_end_time.iloc[-1] - behav.trial_start_time.iloc[0])
    avg_trial_duration = session_duration / len(behav.trial_end_time)
    print("Total duration %f minutes, %d trials, s/trials %f"%(session_duration / 60,
                                                               len(behav.trial_end_time),
                                                               avg_trial_duration))

    # what is this made up of?
    behav['time_prestim'] = behav.trial_go_cue_trigger_time - behav.trial_start_time
    behav['time_rt'] = behav.trial_response_time - behav.trial_go_cue_trigger_time
    behav['time_postrep'] = behav.trial_end_time - behav.trial_response_time

    print(behav[['time_prestim', 'time_rt', 'time_postrep', 'trial_feedback_type']].groupby([
    'trial_feedback_type']).max())
