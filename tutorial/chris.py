# import wrappers etc
import datajoint as dj
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses


import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# INITIALIZE A FEW THINGS
figpath = os.path.join(os.path.expanduser('~'), 'Data', 'Figures_IBL')
subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject
                    & 'subject_project = "ibl_neuropixel_brainwide_01"').proj(
                    'subject_nickname', 'sex', 'subject_birth_date', 'lab_name')


training = (behavioral_analyses.SessionTrainingStatus * behavioral_analyses.BehavioralSummaryByDate * subjects).fetch()
training = pd.DataFrame(training)
training['training_time'] = np.zeros(len(training))
 ## get trained mice
trained = pd.DataFrame(columns = ['subject_nickname','session_date','performance_easy','performance'])
nicknames = []
dates = []
perform_easy = []
perform = []
training_time = []
for i in range(len(training)):
    training_time.append(datetime.time(str(training['session_start_time'].iloc[i]).split()[1]))
    if training['training_status'].iloc[i] == 'trained':
        nicknames.append(training['subject_nickname'].iloc[i])
        dates.append(training['session_date'].iloc[i])
        perform_easy.append(training['performance_easy'].iloc[i])
        perform.append(training['performance'].iloc[i])
        print(f'processing session {i} out of {len(training)}')
trained['subject_nickname'] = nicknames
trained['session_date'] = dates
trained['performance_easy'] = perform_easy
trained['performance'] = perform
training['training_time'] = datetime.time(training_time)
