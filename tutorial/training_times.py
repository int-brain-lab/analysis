# import wrappers etc
# import datajoint as dj
from ibl_pipeline import subject
# from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
# import plotly.express as px
# import plotly.graph_objects as go
import pandas as pd
import numpy as np
# import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
# import datetime
# INITIALIZE A FEW THINGS
figpath = os.path.join(os.path.expanduser('~'), 'Data', 'Figures_IBL')
subjects = (subject.Subject * subject.SubjectLab * subject.SubjectProject &
            'subject_project = "ibl_neuropixel_brainwide_01"')
subjects = subjects.proj('subject_nickname', 'sex', 'subject_birth_date', 'lab_name')


training = (behavioral_analyses.SessionTrainingStatus *
            behavioral_analyses.BehavioralSummaryByDate * subjects).fetch()
training = pd.DataFrame(training)
training['training_time'] = np.zeros(len(training))
# get trained mice
trained = pd.DataFrame(columns=['subject_nickname', 'session_date', 'performance_easy',
                       'performance'])
nicknames = []
dates = []
perform_easy = []
perform = []
training_time = []

for i in range(len(training)):
    t_time = (training['session_start_time'].iloc[i]).time()
    training_time.append(((t_time.hour * 60 + t_time.minute) * 60 + t_time.second) / 3600)
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
training['training_time'] = training_time
mice_list = np.unique(training['subject_nickname'])
trainingTime = pd.DataFrame(columns=['subject_nickname', 'subject_lab', 'mean_time', 'SD_time'])
nicknames = []
labs = []
means = []
SDs = []
training_t = []
training_d = []
for mouse in mice_list:
    print(f'processing {mouse}')
    indices = list(training['subject_nickname'] == mouse)

    if sum(indices) > 10:
        trainingTimes = []
        trainingDates = []
        sample = training.iloc[indices]
        nicknames.append(mouse)
        labs.append(sample['lab_name'].iloc[0])
        means.append(np.mean(sample['training_time']))
        SDs.append(np.std(sample['training_time']))
        trainingTimes.append(np.array(sample['training_time']))
        trainingDates.append(sample['session_start_time'])
        training_d.append(trainingDates)
        training_t.append(trainingTimes)
trainingTime['subject_nickname'] = nicknames
trainingTime['subject_lab'] = labs
trainingTime['mean_time'] = means
trainingTime['SD_time'] = SDs
trainingTime['training_times'] = training_t
trainingTime['training_dates'] = training_d
continuousTraining = pd.DataFrame(data=[training_t, training_d])
# training before 8 am
early_mice = training.loc[(training['training_time'] > 5) & (training['training_time'] < 8)]
# training after 11pm
late_mice = training.iloc[list(training['training_time'] >= 23)]
# plotting how consistent you are at training mice across their training
sns.catplot(data=trainingTime, x='subject_lab', y='SD_time')
labs = np.unique(trainingTime['subject_lab'])
for i in range(len(trainingTime)):
    if trainingTime['subject_lab'].iloc[i] == 'angelakilab':
        fig = plt.Figure()
        plt.plot_date(x=np.transpose(np.array(trainingTime['training_dates'].iloc[i])),
                      y=trainingTime['training_times'].iloc[i][0], linestyle='solid', marker='')
