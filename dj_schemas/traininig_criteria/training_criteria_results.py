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
from IPython import embed as shell  # for debugging
from scipy.special import erf  # for psychometric functions
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

# virtual module, should be populated
criterion_proposal = dj.create_virtual_module('analyses', 'user_anneurai_analyses')
figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

criterion_old = behavioral_analyses.SessionTrainingStatus()
criterion_new = criterion_proposal.SessionTrainingStatus()

for critidx, criterion in enumerate([criterion_old, criterion_new]):

    use_subjects = subject.Subject * subject.SubjectLab * subject.SubjectProject & 'subject_project = "ibl_neuropixel_brainwide_01"'
    subjects = use_subjects.fetch('subject_nickname')

    # find all the criterion names that are present
    status_options = pd.DataFrame(criterion.fetch('training_status', as_dict=True))
    status_options = status_options.training_status.unique()
    
    # ============================================= #
    #
    # ============================================= #

    for statusidx, status in enumerate(status_options):

        print(status)
        # START A NEW FIGURE
        try:
            del behav, bdat
        except:
            pass
        plt.close('all')

        sess = acquisition.Session * use_subjects * \
                (criterion & 'training_status="%s"'%status)
        subjects = pd.DataFrame.from_dict(sess.fetch(as_dict=True))

        for midx, mousename in enumerate(subjects['subject_nickname'].unique()):

            # ============================================= #
            # check whether the subject is trained based the the latest session
            # ============================================= #

            print(mousename)
            subj = subject.Subject & 'subject_nickname="{}"'.format(mousename)
            last_session = subj.aggr(behavior.TrialSet, session_start_time='max(session_start_time)')
            training_status = (criterion & last_session).fetch1('training_status')

            # when was the first time this criterion was reached?
            if (critidx == 0 and training_status in ['trained', 'ready for ephys']) \
                    or (critidx == 1 and training_status in
                        ['trained_1a', 'trained_1b', 'ready4ephysrig', 'ready4recording']):
                first_trained_session = subj.aggr(
                    criterion &
                    'training_status="%s"'%status,
                    first_trained='min(session_start_time)')
                first_trained_session_time = first_trained_session.fetch1(
                    'first_trained')
                # convert to timestamp
                trained_date = pd.DatetimeIndex([first_trained_session_time])[0]
                print(trained_date)
            else:
                # print('WARNING: THIS MOUSE WAS NOT TRAINED!')
                continue

            # now get the sessions that went into this
            # https://github.com/shenshan/IBL-pipeline/blob/master/ibl_pipeline/analyses/behavior.py#L390
            sessions = (behavior.TrialSet & subj &
                        (acquisition.Session) &
                        'session_start_time <= "{}"'.format(
                            trained_date.strftime(
                                '%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')

            sessions_rel = sessions[-3:]
            b = (behavior.TrialSet.Trial & sessions_rel) \
                * (subject.Subject & 'subject_nickname="%s"' % mousename)
            bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time, trial_id'))

            # APPEND
            if not 'behav' in locals():
                behav = bdat.copy()
            else:
                behav = behav.append(bdat.copy(), sort=False, ignore_index=True)

        # ================================= #
        # convert
        # ================================= #

        behav = dj2pandas(behav)
        # print(behav.describe())

        # ================================= #
        # ONE PANEL PER MOUSE
        # ================================= #

        if status in ['ready for ephys', 'ready4ephysrig', 'ready4recording']:
            behav = behav.loc[behav['probabilityLeft'].isin([20, 50, 80])]
            cmap = sns.diverging_palette(220, 20, n=len(behav['probabilityLeft'].unique()), center="dark")
            fig = sns.FacetGrid(behav, hue='probabilityLeft',
                                col="subject_nickname", col_wrap=6,
                                palette=cmap, sharex=True, sharey=True)
        else:
            fig = sns.FacetGrid(behav,
                            col="subject_nickname", col_wrap=6,
                            palette="gist_gray", sharex=True, sharey=True)

        fig.map(plot_psychometric, "signed_contrast", "choice_right", "subject_nickname").add_legend()
        fig.set_axis_labels('Signed contrast (%)', 'Rightward choice (%)')
        fig.set_titles("{col_name}")
        fig.despine(trim=True)
        fig.fig.subplots_adjust(top=0.9)
        fig.fig.suptitle('Criteria definition v%d, status ''%s'''%(critidx, status), fontsize=16)
        fig.savefig(os.path.join(figpath, "criteria_psychfuncs_crit%d_status%d_%s.png"%(critidx, statusidx, status)),
                    dpi=200)
        plt.close('all')

        # ALSO CHRONOMETRIC FUNCTIONS
        sns.set_style("darkgrid", {'xtick.bottom': True, 'ytick.left': True, 'lines.markeredgewidth': 0})
        if status in ['ready for ephys', 'ready4ephysrig', 'ready4recording']:
            fig = sns.FacetGrid(behav, hue='probabilityLeft',
                                col="subject_nickname", col_wrap=6,
                                palette=cmap, sharex=True, sharey=True)
        else:
            fig = sns.FacetGrid(behav,
                                col="subject_nickname", col_wrap=6,
                                palette="gist_gray", sharex=True, sharey=True)

        fig.map(plot_chronometric, "signed_contrast", "rt", "subject_nickname").add_legend()
        fig.set_axis_labels('Signed contrast (%)', 'RT (s)')
        fig.set_titles("{col_name}")
        fig.despine(trim=True)
        fig.fig.subplots_adjust(top=0.9)
        fig.fig.suptitle('Criteria definition v%d, status ''%s'''%(critidx, status), fontsize=16)
        fig.savefig(os.path.join(figpath, "criteria_chrono_crit%d_status%d_%s.png"%(critidx, statusidx, status)),
                    dpi=200)