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
from ibl_pipeline.analyses import analysis_utils as utils

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# =========================================================
# https://github.com/anne-urai/IBL-pipeline/blob/master/ibl_pipeline/analyses/behavior.py#L195
# =========================================================

def compute_reaction_time(trials):
    # median reaction time
    trials_rt = trials.proj(
            signed_contrast='trial_stim_contrast_left- \
                             trial_stim_contrast_right',
            rt='trial_response_time-trial_stim_on_time')

    rt = trials_rt.fetch(as_dict=True)
    rt = pd.DataFrame(rt)
    rt = rt[['signed_contrast', 'rt']]

    try:
        median_rt = rt.groupby('signed_contrast').median().reset_index()
    except:
        median_rt = rt.groupby('signed_contrast').count().reset_index()
        median_rt['rt'] = np.nan

    return median_rt

schema = dj.schema('user_anneurai_analyses')
print('defining table')

@schema
class TrainingStatus(dj.Lookup):
    definition = """
    training_status: varchar(32)
    """
    contents = zip(['untrainable',
                    'training in progress',
                    'trained',
                    'ready for ephys', 
                    'wrong session type run'])

# =========================================================
# ORIGINAL
# =========================================================

@schema
class SessionTrainingStatus_v0(dj.Computed):
    definition = """
    -> behavioral_analyses.PsychResults
    ---
    -> TrainingStatus
    """

    def make(self, key):
        cum_psych_results = key.copy()
        subject_key = key.copy()
        subject_key.pop('session_start_time')

        previous_sessions = SessionTrainingStatus_v0 & subject_key & \
            'session_start_time < "{}"'.format(
                key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
            )
        status = previous_sessions.fetch('training_status')

        # if the protocol for the current session is a biased session,
        # set the status to be "trained" and check up the criteria for
        # "read for ephys"
        task_protocol = (acquisition.Session & key).fetch1('task_protocol')
        if task_protocol and 'biased' in task_protocol:
            # if not(len(status) and np.any(status == 'trained')):
            #     key['training_status'] = 'wrong session type run'
            #     self.insert1(key)
            #     return

            if len(status) and np.any(status == 'ready for ephys'):
                key['training_status'] = 'ready for ephys'
                self.insert1(key)
                return

            #key['training_status'] = 'trained'

            # Criteria for "ready for ephys" status
            sessions = (behavior.TrialSet & subject_key &
                        (acquisition.Session & 'task_protocol LIKE "%biased%"') &
                        'session_start_time <= "{}"'.format(
                            key['session_start_time'].strftime(
                                '%Y-%m-%d %H:%M:%S')
                            )).fetch('KEY')
            # if not more than 3 biased sessions, keep status trained
            if len(sessions) >= 3:
                # self.insert1(key)
                # return

                sessions_rel = sessions[-3:]
                n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
                performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                    'performance_easy')
                if np.all(n_trials > 200) and np.all(performance_easy > 0.8):
                    trials = behavior.TrialSet.Trial & sessions_rel
                    prob_lefts = (dj.U('trial_stim_prob_left') & trials).fetch(
                        'trial_stim_prob_left')

                    # if no 0.5 of prob_left, keep trained
                    if not np.all(abs(prob_lefts - 0.5) > 0.001):
                        # self.insert1(key)
                        # return

                        trials_unbiased = trials & \
                            'ABS(trial_stim_prob_left - 0.5) < 0.001'

                        trials_80 = trials & \
                            'ABS(trial_stim_prob_left - 0.2) < 0.001'

                        trials_20 = trials & \
                            'ABS(trial_stim_prob_left - 0.8) < 0.001'

                        psych_unbiased = utils.compute_psych_pars(trials_unbiased)
                        psych_80 = utils.compute_psych_pars(trials_80)
                        psych_20 = utils.compute_psych_pars(trials_20)

                        criterion = abs(psych_unbiased['bias']) < 16 and \
                            psych_unbiased['threshold'] < 19 and \
                            psych_unbiased['lapse_low'] < 0.2 and \
                            psych_unbiased['lapse_high'] < 0.2 and \
                            psych_20['bias'] - psych_80['bias'] > 5

                        if criterion:
                            key['training_status'] = 'ready for ephys'
                            self.insert1(key)
                            return

        # if the current session is not a biased session
        key['training_status'] = 'training in progress'

        # if has reached 'trained' before, mark the current session 'trained as well'
        if len(status) and np.any(status == 'trained'):
            key['training_status'] = 'trained'
            self.insert1(key)
            return

        # training in progress if the animals was trained in < 3 sessions
        sessions = (behavior.TrialSet & subject_key &
                    'session_start_time <= "{}"'.format(
                        key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')
        if len(sessions) >= 3:
            # self.insert1(key)
            # return

            # training in progress if any of the last three sessions have
            # < 200 trials or performance of easy trials < 0.8
            sessions_rel = sessions[-3:]
            n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
            performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                'performance_easy')

            if np.all(n_trials > 200) and np.all(performance_easy > 0.8):
                # training in progress if the current session does not
                # have low contrasts
                contrasts = abs(
                    (behavioral_analyses.PsychResults & key).fetch1('signed_contrasts'))
                if 0 in contrasts and \
                   np.sum((contrasts < 0.065) & (contrasts > 0.001)):
                    # compute psych results of last three sessions
                    trials = behavior.TrialSet.Trial & sessions_rel
                    psych = utils.compute_psych_pars(trials)
                    cum_perform_easy = utils.compute_performance_easy(trials)

                    criterion = abs(psych['bias']) < 16 and \
                        psych['threshold'] < 19 and \
                        psych['lapse_low'] < 0.2 and \
                        psych['lapse_high'] < 0.2

                    if criterion:
                        key['training_status'] = 'trained'
                        self.insert1(key)
                        # insert computed results into the part table
                        n_trials, n_correct_trials = \
                            (behavior.TrialSet & key).fetch(
                                'n_trials', 'n_correct_trials')
                        cum_psych_results.update({
                            'cum_performance': np.divide(
                                np.sum(n_correct_trials),
                                np.sum(n_trials)),
                            'cum_performance_easy': cum_perform_easy,
                            'cum_signed_contrasts': psych['signed_contrasts'],
                            'cum_n_trials_stim': psych['n_trials_stim'],
                            'cum_n_trials_stim_right': psych[
                                'n_trials_stim_right'],
                            'cum_prob_choose_right': psych['prob_choose_right'],
                            'cum_bias': psych['bias'],
                            'cum_threshold': psych['threshold'],
                            'cum_lapse_low': psych['lapse_low'],
                            'cum_lapse_high': psych['lapse_high']
                        })
                        self.CumulativePsychResults.insert1(cum_psych_results)
                        return

        # check whether the subject has been trained over 40 days.
        if len(sessions) >= 40:
            key['training_status'] = 'untrainable'

        self.insert1(key)

    class CumulativePsychResults(dj.Part):
        definition = """
        # cumulative psych results from the last three sessions
        -> master
        ---
        cum_performance:            float   # percentage correct in this session
        cum_performance_easy=null:  float   # percentage correct on easy trials 0.5 and 1
        cum_signed_contrasts:       blob    # contrasts used in this session, negative when on the left
        cum_n_trials_stim:          blob    # number of trials for each contrast
        cum_n_trials_stim_right:    blob    # number of reporting "right" trials for each contrast
        cum_prob_choose_right:      blob    # probability of choosing right, same size as contrasts
        cum_threshold:              float
        cum_bias:                   float
        cum_lapse_low:              float
        cum_lapse_high:             float
        """


# =========================================================
# v1: min nr trials = 400
# min accuracy: 90%
# max lapse: 0.1
# max bias: 10
# max medRT: 2s
# biased blocks: gamma < 0.1, lambda < 0.1
# =========================================================

@schema
class SessionTrainingStatus_v1(dj.Computed):
    definition = """
    -> behavioral_analyses.PsychResults
    ---
    -> TrainingStatus
    """

    def make(self, key):
        cum_psych_results = key.copy()
        subject_key = key.copy()
        subject_key.pop('session_start_time')

        previous_sessions = SessionTrainingStatus_v1 & subject_key & \
            'session_start_time < "{}"'.format(
                key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
            )
        status = previous_sessions.fetch('training_status')

        # ================================================
        # if the protocol for the current session is a biased session,
        # check up the criteria for "ready for ephys"
        # ================================================

        task_protocol = (acquisition.Session & key).fetch1('task_protocol')
        if task_protocol and 'biased' in task_protocol:

            # if the animal previously reached 'ready4ephys', keep
            if len(status) and np.any(status == 'ready for ephys'):
                key['training_status'] = 'ready for ephys'
                self.insert1(key)
                return

            # assume the animal is trained once it's reached biasedChoiceWorld
            # key['training_status'] = 'trained'

            # Criteria for "ready for ephys" status
            sessions = (behavior.TrialSet & subject_key &
                        (acquisition.Session & 'task_protocol LIKE "%biased%"') &
                        'session_start_time <= "{}"'.format(
                            key['session_start_time'].strftime(
                                '%Y-%m-%d %H:%M:%S')
                            )).fetch('KEY')

            # if not more than 3 biased sessions, keep status trained
            if len(sessions) >= 3:

                sessions_rel = sessions[-3:]
                n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
                performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                    'performance_easy')
                if np.all(n_trials > 400) and np.all(performance_easy > 0.9):
                    trials = behavior.TrialSet.Trial & sessions_rel
                    prob_lefts = (dj.U('trial_stim_prob_left') & trials).fetch(
                        'trial_stim_prob_left')

                    # if all trials have prob_left of 0.5, don't test for ready4ephys
                    if not np.all(abs(prob_lefts - 0.5) > 0.001):
                                # self.insert1(key)
                                # return

                        trials_unbiased = trials & \
                            'ABS(trial_stim_prob_left - 0.5) < 0.001'

                        trials_80 = trials & \
                            'ABS(trial_stim_prob_left - 0.2) < 0.001'

                        trials_20 = trials & \
                            'ABS(trial_stim_prob_left - 0.8) < 0.001'

                        # psych_unbiased = utils.compute_psych_pars(trials_unbiased)
                        psych_80 = utils.compute_psych_pars(trials_80)
                        psych_20 = utils.compute_psych_pars(trials_20)

                        criterion = abs(psych_80['lapse_low']) < 0.1 and \
                            psych_80['lapse_high'] < 0.1 and \
                            psych_20['lapse_low'] < 0.1 and \
                            psych_20['lapse_high'] < 0.1 and \
                            psych_20['bias'] - psych_80['bias'] > 5

                        if criterion:
                            key['training_status'] = 'ready for ephys'
                            self.insert1(key)
                            return

        # if has reached 'trained' before, mark the current session 'trained as well'
        if len(status) and np.any(status == 'trained'):
            key['training_status'] = 'trained'
            self.insert1(key)
            return

        # ================================================
        # if the animal wasn't previously 'trained',
        # check if it is now
        # ================================================

        # key['training_status'] = 'training in progress'

        # training in progress if the animals was trained in < 3 sessions
        sessions = (behavior.TrialSet & subject_key &
                    'session_start_time <= "{}"'.format(
                        key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')
        if len(sessions) < 3:
            self.insert1(key)
            return

        # training in progress if any of the last three sessions have
        # < 400 trials or performance of easy trials < 0.8
        sessions_rel = sessions[-3:]
        n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
        performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
            'performance_easy')

        if np.all(n_trials > 400) and np.all(performance_easy > 0.9):
            # training in progress if the current session does not
            # have low contrasts
            contrasts = abs(
                (behavioral_analyses.PsychResults & key).fetch1('signed_contrasts'))
            if 0 in contrasts and \
               np.sum((contrasts < 0.065) & (contrasts > 0.001)):

                # compute psych results of last three sessions
                trials = behavior.TrialSet.Trial & sessions_rel
                psych = utils.compute_psych_pars(trials)
                cum_perform_easy = utils.compute_performance_easy(trials)
                medRT = compute_reaction_time(trials)

                criterion = abs(psych['bias']) < 10 and \
                    psych['threshold'] < 20 and \
                    psych['lapse_low'] < 0.1 and \
                    psych['lapse_high'] < 0.1 and \
                    medRT.loc[medRT['signed_contrast'] == 0, 'rt'].item() < 2

                if criterion:
                    key['training_status'] = 'trained'
                    self.insert1(key)

                    # insert computed results into the part table
                    n_trials, n_correct_trials = \
                        (behavior.TrialSet & key).fetch(
                            'n_trials', 'n_correct_trials')
                    cum_psych_results.update({
                        'cum_performance': np.divide(
                            np.sum(n_correct_trials),
                            np.sum(n_trials)),
                        'cum_performance_easy': cum_perform_easy,
                        'cum_signed_contrasts': psych['signed_contrasts'],
                        'cum_n_trials_stim': psych['n_trials_stim'],
                        'cum_n_trials_stim_right': psych[
                            'n_trials_stim_right'],
                        'cum_prob_choose_right': psych['prob_choose_right'],
                        'cum_bias': psych['bias'],
                        'cum_threshold': psych['threshold'],
                        'cum_lapse_low': psych['lapse_low'],
                        'cum_lapse_high': psych['lapse_high']
                    })
                    self.CumulativePsychResults.insert1(cum_psych_results)

                    # return 
                    return

        # ================================================
        # check whether the subject has been trained over 40 days
        # ================================================

        if len(sessions) >= 40:
            key['training_status'] = 'untrainable'

        # at the very end, insert into database
        self.insert1(key)

    class CumulativePsychResults(dj.Part):
        definition = """
        # cumulative psych results from the last three sessions
        -> master
        ---
        cum_performance:            float   # percentage correct in this session
        cum_performance_easy=null:  float   # percentage correct on easy trials 0.5 and 1
        cum_signed_contrasts:       blob    # contrasts used in this session, negative when on the left
        cum_n_trials_stim:          blob    # number of trials for each contrast
        cum_n_trials_stim_right:    blob    # number of reporting "right" trials for each contrast
        cum_prob_choose_right:      blob    # probability of choosing right, same size as contrasts
        cum_threshold:              float
        cum_bias:                   float
        cum_lapse_low:              float
        cum_lapse_high:             float
        """

# =========================================================
# v1: min nr trials = 400
# min accuracy: 85%
# max lapse: 0.2
# max bias: 10
# biased blocks: trained for black curve, diffBias > 5
# =========================================================

@schema
class SessionTrainingStatus_v2(dj.Computed):
    definition = """
    -> behavioral_analyses.PsychResults
    ---
    -> TrainingStatus
    """

    def make(self, key):
        cum_psych_results = key.copy()
        subject_key = key.copy()
        subject_key.pop('session_start_time')

        previous_sessions = SessionTrainingStatus_v2 & subject_key & \
            'session_start_time < "{}"'.format(
                key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
            )
        status = previous_sessions.fetch('training_status')

        # if the protocol for the current session is a biased session,
        # set the status to be "trained" and check up the criteria for
        # "read for ephys"
        task_protocol = (acquisition.Session & key).fetch1('task_protocol')
        if task_protocol and 'biased' in task_protocol:
            # if not(len(status) and np.any(status == 'trained')):
            #     key['training_status'] = 'wrong session type run'
            #     self.insert1(key)
            #     return

            if len(status) and np.any(status == 'ready for ephys'):
                key['training_status'] = 'ready for ephys'
                self.insert1(key)
                return

            key['training_status'] = 'trained'

            # Criteria for "ready for ephys" status
            sessions = (behavior.TrialSet & subject_key &
                        (acquisition.Session & 'task_protocol LIKE "%biased%"') &
                        'session_start_time <= "{}"'.format(
                            key['session_start_time'].strftime(
                                '%Y-%m-%d %H:%M:%S')
                            )).fetch('KEY')
            # if not more than 3 biased sessions, keep status trained
            if len(sessions) >= 3:
                # self.insert1(key)
                # return

                sessions_rel = sessions[-3:]
                n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
                performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                    'performance_easy')
                if np.all(n_trials > 400) and np.all(performance_easy > 0.85):
                    trials = behavior.TrialSet.Trial & sessions_rel
                    prob_lefts = (dj.U('trial_stim_prob_left') & trials).fetch(
                        'trial_stim_prob_left')

                    # if no 0.5 of prob_left, keep trained
                    if not np.all(abs(prob_lefts - 0.5) > 0.001):
                        # self.insert1(key)
                        # return

                        trials_unbiased = trials & \
                            'ABS(trial_stim_prob_left - 0.5) < 0.001'

                        trials_80 = trials & \
                            'ABS(trial_stim_prob_left - 0.2) < 0.001'

                        trials_20 = trials & \
                            'ABS(trial_stim_prob_left - 0.8) < 0.001'

                        psych_unbiased = utils.compute_psych_pars(trials_unbiased)
                        psych_80 = utils.compute_psych_pars(trials_80)
                        psych_20 = utils.compute_psych_pars(trials_20)


                        criterion = abs(psych_unbiased['bias']) < 10 and \
                            psych_unbiased['threshold'] < 20 and \
                            psych_unbiased['lapse_low'] < 0.2 and \
                            psych_unbiased['lapse_high'] < 0.2 and \
                            psych_20['bias'] - psych_80['bias'] > 5

                        # criterion = abs(psych_unbiased['bias']) < 16 and \
                        #     psych_unbiased['threshold'] < 19 and \
                        #     psych_unbiased['lapse_low'] < 0.2 and \
                        #     psych_unbiased['lapse_high'] < 0.2 and \
                        #     psych_20['bias'] - psych_80['bias'] > 5

                        if criterion:
                            key['training_status'] = 'ready for ephys'
                            self.insert1(key)
                            return

        # if the current session is not a biased session
        key['training_status'] = 'training in progress'

        # if has reached 'trained' before, mark the current session 'trained as well'
        if len(status) and np.any(status == 'trained'):
            key['training_status'] = 'trained'
            self.insert1(key)
            return

        # training in progress if the animals was trained in < 3 sessions
        sessions = (behavior.TrialSet & subject_key &
                    'session_start_time <= "{}"'.format(
                        key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')
        if len(sessions) >= 3:
            # self.insert1(key)
            # return

            # training in progress if any of the last three sessions have
            # < 200 trials or performance of easy trials < 0.8
            sessions_rel = sessions[-3:]
            n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
            performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                'performance_easy')

            if np.all(n_trials > 400) and np.all(performance_easy > 0.85):
                # training in progress if the current session does not
                # have low contrasts
                contrasts = abs(
                    (behavioral_analyses.PsychResults & key).fetch1('signed_contrasts'))
                if 0 in contrasts and \
                   np.sum((contrasts < 0.065) & (contrasts > 0.001)):
                    # compute psych results of last three sessions
                    trials = behavior.TrialSet.Trial & sessions_rel
                    psych = utils.compute_psych_pars(trials)
                    cum_perform_easy = utils.compute_performance_easy(trials)

                    criterion = abs(psych['bias']) < 10 and \
                        psych['threshold'] < 20 and \
                        psych['lapse_low'] < 0.2 and \
                        psych['lapse_high'] < 0.2

                    if criterion:
                        key['training_status'] = 'trained'
                        self.insert1(key)
                        # insert computed results into the part table
                        n_trials, n_correct_trials = \
                            (behavior.TrialSet & key).fetch(
                                'n_trials', 'n_correct_trials')
                        cum_psych_results.update({
                            'cum_performance': np.divide(
                                np.sum(n_correct_trials),
                                np.sum(n_trials)),
                            'cum_performance_easy': cum_perform_easy,
                            'cum_signed_contrasts': psych['signed_contrasts'],
                            'cum_n_trials_stim': psych['n_trials_stim'],
                            'cum_n_trials_stim_right': psych[
                                'n_trials_stim_right'],
                            'cum_prob_choose_right': psych['prob_choose_right'],
                            'cum_bias': psych['bias'],
                            'cum_threshold': psych['threshold'],
                            'cum_lapse_low': psych['lapse_low'],
                            'cum_lapse_high': psych['lapse_high']
                        })
                        self.CumulativePsychResults.insert1(cum_psych_results)
                        return

        # check whether the subject has been trained over 40 days.
        if len(sessions) >= 40:
            key['training_status'] = 'untrainable'

        self.insert1(key)

    class CumulativePsychResults(dj.Part):
        definition = """
        # cumulative psych results from the last three sessions
        -> master
        ---
        cum_performance:            float   # percentage correct in this session
        cum_performance_easy=null:  float   # percentage correct on easy trials 0.5 and 1
        cum_signed_contrasts:       blob    # contrasts used in this session, negative when on the left
        cum_n_trials_stim:          blob    # number of trials for each contrast
        cum_n_trials_stim_right:    blob    # number of reporting "right" trials for each contrast
        cum_prob_choose_right:      blob    # probability of choosing right, same size as contrasts
        cum_threshold:              float
        cum_bias:                   float
        cum_lapse_low:              float
        cum_lapse_high:             float
        """

# =================
# populate
# =================

# shell()

# SessionTrainingStatus_v2.drop() # remove old definition
# SessionTrainingStatus_v2.populate(display_progress=True)

# SessionTrainingStatus_v1.drop() # remove old definition
# SessionTrainingStatus_v2.drop() # remove old definition

# print('dropping table')
# try:
#     SessionTrainingStatus.drop() # remove old definition
#     print('table dropped')
# except:
#     print('could not drop table')

# print('populating table')
# SessionTrainingStatus_v0.populate(display_progress=True)

#SessionTrainingStatus_v1.populate(display_progress=True)


