"""
New schemas for behavioral training criteria
Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import datajoint as dj
from IPython import embed as shell # for debugging
import datetime

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *
from ibl_pipeline.analyses import analysis_utils as utils

# https://int-brain-lab.slack.com/archives/CB13FQFK4/p1561595587061400
behavior_bpod = dj.create_virtual_module('behavior', 'ibl_behavior')

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

# schema = dj.schema('group_shared_anneurai_analyses')
schema = dj.schema('user_anneurai_analyses')

print('defining table')

# =========================================================
# DEFINE THE SCHEMA
# =========================================================

@schema
class TrainingStatus(dj.Lookup):
    definition = """
    training_status: varchar(32)
    """
    contents = zip(['untrainable',
                    'unbiasable',
                    'in_training',
                    'trained_1a',
                    'trained_1b',
                    'ready4ephysrig',
                    'ready4recording'])

@schema
class SessionTrainingStatus(dj.Computed):
    definition = """
    -> behavioral_analyses.PsychResults
    ---
    -> TrainingStatus
    """

    def make(self, key):

        subject_key = key.copy()
        subject_key.pop('session_start_time')

        previous_sessions = SessionTrainingStatus & subject_key & \
            'session_start_time < "{}"'.format(
                key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
            )
        status = previous_sessions.fetch('training_status')

        # ========================================================= #
        # is the animal ready to be recorded?
        # ========================================================= #

        # if the protocol for the current session is a biased session,
        # set the status to be "trained" and check up the criteria for
        # "read for ephys"
        task_protocol = (acquisition.Session & key).fetch1('task_protocol')
        if task_protocol and 'biased' in task_protocol:

            # if the previous status was 'ready4recording', keep
            if len(status) and np.any(status == 'ready4recording'):
                key['training_status'] = 'ready4recording'
                self.insert1(key)
                return

            # Criteria for "ready4recording"
            sessions = (behavior.TrialSet & subject_key &
                        (acquisition.Session & 'task_protocol LIKE "%biased%"') &
                        'session_start_time <= "{}"'.format(
                            key['session_start_time'].strftime(
                                '%Y-%m-%d %H:%M:%S')
                            )).fetch('KEY')

            # if more than 3 biased sessions, see what's up
            if len(sessions) >= 3:

                sessions_rel = sessions[-3:]

                # were these last 3 sessions done on an ephys rig?
                bpod_board = (behavior_bpod.Settings & sessions_rel).fetch('pybpod_board')
                ephys_board = [True for i in list(bpod_board) if 'ephys' in i]

                if len(ephys_board) == 3:

                    n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
                    performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                        'performance_easy')

                    # criterion: 3 sessions with >400 trials, and >90% correct on high contrasts
                    if np.all(n_trials > 400) and np.all(performance_easy > 0.9):

                        trials = behavior.TrialSet.Trial & sessions_rel
                        prob_lefts = (dj.U('trial_stim_prob_left') & trials).fetch(
                            'trial_stim_prob_left')

                        # if no 0.5 of prob_left, keep trained
                        if not np.all(abs(prob_lefts - 0.5) > 0.001):

                            # compute psychometric functions for each of 3 conditions
                            trials_50 = trials & \
                                'ABS(trial_stim_prob_left - 0.5) < 0.001'

                            trials_80 = trials & \
                                'ABS(trial_stim_prob_left - 0.2) < 0.001'

                            trials_20 = trials & \
                                'ABS(trial_stim_prob_left - 0.8) < 0.001'

                            # also compute the median reaction time
                            medRT = compute_reaction_time(trials)

                            # psych_unbiased = utils.compute_psych_pars(trials_unbiased)
                            psych_80 = utils.compute_psych_pars(trials_80)
                            psych_20 = utils.compute_psych_pars(trials_20)
                            psych_50 = utils.compute_psych_pars(trials_50)

                            # repeat the criteria for training_1b
                            # add on criteria for lapses and bias shift in the biased blocks
                            criterion = psych_80['lapse_low'] < 0.1 and \
                                psych_80['lapse_high'] < 0.1 and \
                                psych_20['lapse_low'] < 0.1 and \
                                psych_20['lapse_high'] < 0.1 and \
                                psych_20['bias'] - psych_80['bias'] > 5 and \
                                abs(psych_50['bias']) < 10 and \
                                psych_50['threshold'] < 20 and \
                                psych_50['lapse_low'] < 0.1 and \
                                psych_50['lapse_high'] < 0.1 and \
                                medRT.loc[medRT['signed_contrast'] == 0, 'rt'].item() < 2

                            if criterion:
                                # were all 3 sessions done on an ephys rig already?
                                key['training_status'] = 'ready4recording'
                                self.insert1(key)
                                return

        # ========================================================= #
        # is the animal doing biasedChoiceWorld
        # ========================================================= #

        # if the protocol for the current session is a biased session,
        # set the status to be "trained" and check up the criteria for
        # "read for ephys"
        task_protocol = (acquisition.Session & key).fetch1('task_protocol')
        if task_protocol and 'biased' in task_protocol:

            # if the previous status was 'ready4ephysrig', keep
            if len(status) and np.any(status == 'ready4ephysrig'):
                key['training_status'] = 'ready4ephysrig'
                self.insert1(key)
                return

            # Criteria for "ready4recording" or "ready4ephysrig" status
            sessions = (behavior.TrialSet & subject_key &
                        (acquisition.Session & 'task_protocol LIKE "%biased%"') &
                        'session_start_time <= "{}"'.format(
                            key['session_start_time'].strftime(
                                '%Y-%m-%d %H:%M:%S')
                            )).fetch('KEY')

            # if there are more than 40 sessions of biasedChoiceWorld, give up on this mouse
            if len(sessions) >= 40:
                key['training_status'] = 'unbiasable'

            # if not more than 3 biased sessions, see what's up
            if len(sessions) >= 3:

                sessions_rel = sessions[-3:]
                n_trials = (behavior.TrialSet & sessions_rel).fetch('n_trials')
                performance_easy = (behavioral_analyses.PsychResults & sessions_rel).fetch(
                    'performance_easy')

                # criterion: 3 sessions with >400 trials, and >90% correct on high contrasts
                if np.all(n_trials > 400) and np.all(performance_easy > 0.9):

                    trials = behavior.TrialSet.Trial & sessions_rel
                    prob_lefts = (dj.U('trial_stim_prob_left') & trials).fetch(
                        'trial_stim_prob_left')

                    # if no 0.5 of prob_left, keep trained
                    if not np.all(abs(prob_lefts - 0.5) > 0.001):

                        # compute psychometric functions for each of 3 conditions
                        trials_50 = trials & \
                            'ABS(trial_stim_prob_left - 0.5) < 0.001'

                        trials_80 = trials & \
                            'ABS(trial_stim_prob_left - 0.2) < 0.001'

                        trials_20 = trials & \
                            'ABS(trial_stim_prob_left - 0.8) < 0.001'

                        # also compute the median reaction time
                        medRT = compute_reaction_time(trials)

                        # psych_unbiased = utils.compute_psych_pars(trials_unbiased)
                        psych_80 = utils.compute_psych_pars(trials_80)
                        psych_20 = utils.compute_psych_pars(trials_20)
                        psych_50 = utils.compute_psych_pars(trials_50)

                        # repeat the criteria for training_1b
                        # add on criteria for lapses and bias shift in the biased blocks
                        criterion = psych_80['lapse_low'] < 0.1 and \
                            psych_80['lapse_high'] < 0.1 and \
                            psych_20['lapse_low'] < 0.1 and \
                            psych_20['lapse_high'] < 0.1 and \
                            psych_20['bias'] - psych_80['bias'] > 5 and \
                            abs(psych_50['bias']) < 10 and \
                            psych_50['threshold'] < 20 and \
                            psych_50['lapse_low'] < 0.1 and \
                            psych_50['lapse_high'] < 0.1 and \
                            medRT.loc[medRT['signed_contrast'] == 0, 'rt'].item() < 2

                        if criterion:
                            key['training_status'] = 'ready4ephysrig'
                            self.insert1(key)
                            return

        # ========================================================= #
        # is the animal doing trainingChoiceWorld?
        # 1B training
        # ========================================================= #

        # if has reached 'trained_1b' before, mark the current session 'trained_1b' as well
        if len(status) and np.any(status == 'trained_1b'):
            key['training_status'] = 'trained_1b'
            self.insert1(key)
            return

        # training in progress if the animals was trained in < 3 sessions
        sessions = (behavior.TrialSet & subject_key &
                    'session_start_time <= "{}"'.format(
                        key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')
        if len(sessions) >= 3:

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

                    # also compute the median reaction time
                    medRT = compute_reaction_time(trials)

                    # cum_perform_easy = utils.compute_performance_easy(trials)
                    criterion = abs(psych['bias']) < 10 and \
                        psych['threshold'] < 20 and \
                        psych['lapse_low'] < 0.1 and \
                        psych['lapse_high'] < 0.1 and \
                        medRT.loc[medRT['signed_contrast'] == 0, 'rt'].item() < 2

                    if criterion:
                        key['training_status'] = 'trained_1b'
                        self.insert1(key)
                        return

        # ========================================================= #
        # is the animal still doing trainingChoiceWorld?
        # 1A training
        # ========================================================= #

        # if has reached 'trained_1b' before, mark the current session 'trained_1b' as well
        if len(status) and np.any(status == 'trained_1a'):
            key['training_status'] = 'trained_1a'
            self.insert1(key)
            return

        # training in progress if the animals was trained in < 3 sessions
        sessions = (behavior.TrialSet & subject_key &
                    'session_start_time <= "{}"'.format(
                        key['session_start_time'].strftime('%Y-%m-%d %H:%M:%S')
                        )).fetch('KEY')
        if len(sessions) >= 3:

            # training in progress if any of the last three sessions have
            # < 400 trials or performance of easy trials < 0.8
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
                    # cum_perform_easy = utils.compute_performance_easy(trials)

                    criterion = abs(psych['bias']) < 16 and \
                        psych['threshold'] < 19 and \
                        psych['lapse_low'] < 0.2 and \
                        psych['lapse_high'] < 0.2

                    if criterion:
                        key['training_status'] = 'trained_1a'
                        self.insert1(key)
                        return

        # ========================================================= #
        # did the animal not get any criterion assigned?
        # ========================================================= #

        # check whether the subject has been trained over 40 days
        if len(sessions) >= 40:
            key['training_status'] = 'untrainable'

        # ========================================================= #
        # assume a base key of 'in_training' for all mice
        # ========================================================= #

        key['training_status'] = 'in_training'
        self.insert1(key)

# =================
# populate this
# =================

# SessionTrainingStatus.drop()

SessionTrainingStatus.populate(display_progress=True)
