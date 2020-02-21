import datajoint as dj
from ibl_pipeline import acquisition, behavior
from oneibl.one import ONE
import numpy as np
import brainbox.behavior.wheel as wh
import logging
from logging.handlers import RotatingFileHandler
import alf.io
import os

log_dir = 'C:\\Users\\Miles\\Documents\\GitHub\\ibllib\\logs'
os.chmod(log_dir, 0o0777)

logger = logging.getLogger(__name__)
fh = RotatingFileHandler(log_dir + '\\Movements.log', maxBytes=(1048576*5))
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

schema = dj.schema('user_mileswells_wheel')  # dj.schema('group_shared_end_criteria')


@schema
class WheelMoveSet(dj.Imported):
    definition = """
    # Wheel movements occurring within a session
    -> acquisition.Session
    ---
    n_movements:            int # total number of movements within the session
    total_displacement:     float # total displacement of the wheel during session 
    total_distance:         float # total movement of the wheel
    """

    class Move(dj.Part):
        # all times are in absolute seconds relative to session
        definition = """
        -> master
        move_id:                int # movement id
        ---
        movement_onset:         float # time of movement onset in seconds from session start
        movement_offset:        float # time of movement offset in seconds from session start 
        max_velocity:           float # peak velocity of movement
        movement_amplitude:     float # movement amplitude
        """

    # key_source = behavior.CompleteWheelSession & \
    #    (acquisition.Session & 'task_protocol LIKE "%_iblrig_tasks_ephys%"')

    key_source = behavior.CompleteTrialSession & \
        (acquisition.Session & 'task_protocol LIKE "%_iblrig_tasks_ephys%"')

    def make(self, key):
        # Load the wheel for this session
        move_key = key.copy()
        one = ONE()
        eid, ver = (acquisition.Session & key).fetch1('session_uuid', 'task_protocol')
        logger.info('WheelMoves for session %s, %s', str(eid), ver)

        try:  # Should be able to remove this
            wheel = one.load_object(str(eid), 'wheel')  # Fails for some sessions, e.g. CSHL_007\2019-11-08\002
            assert wheel is not None and len(wheel) > 1  # warning we have times and timestamps
            alf.io.check_dimensions(wheel)
        except ValueError:
            logger.exception('Inconsistent wheel data')
            raise
        except AssertionError:
            logger.exception('Missing wheel data')
            raise

        try:
            pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
            thresholds = wh.samples_to_cm(np.array([8, 1.5]))
            on, off, amp, peak_vel = wh.movements(
                t, pos, freq=1000, pos_thresh=thresholds[0], pos_thresh_onset=thresholds[1])
            assert on.size == off.size
        except ValueError:
            logger.exception('Failed to find movements')
            raise

        key['n_movements'] = on.size  # total number of movements within the session
        key['total_displacement'] = float(np.diff(pos[[0, -1]]))  # total displacement of the wheel during session
        key['total_distance'] = np.abs(np.diff(pos)).sum()  # total movement of the wheel
        self.insert1(key)

        keys = ('move_id', 'movement_onset', 'movement_offset', 'max_velocity', 'movement_amplitude')
        moves = [dict(zip(keys, (i, on[i], off[i], amp[i], peak_vel[i]))) for i in np.arange(on.size)]
        [x.update(move_key) for x in moves]

        self.Move.insert(moves)


@schema
class MovementTimes(dj.Computed):
    definition = """
    # Wheel movements occurring within a session
    -> behavior.TrialSet.Trial
    ---
    -> WheelMoveSet.Move
    reaction_time:          float # time in seconds from go cue to last movement onset of the trial
    movement_time:          float # time in seconds from last movement onset to feedback time
    response_time:          float # time in seconds from go cue to feedback time 
    movement_onset:         float # time in seconds when last movement onset occurred
    """

    key_source = behavior.CompleteTrialSession & WheelMoveSet & \
        (acquisition.Session & 'task_protocol LIKE "%_iblrig_tasks_ephys%"')

    def make(self, key):
        eid, ver = (acquisition.Session & key).fetch1('session_uuid', 'task_protocol')  # For logging purposes
        logger.info('MovementTimes for session %s, %s', str(eid), ver)
        query = (WheelMoveSet.Move & key).proj(
            'move_id',
            'movement_onset',
            'movement_offset')
        wheel_move_data = query.fetch(order_by='move_id')

        query = (behavior.TrialSet.Trial & key).proj(
            'trial_response_time',
            'trial_stim_on_time',
            'trial_go_cue_time',
            'trial_feedback_time')
        trial_data = query.fetch(order_by='trial_id')

        if trial_data.size == 0 or wheel_move_data.size == 0:
            logger.warning('Missing DJ trial or move data')
            return

        # find onsets for each trial
        all_move_onsets = wheel_move_data['movement_onset']
        feedback_times = trial_data['trial_feedback_time']
        if np.isnan(feedback_times).any():
            logger.warning('%i feedback_times nan', np.count_nonzero(np.isnan(feedback_times)))

        def last(arr): return arr[-1] if arr.size > 0 else np.nan

        ids = np.array([last(np.where(all_move_onsets < t)[0]) for t in feedback_times])
        try:
            assert ids.size == trial_data['trial_id'].size
        except AssertionError:
            logger.exception('Move onsets total trials mismatch')
            raise
        onsets = np.full(trial_data['trial_id'].shape, np.nan)  # Initialize onsets for each trial
        onsets[~np.isnan(ids)] = all_move_onsets[ids[~np.isnan(ids)]]  # Insert onsets for each trial

        cue_times = trial_data['trial_go_cue_time']
        if np.isnan(cue_times).any():
            # If all nan, use stim on
            if np.isnan(cue_times).all():
                logger.warning('trial_go_cue_time is all nan, using trial_stim_on_time')
                cue_times = trial_data['trial_stim_on_time']
                if np.isnan(cue_times).any():
                    n_nan = 'all' if np.isnan(cue_times).all() else str(np.count_nonzero(np.isnan(cue_times)))
                    logger.warning('trial_stim_on_time nan for %s trials', n_nan)
            else:
                logger.warning('trial_go_cue_time is nan for %i trials', np.count_nonzero(np.isnan(cue_times)))

        movement_data = zip(
            trial_data['trial_id'],  # trial_id
            ids,  # wheel_move_id
            onsets,  # movement_onset
            feedback_times - cue_times,  # response_time
            onsets - cue_times,  # reaction_time
            feedback_times - onsets  # movement_time
        )
        data = []
        for n, mid, onset, resp, rt, mvmt in movement_data:
            if np.isnan([n, mid, onset, resp, rt, mvmt]).any():
                logger.warning('nan found for trial %i', n)
                continue
            key.update({
                'move_id': mid,
                'trial_id': n,
                'movement_onset': onset,
                'response_time': resp,
                'reaction_time': rt,
                'movement_time': mvmt})
            data.append(key.copy())
            # self.insert1(key)
        self.insert(data)
