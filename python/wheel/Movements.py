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

schema = dj.schema('group_shared_wheel')  # dj.schema('group_shared_end_criteria')


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

    key_source = behavior.CompleteWheelSession

    # key_source = behavior.CompleteWheelSession & \
    #     (acquisition.Session & 'task_protocol LIKE "%_iblrig_tasks_ephys%"')

    def make(self, key):
        # Load the wheel for this session
        move_key = key.copy()
        one = ONE()
        eid, ver = (acquisition.Session & key).fetch1('session_uuid', 'task_protocol')
        logger.info('WheelMoves for session %s, %s', str(eid), ver)

        try:  # Should be able to remove this
            wheel = one.load_object(str(eid), 'wheel')  # Fails for some sessions, e.g. CSHL_007\2019-11-08\002
            all_loaded = \
                all([isinstance(wheel[lab], np.ndarray) for lab in wheel]) and \
                all(k in wheel for k in ('timestamps', 'position'))
            assert all_loaded, 'wheel data missing'
            alf.io.check_dimensions(wheel)
            if len(wheel['timestamps'].shape) == 1:
                assert wheel['timestamps'].size == wheel['position'].size, 'wheel data dimension mismatch'
                assert np.all(np.diff(wheel['timestamps']) > 0), 'wheel timestamps not monotonically increasing'
            else:
                logger.debug('2D timestamps')
            # Check the values and units of wheel position
            res = np.array([wh.ENC_RES, wh.ENC_RES/2, wh.ENC_RES/4])
            min_change_rad = 2 * np.pi / res
            min_change_cm = wh.WHEEL_DIAMETER * np.pi / res
            pos_diff = np.abs(np.ediff1d(wheel['position']))
            if pos_diff.min() < min_change_cm.min():
                # Assume values are in radians
                units = 'rad'
                encoding = np.argmin(np.abs(min_change_rad - pos_diff.min()))
                min_change = min_change_rad[encoding]
            else:
                units = 'cm'
                encoding = np.argmin(np.abs(min_change_cm - pos_diff.min()))
                min_change = min_change_cm[encoding]
            enc_names = {0: '4X', 1: '2X', 2: '1X'}
            logger.info('Wheel in %s units using %s encoding', units, enc_names[int(encoding)])
            if '_iblrig_tasks_ephys' in ver:
                assert np.allclose(pos_diff, min_change, rtol=1e-05), 'wheel position skips'
        except ValueError:
            logger.exception('Inconsistent wheel data')
            raise
        except AssertionError as ex:
            logger.exception(str(ex))
            raise

        try:
            # Convert the pos threshold defaults from samples to correct unit
            thresholds = wh.samples_to_cm(np.array([8, 1.5]), resolution=res[encoding])
            if units == 'rad':
                thresholds = wh.cm_to_rad(thresholds)
            kwargs = {'pos_thresh': thresholds[0], 'pos_thresh_onset': thresholds[1]}
            #  kwargs = {'make_plots': True, **kwargs}
            # Interpolate and get onsets
            pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
            on, off, amp, peak_vel = wh.movements(t, pos, freq=1000, **kwargs)
            assert on.size == off.size, 'onset/offset number mismatch'
            assert np.all(np.diff(on) > 0) and np.all(np.diff(off) > 0), 'onsets/offsets not monotonically increasing'
            assert np.all((off - on) > 0), 'not all offsets occur after onset'
        except ValueError:
            logger.exception('Failed to find movements')
            raise
        except AssertionError as ex:
            logger.exception('Wheel integrity check failed: ' + str(ex))
            raise

        key['n_movements'] = on.size  # total number of movements within the session
        key['total_displacement'] = float(np.diff(pos[[0, -1]]))  # total displacement of the wheel during session
        key['total_distance'] = float(np.abs(np.diff(pos)).sum())  # total movement of the wheel
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

    key_source = behavior.CompleteTrialSession & WheelMoveSet

    def make(self, key):
        eid, ver = (acquisition.Session & key).fetch1('session_uuid', 'task_protocol')  # For logging purposes
        logger.info('MovementTimes for session %s, %s', str(eid), ver)
        query = (WheelMoveSet.Move & key).proj(
            'move_id',
            'movement_onset',
            'movement_offset')
        wheel_move_data = query.fetch(order_by='move_id')

        query = (behavior.TrialSet.Trial & key).proj(
            'trial_response_choice',
            'trial_response_time',
            'trial_stim_on_time',
            'trial_go_cue_time',
            'trial_feedback_time',
            'trial_start_time')
        trial_data = query.fetch(order_by='trial_id')

        if trial_data.size == 0 or wheel_move_data.size == 0:
            logger.warning('Missing DJ trial or move data')
            return

        all_move_onsets = wheel_move_data['movement_onset']
        go_trial = trial_data['trial_response_choice'] != 'No Go'
        feedback_times = trial_data['trial_feedback_time'][go_trial]
        start_times = trial_data['trial_start_time'][go_trial]
        cue_times = trial_data['trial_go_cue_time'][go_trial]

        # Check integrity of feedback and start times
        try:
            # Assert no nans and monotonically increasing for start times
            assert np.all(~np.isnan(start_times)) and np.all(np.diff(start_times)), 'start times missing or disordered'
            # Log presence of nans in feedback times (common)
            nan_trial = np.isnan(feedback_times)
            if nan_trial.any():
                n_feedback_nans = np.count_nonzero(nan_trial)
                logger.warning('%i feedback_times nan', np.count_nonzero(nan_trial))
                response_times = trial_data['trial_response_time'][go_trial]
                if n_feedback_nans > np.count_nonzero(np.isnan(response_times)):
                    logger.warning('using response times instead of feedback times')
                    feedback_times = response_times
                    nan_trial = np.isnan(feedback_times)

            # Assert all feedback times are monotonically increasing
            assert np.all(np.diff(feedback_times[~nan_trial]) > 0), 'feedback times not monotonically increasing'
            # Log presence of nans in go cue times times (common)
            if np.isnan(cue_times).any():
                # If all nan, use stim on
                if np.isnan(cue_times).all():
                    logger.warning('trial_go_cue_time is all nan, using trial_stim_on_time')
                    cue_times = trial_data['trial_stim_on_time'][go_trial]
                    if np.isnan(cue_times).any():
                        n_nan = 'all' if np.isnan(cue_times).all() else str(np.count_nonzero(np.isnan(cue_times)))
                        logger.warning('trial_stim_on_time nan for %s trials', n_nan)
                else:
                    logger.warning('trial_go_cue_time is nan for %i trials', np.count_nonzero(np.isnan(cue_times)))
            # Assert all cue times are montonically increasing
            assert np.all(np.diff(cue_times[~np.isnan(cue_times)]) > 0), 'cue times not monotonically increasing'
            # Assert all start times occur before feedback times
            assert np.all((feedback_times[~nan_trial] - start_times) > 0), 'feedback occurs before start time'
        except AssertionError as ex:
            logger.exception('Movement integrity check failed: ' + str(ex))
            raise

        # Segregate move onsets by trial
        onsets_by_trial = ()
        for t1, t2 in zip(start_times, feedback_times):
            if ~np.isnan(t2):
                mask = (all_move_onsets > t1) & (all_move_onsets < t2)
                if np.any(mask):
                    onsets_by_trial += np.where(mask)

        # Fetch last movement of each trial (end value of each array)
        ids = np.array([trial.take(-1) for trial in onsets_by_trial])  # Identical
        #  ids = np.array(list(zip(*onsets_by_trial))[-1])

        try:
            # Test movements for all go trials
            assert len(ids) == np.count_nonzero(go_trial)
            onsets = all_move_onsets[ids]  # Insert onsets for each trial
            assert np.all((onsets - cue_times) > 0), 'Not all response times positive'
        except AssertionError:
            logger.exception('failed to find onsets for all trials')
            raise

        movement_data = zip(
            trial_data['trial_id'][go_trial],  # trial_id
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
