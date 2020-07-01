"""
Wheel trace viewer.  Requires cv2

Example 1 - inspect trial 100 of a given session
    from python.wheel.plot_movement_on_trial import Viewer
    eid = '77224050-7848-4680-ad3c-109d3bcd562c'
    v = Viewer(eid=eid, trial=100)

Example 2 - pick a random session to inspect
    from python.wheel.plot_movement_on_trial import Viewer
    v = Viewer()

"""
import time
import random
import json
import sys
import logging
from itertools import cycle
from more_itertools import chunked

import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# from ibl_pipeline import acquisition, behavior
from oneibl.one import ONE
from oneibl.webclient import http_download_file_list
import brainbox.behavior.wheel as wh
from python.wheel.exp_ref import eid2ref


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    #  fps = cap.get(cv2.CAP_PROP_FPS)
    #  print("Frame rate = " + str(fps))
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def get_video_frames_preload(video_path, frame_numbers):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_numbers: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280,
    3).  Also returns the frame rate and total number of frames
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if len(frame_numbers) == 0:
        return None, fps, total_frames
    elif 0 < frame_numbers[-1] >= total_frames:
        raise IndexError('frame numbers must be between 0 and ' + str(total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
    frame_images = []
    for i in frame_numbers:
        sys.stdout.write('\r')
        sys.stdout.write('loading frame {}/{}'.format(i, frame_numbers[-1]))
        sys.stdout.flush()
        ret, frame = cap.read()
        frame_images.append(frame)
    cap.release()
    return np.array(frame_images), fps, total_frames


def get_pupil_diameter(alf_path):
    """
    Shamelessly stolen from Micheal Schartner
    """
    json1_file = open(alf_path / '_ibl_leftCamera.dlc.metadata.json')
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)['columns']

    # check order
    assert json1_data[0] == 'pupil_top_r_x', 'Order is off!'
    assert json1_data[11] == 'pupil_left_r_likelihood', 'Order is off!'

    dlc = np.load(alf_path / '_ibl_leftCamera.dlc.npy')

    K = {}
    K['pupil_top_r'] = dlc[:, :3]
    K['pupil_right_r'] = dlc[:, 3:6]
    K['pupil_bottom_r'] = dlc[:, 6:9]
    K['pupil_left_r'] = dlc[:, 9:12]

    # Set values to nan if likelyhood is too low
    XYs = {}
    for part in K:
        x = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 0])
        x = x.filled(np.nan)
        y = np.ma.masked_where(K[part][:, 2] < 0.9, K[part][:, 1])
        y = y.filled(np.nan)
        XYs[part] = [x, y]

    # get both diameters (d1 = top - bottom, d2 = left - right)
    d1 = ((XYs['pupil_top_r'][0] - XYs['pupil_bottom_r'][0])**2 +
          (XYs['pupil_top_r'][1] - XYs['pupil_bottom_r'][1])**2)**0.5
    d2 = ((XYs['pupil_left_r'][0] - XYs['pupil_right_r'][0])**2 +
          (XYs['pupil_left_r'][1] - XYs['pupil_right_r'][1])**2)**0.5
    d = np.mean([d1, d2], axis=0)

    return d


def calc_optic_flow(frames, roi=None):
    # https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    # https://stackoverflow.com/questions/38131822/what-is-output-from-opencvs-dense-optical-flow-farneback-function-how-can-th
    #     flow = cv2.calcOpticalFlowFarneback(prvs,next,None, 0.5, 3, 15, 3, 10, 1.2, 0)
    #     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    pass  # TODO


class Viewer:
    def __init__(self, eid=None, trial=None, camera='left', dlc_features=None, quick_load=True,
                 t_win=3, one=None):
        """
        Plot the wheel trace alongside the video frames.  Below is list of key bindings:
        :key n: plot movements of next trial
        :key p: plot movements of previous trial
        :key r: plot movements of a random trial
        :key t: promp for a trial number to plot
        :key space: pause/play frames
        :key left: move to previous frame
        :key right: move to next frame

        :param eid: uuid of experiment session to load
        :param trial: the trial id to plot
        :param camera: the camera position to load, options: 'left' (default), 'right', 'body'
        :param plot_dlc: tuple of dlc features overlay onto frames
        :param quick_load: when true, move onset detection is performed on individual trials
        instead of entire session
        :param t_win: the window in seconds over which to plot the wheel trace
        :return: Viewer object
        """
        self._logger = logging.getLogger('ibllib')

        self.t_win = t_win  # Time window of wheel plot
        self.one = one or ONE()
        self.quick_load = quick_load

        # If None, randomly pick a session to load
        if not eid:
            self._logger.info('Finding random session')
            eids = self.find_sessions(dlc=dlc_features is not None)
            eid = random.choice(eids)
            self._logger.info('using session %s (%s)', eid, eid2ref(eid, as_dict=False))

        # Store complete session data: trials, timestamps, etc.
        self._session_data = {'eid': eid, 'ref': eid2ref(eid, one=one, parse=False), 'dlc': None}
        self._plot_data = {}  # Holds data specific to current plot, namely data for single trial

        # Download the DLC data if required
        if dlc_features:
            self._session_data['dlc'] = self.get_dlc(dlc_features, camera=camera)

        # These are for the dict returned by ONE
        trial_data = self.get_trial_data('ONE')
        total_trials = trial_data['intervals'].shape[0]
        trial = random.randint(0, total_trials) if not trial else trial
        self._session_data['total_trials'] = total_trials
        self._session_data['trials'] = trial_data

        # Check for local first movement times
        first_moves = self.one.path_from_eid(eid) / 'alf' / '_ibl_trials.firstMovement_times.npy'
        if first_moves.exists() and 'firstMovement_times' not in trial_data:
            # Load file if exists locally
            self._session_data['trials']['firstMovement_times'] = np.load(first_moves)

        # Download the raw video for left camera only
        self.video_path, = self.download_raw_video(camera)
        cam_ts = self.one.load(self._session_data['eid'], ['camera.times'], dclass_output=True)
        cam_ts, = [ts for ts, url in zip(cam_ts.data, cam_ts.url) if camera in url]
        # _, cam_ts, _ = one.load(eid, ['camera.times'])  # leftCamera is in the middle of the list
        Fs = 1 / np.diff(cam_ts).mean()  # Approx. frequency of camera timestamps
        # Verify video frames and timestamps agree
        _, fps, count = get_video_frames_preload(self.video_path, [])

        if count != cam_ts.size:
            assert count <= cam_ts.size, 'fewer camera timestamps than frames'
            msg = 'number of timestamps does not match number video file frames: '
            self._logger.warning(msg + '%i more timestamps than frames' % cam_ts.size - count)

        assert Fs - fps < 1, 'camera timestamps do not match reported frame rate'
        print("Frame rate = %.0fHz" % fps)
        self._session_data['camera_ts'] = cam_ts#[-count:]  # Remove extraneous timestamps

        # Load wheel data
        self._session_data['wheel'] = self.one.load_object(self._session_data['eid'], 'wheel')
        if 'firstMovement_times' in self._session_data['trials']:
            pos, t = wh.interpolate_position(
                self._session_data['wheel']['timestamps'],
                self._session_data['wheel']['position'], freq=1000)
            v, acc = wh.velocity_smoothed(pos, 1000)
            first_vel = np.full(self._session_data['trials']['goCue_times'].size, np.nan)
            for i, move in enumerate(self._session_data['trials']['firstMovement_times']):
                mask = np.logical_and(move + .5 > t, t > move - .5)
                idx, = np.where(np.abs(v[mask]) >= 0.5)
                if len(idx) > 0:
                    first_vel[i] = t[mask][idx[0]]
            self._session_data['trials']['adjusted_move_times'] = first_vel

        # Plot the first frame in the upper subplot
        fig, axes = plt.subplots(nrows=2)
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        self._plot_data['figure'] = fig
        self._plot_data['axes'] = axes
        self._trial_num = trial

        self.anim = animation.FuncAnimation(fig, self.animate, init_func=self.init_plot,
                                            frames=cycle(range(60)), interval=20, blit=False,
                                            repeat=True, cache_frame_data=False)
        self.anim.running = False
        self.trial_num = trial  # Set trial and prepare plot/frame data

    @staticmethod
    def run():
        plt.show()  # Start animation

    @property
    def trial_num(self):
        return self._trial_num

    @trial_num.setter
    def trial_num(self, trial):
        """
        Setter for the trial_num property.  Loads frames for trial, extracts onsets
        for trial and reinitializes plot
        :param trial: the trial number to select.  Must be > 0, <= len(trials['intervals'])
        :return: None
        """
        # Validate input: trial must be within range (1, total trials)
        trial = int(trial)
        total_trials = self._session_data['total_trials']
        if not 0 < trial <= total_trials:
            raise IndexError(
                'Trial number must be between 1 and {}'.format(total_trials))
        self._trial_num = trial
        sys.stdout.write('\rLoading trial ' + str(self._trial_num))

        # Our plot data, e.g. data that falls within trial
        frame_ids = self.frames_for_period(self._session_data['camera_ts'], trial-1)
        data = {
            'frames': frame_ids,
            'camera_ts': self._session_data['camera_ts'][frame_ids],
            'frame_images': get_video_frames_preload(self.video_path, frame_ids)[0]
        }
        #  frame = get_video_frame(video_path, frames[0])

        on, off, ts, pos, units = self.extract_onsets_for_trial()
        data['moves'] = {'intervals': np.c_[on, off]}
        data['wheel'] = {'ts': ts, 'pos': pos, 'units': units}

        # Update title
        ref = '{date:s}_{sequence:s}_{subject:s}'.format(**self._session_data['ref'])
        if 'firstMovement_times' in self._session_data['trials']:
            first_move = self._session_data['trials']['firstMovement_times'][trial - 1]
            go_cue = self._session_data['trials']['goCue_times'][trial - 1]
            rt = (first_move - go_cue) * 1000
            title = '%s #%i rt = %.2f ms' % (ref, trial, rt)
        else:
            title = '%s #%i' % (ref, trial)

        self._plot_data['axes'][0].set_title(title)

        # Get the sample numbers for each onset and offset
        onoff_samps = np.c_[np.searchsorted(ts, on), np.searchsorted(ts, off)]
        data['moves']['onoff_samps'] = np.array(onoff_samps)
        # Points to split trace
        data['moves']['indicies'] = np.sort(np.hstack(onoff_samps))
        data['frame_num'] = 0
        data['figure'] = self._plot_data['figure']
        data['axes'] = self._plot_data['axes']
        if 'im' in self._plot_data.keys():
            # Copy over artists
            data['im'] = self._plot_data['im']
            data['ln'] = self._plot_data['ln']
            data['dlc'] = self._plot_data['dlc']

        # Stop running so we have to to reinitialize the plot after swapping out the plot data
        if self.anim.running:
            self.anim.running = False
            if self.anim:  # deals with issues on cleanup
                self.anim.event_source.stop()
        self._plot_data = data

    def find_sessions(self, dlc=False):
        """
        Compile list of eids with required files, i.e. raw camera and wheel data
        :param dlc: search for sessions with DLC output
        :return: list of session eids
        """
        datasets = ['_iblrig_Camera.raw', 'camera.times',
                    'wheel.timestamps', 'wheel.position']
        if dlc:
            datasets.append('camera.dlc')
        return self.one.search(dataset_types=datasets)

    def download_raw_video(self, cameras=None):
        """
        Downloads the raw video from FlatIron or cache dir.  This allows you to download just one
        of the three videos
        :param cameras: the specific camera to load (i.e. 'left', 'right', or 'body') If None all
        three videos are downloaded.
        :return: the file path(s) of the raw videos
        """
        one = self.one
        eid = self._session_data['eid']
        if cameras:
            cameras = [cameras] if isinstance(cameras, str) else cameras
            cam_files = ['_iblrig_{}Camera.raw.mp4'.format(cam) for cam in cameras]
            datasets = one.alyx.rest('sessions', 'read', id=eid)['data_dataset_session_related']
            urls = [ds['data_url'] for ds in datasets if ds['name'] in cam_files]
            cache_dir = one.path_from_eid(eid).joinpath('raw_video_data')
            cache_dir.mkdir(exist_ok=True)  # Create folder if doesn't already exist
            # Check if file already downloaded
            cam_files = [file[:-4] for file in cam_files]  # Remove ext
            filenames = [f for f in cache_dir.iterdir()
                         if any([cam in str(f) for cam in cam_files])]
            if filenames:
                return [cache_dir.joinpath(file) for file in filenames]
            return http_download_file_list(urls, username=one._par.HTTP_DATA_SERVER_LOGIN,
                                           password=one._par.HTTP_DATA_SERVER_PWD,
                                           cache_dir=str(cache_dir))
        else:
            return one.load(eid, ['_iblrig_Camera.raw'], download_only=True)

    def get_trial_data(self, mode='ONE'):
        """
        Obtain dict of trial data
        :param mode: get data from ONE (default) or DataJoint
        :return: dict of ALF trials object
        @todo return with same keys
        """
        DJ2ONE = {'trial_response_choice': 'choice',
                  'trial_stim_on_time': 'stimOn_times',
                  'trial_feedback_time': 'feedback_times'}
        if mode is 'DataJoint':
            # restriction = acquisition.Session & {'session_uuid': self._session_data['eid']}
            # query = (behavior.TrialSet.Trial & restriction).proj(
            #     'trial_response_choice',
            #     'trial_response_time',
            #     'trial_stim_on_time',
            #     'trial_go_cue_time',
            #     'trial_feedback_time',
            #     'trial_start_time',
            #     'trial_end_time')
            # data = query.fetch(order_by='trial_id')
            # data['intervals'] = np.c_[data['trial_start_time'], data['trial_end_time']]
            # return data
            raise NotImplementedError('DataJoint support has been removed')
        else:
            return self.one.load_object(self._session_data['eid'], 'trials')

    def get_dlc(self, features='all', camera='left'):
        """
        Load the DLC data from file and discard features we don't need.
        :param features: tuple of features to load, or 'all' to load all available
        :param camera: which camera dlc data to load ('left', 'right', 'body')
        :return: None; data saved to self._session_data as dict with keys ('columns, 'data')
        """
        files = self.one.load(self._session_data['eid'], ['camera.dlc'], download_only=True)
        filename = '_ibl_{}Camera.dlc'.format(camera)
        if not [f for f in files if f and camera in str(f)]:
            self._logger.warning('No DLC found for %s camera', camera)
            return
        dlc_path = self.one.path_from_eid(self._session_data['eid']) / 'alf' / filename
        with open(str(dlc_path) + '.metadata.json', 'r') as meta_file:
            meta = meta_file.read()
        columns = json.loads(meta)['columns']  # parse file
        dlc = np.load(str(dlc_path) + '.npy')

        # Discard unused columns
        if features != 'all':
            incl = np.array([col.startswith(tuple(features)) for col in columns])
        else:
            incl = np.ones(len(columns), dtype=bool)

        dlc = {
            'labels': [col for col, keep in zip(columns, incl) if keep],
            'features': dlc[:, incl]}
        assert len(dlc['labels']) % 3 == 0, \
            'should be three columns per feature in the form (x, y, likelihood)'
        return dlc

    def update_dlc_plot(self, frame_idx=0):
        """
        Update the coordinates and alpha values of DLC overlay markers
        :param frame_idx: the index in the trial's frame list
        :return: None
        """
        i = self._plot_data['frames'][frame_idx]
        dlc = self._session_data['dlc']
        # Loop through labels list range in threes, i.e. [0, 1, 2], [3, 4, 5]
        for j, ln in zip(chunked(range(len(dlc['labels'])), 3), self._plot_data['dlc']):
            x, y, l = dlc['features'][i, j]
            ln.set_offsets(np.c_[x, y])  # Update marker position
            ln.set_alpha(l)  # Update alpha based on likelihood

    def frames_for_period(self, cam_ts, start_time=None, end_time=None):
        """
        Load video frames between two events
        :param cam_ts: a camera.times numpy array
        :param start_time: a timestamp for the start of the period. If an int, the trial
        interval start at that index is used.  If None, period starts at first frame
        :param end_time: a timestamp for the end of the period. If an int, the trial
        interval end at that index is used.  If None, period ends at last frame, unless start_time
        is an int, in which case the trial interval at the start_time index is used
        :return: numpy bool mask the same size as cam_ts
        """
        if isinstance(start_time, int):
            trial_ends = self._session_data['trials']['intervals'][:, 1]
            trial_starts = self._session_data['trials']['intervals'][:, 0]
            end_time = trial_ends[start_time] if not end_time else trial_ends[end_time]
            start_time = trial_starts[start_time]
        else:
            if not start_time:
                start_time = cam_ts[0]
            if not end_time:
                end_time = cam_ts[-1]
        mask = np.logical_and(cam_ts >= start_time, cam_ts <= end_time)
        return np.where(mask)[0]

    def extract_onsets_for_trial(self):
        """
        Extracts the movement onsets and offsets for the current trial
        :return: tuple of onsets, offsets on, interpolated timestamps, interpolated positions,
        and position units
        """
        wheel = self._session_data['wheel']
        trials = self._session_data['trials']
        trial_idx = self.trial_num - 1  # Trials num starts at 1
        # Check the values and units of wheel position
        res = np.array([wh.ENC_RES, wh.ENC_RES / 2, wh.ENC_RES / 4])
        # min change in rad and cm for each decoding type
        # [rad_X4, rad_X2, rad_X1, cm_X4, cm_X2, cm_X1]
        min_change = np.concatenate([2 * np.pi / res, wh.WHEEL_DIAMETER * np.pi / res])
        pos_diff = np.abs(np.ediff1d(wheel['position'])).min()

        # find min change closest to min pos_diff
        idx = np.argmin(np.abs(min_change - pos_diff))
        if idx < len(res):
            # Assume values are in radians
            units = 'rad'
            encoding = idx
        else:
            units = 'cm'
            encoding = idx - len(res)
        thresholds = wh.samples_to_cm(np.array([8, 1.5]), resolution=res[encoding])
        if units == 'rad':
            thresholds = wh.cm_to_rad(thresholds)
        kwargs = {'pos_thresh': thresholds[0], 'pos_thresh_onset': thresholds[1]}
        #  kwargs = {'make_plots': True, **kwargs}  # Uncomment for plot

        # Interpolate and get onsets
        pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
        # Get the positions and times between our trial start and the next trial start
        if self.quick_load or not self.trial_num:
            try:
                # End of previous trial to beginning of next
                t_mask = np.logical_and(t >= trials['intervals'][trial_idx - 1, 1],
                                        t <= trials['intervals'][trial_idx + 1, 0])
            except IndexError:  # We're on the last trial
                # End of previous trial to end of current
                t_mask = np.logical_and(t >= trials['intervals'][trial_idx - 1, 1],
                                        t <= trials['intervals'][trial_idx, 1])
        else:
            t_mask = np.ones_like(t, dtype=bool)
        wheel_ts = t[t_mask]
        wheel_pos = pos[t_mask]
        on, off, _, _ = wh.movements(wheel_ts, wheel_pos, freq=1000, **kwargs)
        return on, off, wheel_ts, wheel_pos, units

    def init_plot(self):
        """
        Plot the wheel data for the current trial
        :return: None
        """
        self._logger.debug('Initializing plot')
        data = self._plot_data
        trials = self._session_data['trials']
        trial_idx = self.trial_num - 1
        if 'im' in data.keys():
            data['im'].set_data(data['frame_images'][0])
        else:
            data['im'] = data['axes'][0].imshow(data['frame_images'][0])

        # Plot DLC features
        dlc = self._session_data.get('dlc')
        dlc_paths = []  # DLC marker PathsCollections
        if dlc:  # DLC data loaded
            if data.get('dlc'):  # Already initialized
                self.update_dlc_plot()  # Update positions for first frame
            else:  # Plot new elements
                i = data['frames'][0]  # First frame
                cmap = ['b', 'g', 'r', 'c', 'm', 'y']
                # Each feature has three dimensions (x, y, likelihood)
                labels = chunked(range(len(dlc['labels'])), 3)
                for j, colour in zip(labels, cycle(sorted(cmap * 2))):
                    x, y, l = dlc['features'][i, j]
                    p = data['axes'][0].scatter(x, y, marker='+', c=colour, s=100, alpha=l)
                    dlc_paths.append(p)
        data['dlc'] = dlc_paths or None

        data['axes'][0].axis('off')

        indicies = data['moves']['indicies']
        on = data['moves']['intervals'][:, 0]
        off = data['moves']['intervals'][:, 1]
        onoff_samps = data['moves']['onoff_samps']
        wheel_pos = data['wheel']['pos']
        wheel_ts = data['wheel']['ts']
        cam_ts = data['camera_ts']

        # Plot the wheel position
        ax = data['axes'][1]
        ax.clear()
        ax.plot(on, wheel_pos[onoff_samps[:, 0]], 'go')
        ax.plot(off, wheel_pos[onoff_samps[:, 1]], 'bo')
        if 'firstMovement_times' in trials:
            first_move = trials['firstMovement_times'][trial_idx]
            if ~np.isnan(first_move):
                first_move_pos = wheel_pos[np.where(wheel_ts > first_move)[0][0]]
                ax.plot(first_move, first_move_pos, 'ro')
        if 'adjusted_move_times' in trials:
            m = trials['adjusted_move_times'][trial_idx]
            if ~np.isnan(m):
                ax.axvline(x=m, color='m')

        t_split = np.split(np.vstack((wheel_ts, wheel_pos)).T, indicies, axis=0)
        ax.add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
        ax.add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving
        ax.set_ylabel('position / ' + data['wheel']['units'])
        ax.legend(['onsets', 'offsets', 'first move', 'in movement'])

        # Plot some trial events
        t1 = trials['intervals'][trial_idx, 0]
        t2 = trials['feedback_times'][trial_idx]
        t3 = trials['goCue_times'][trial_idx]
        t4 = trials['stimOn_times'][trial_idx]
        pos_rng = [wheel_pos.min(), wheel_pos.max()]  # The range for vertical lines on plot
        ax.vlines([t1, t2, t3, t4], pos_rng[0], pos_rng[1],
                  colors=['r', 'b', 'y', 'g'], linewidth=0.5,
                  label=['start_time', 'feedback_time', 'stim_on', 'cue_time'])

        ax.set_ylim(pos_rng)
        data['ln'] = ax.axvline(x=cam_ts[0], color='k')
        ax.set_xlim([cam_ts[0]-(self.t_win/2), cam_ts[0]+(self.t_win/2)])

        self._plot_data = data

        return data['im'], data['ln'], data['dlc']

    def animate(self, i):
        """
        Callback for figure animation.  Sets image data for current frame and moves pointer
        along axis
        :param i: unused; the current timestep of the calling method
        :return: None
        """
        t_start = time.time()
        data = self._plot_data
        if i < 0:
            self._plot_data['frame_num'] -= 1
            if self._plot_data['frame_num'] < 0:
                self._plot_data['frame_num'] = len(data['frame_images']) - 1
        else:
            self._plot_data['frame_num'] += 1
            if self._plot_data['frame_num'] >= len(data['frame_images']):
                self._plot_data['frame_num'] = 0
        i = self._plot_data['frame_num']  # NB: This is index for current trial's frame list
        # Print current frame number to terminal
        sys.stdout.write('\rFrame {} / {}'.format(i, len(data['frame_images'])))
        sys.stdout.flush()

        frame = data['frame_images'][i]
        t_x = data['camera_ts'][i]
        data['ln'].set_xdata([t_x, t_x])
        data['axes'][1].set_xlim([t_x - (self.t_win / 2), t_x + (self.t_win / 2)])
        data['im'].set_data(frame)
        if data['dlc']:
            self.update_dlc_plot(i)
        self._logger.debug('Render time: %.3f', time.time() - t_start)

        return data['im'], data['ln']

    def process_key(self, event):
        """
        Callback for key presses.
        :param event: a figure key_press_event
        :return: None
        """
        total_trials = self._session_data['total_trials']
        if event.key.isspace():
            if self.anim.running:
                self.anim.event_source.stop()
            else:
                self.anim.event_source.start()
            self.anim.running = ~self.anim.running
        elif event.key is 'right':
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            self.animate(1)
            self._plot_data['figure'].canvas.draw()
        elif event.key is 'left':
            if self.anim.running:
                self.anim.event_source.stop()
                self.anim.running = False
            self.animate(-1)
            self._plot_data['figure'].canvas.draw()
        elif event.key == 'r':
            # Pick random trial
            self.trial_num = random.randint(0, total_trials)
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 't':
            # Select trial
            sys.stdout.write('\r')
            trial = input("Input a trial within range (1, {}): \n".format(total_trials))
            if trial:
                self.trial_num = int(trial)
                self.init_plot()
                self.anim.event_source.start()
                self.anim.running = True
        elif event.key == 'n':
            # Next trial
            self.trial_num = self.trial_num + 1 if self.trial_num < total_trials else 1
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 'p':
            # Previous trial
            self.trial_num = self.trial_num - 1 if self.trial_num > 1 else total_trials
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
