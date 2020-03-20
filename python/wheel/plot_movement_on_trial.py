#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ibl_pipeline import acquisition, behavior
from oneibl.one import ONE
from oneibl.webclient import http_download_file_list
import numpy as np
import brainbox.behavior.wheel as wh
import cv2
import time
from itertools import cycle
import os
import random


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
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
    frame_images = []
    for i in frame_numbers:
        print('loading frame #{}'.format(i))
        ret, frame = cap.read()
        frame_images.append(frame)
    cap.release()
    return frame_images


class Viewer:
    def __init__(self, eid, trial, camera='left', plot_dlc=True, quick_load=True, t_win=3):
        """
        Plot the wheel trace alongside the video frames
        :param eid: uuid of experiment session to load
        :param trial: the trial id to plot
        :return: Viewer object
        """
        self.t_win = t_win  # Time window of wheel plot
        self.one = ONE()
        self.quick_load = quick_load
        self.eid = eid  # '77224050-7848-4680-ad3c-109d3bcd562c'  # ibl_witten_13\2019-11-01\001
        self._session_data = {}  # Holds complete session data: trials, timestamps, etc.
        self._plot_data = {}  # Holds data specific to current plot, namely data for single trial

        # If None, randomly pick a session to load
        if not eid:
            eids = self.find_sessions()
            self.eid = random.choice(eids)
            print('using session {}'.format(self.eid))

        # These are for the dict returned by ONE
        self._session_data['trials'] = self.get_trial_data('ONE')
        total_trials = self._session_data['trials']['intervals'].shape[0] - 1
        trial = random.randint(0, total_trials) if not trial else trial

        # Download the raw video for left camera only
        self.video_path, = self.download_raw_video(camera)
        cam_ts = self.one.load(self.eid, ['camera.times'], dclass_output=True)
        cam_ts, = [ts for ts, url in zip(cam_ts.data, cam_ts.url) if camera in url]
        # _, cam_ts, _ = one.load(eid, ['camera.times'])  # leftCamera is in the middle of the list
        self._session_data['camera_ts'] = cam_ts
        # Load wheel data
        self._session_data['wheel'] = self.one.load_object(self.eid, 'wheel')

        # Plot the first frame in the upper subplot
        fig, axes = plt.subplots(nrows=2)
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        self._plot_data['figure'] = fig
        self._plot_data['axes'] = axes
        self._trial_num = trial

        self.anim = animation.FuncAnimation(fig, self.animate, init_func=self.init_plot,
                                            frames=5, interval=20, blit=False, repeat=True)
        self.anim.running = False
        self.trial_num = trial
        plt.show()

    @property
    def trial_num(self):
        return self._trial_num

    @trial_num.setter
    def trial_num(self, trial):
        self._trial_num = int(trial)

        # Our plot data, e.g. data that falls within trial
        data = {'frames': self.frames_for_period(self._session_data['camera_ts'], trial)}
        data['camera_ts'] = self._session_data['camera_ts'][data['frames']]
        data['frame_images'] = get_video_frames_preload(self.video_path, data['frames'])
        #  frame = get_video_frame(video_path, frames[0])

        on, off, ts, pos, units = self.extract_onsets_for_trial()
        data['moves'] = {'intervals': np.c_[on, off]}
        data['wheel'] = {'ts': ts, 'pos': pos, 'units': units}

        # Get the sample numbers for each onset and offset
        #  onoff_samps = np.array([[np.argmax(t >= a), np.argmax(t >= b)] for a, b in zip(on, off)])
        onoff_samps = list()
        on_samp = 0
        off_samp = 0
        for a, b in zip(on, off):
            on_samp += np.argmax(ts[on_samp:] >= a)
            off_samp += np.argmax(ts[off_samp:] >= b)
            onoff_samps.append((on_samp, off_samp))

        # Plot the wheel trace in the lower subplot
        data['moves']['onoff_samps'] = np.array(onoff_samps)
        # Points to split trace
        data['moves']['indicies'] = np.sort(np.hstack(onoff_samps))
        data['frame_num'] = 0
        data['figure'] = self._plot_data['figure']
        data['axes'] = self._plot_data['axes']

        if self.anim.running:
            self.anim.running = False
            self.anim.event_source.stop()
        self._plot_data = data
        #self.init_plot()
        #self.anim.event_source.start()
        #self.anim.running = True

    def find_sessions(self):
        """
        Compile list of eids with required files, i.e. raw camera and wheel data
        :return: list of session eids
        """
        datasets = ['_iblrig_Camera.raw', 'camera.times',
                    'wheel.timestamps', 'wheel.position']
        return self.one.search(dataset_types=datasets)

    def download_raw_video(self, cameras=None):
        """
        Downloads the raw video from FlatIron or cache dir.  This allows you to download just one of the
        three videos
        :param cameras: the specific camera to load (i.e. 'left', 'right', or 'body') If None all
        three videos are downloaded.
        :return: the file path(s) of the raw videos
        """
        one = self.one
        eid = self.eid
        if cameras:
            cameras = [cameras] if isinstance(cameras, str) else cameras
            cam_files = ['_iblrig_{}Camera.raw.mp4'.format(cam) for cam in cameras]
            datasets = one._alyxClient.get('sessions/' + eid)['data_dataset_session_related']
            urls = [ds['data_url'] for ds in datasets if ds['name'] in cam_files]
            cache_dir = one.path_from_eid(eid).joinpath('raw_video_data')
            if not os.path.exists(str(cache_dir)):
                os.mkdir(str(cache_dir))
            else:  # Check if file already downloaded
                cam_files = [file[:-4] for file in cam_files]  # Remove ext
                filenames = [f for f in os.listdir(str(cache_dir))
                             if any([cam in f for cam in cam_files])]
                if filenames:
                    return [cache_dir.joinpath(file) for file in filenames]
            return http_download_file_list(urls, username=one._par.HTTP_DATA_SERVER_LOGIN,
                    password=one._par.HTTP_DATA_SERVER_PWD, cache_dir=str(cache_dir))
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
            restriction = acquisition.Session & {'session_uuid': self.eid}
            query = (behavior.TrialSet.Trial & restriction).proj(
                'trial_response_choice',
                'trial_response_time',
                'trial_stim_on_time',
                'trial_go_cue_time',
                'trial_feedback_time',
                'trial_start_time',
                'trial_end_time')
            data = query.fetch(order_by='trial_id')
            data['intervals'] = np.c_[data['trial_start_time'], data['trial_end_time']]
            return data
        else:
            return self.one.load_object(self.eid, 'trials')

    def frames_for_period(self, cam_ts, start_time=None, end_time=None):
        if isinstance(start_time, int):
            end_times = self._session_data['trials']['intervals'][:, 1]
            strt_times = self._session_data['trials']['intervals'][:, 0]
            end_time = end_times[start_time] if not end_time else end_times[end_time]
            start_time = strt_times[start_time]
        mask = np.logical_and(cam_ts >= start_time, cam_ts <= end_time)
        return np.where(mask)[0]

    def extract_onsets_for_trial(self):
        wheel = self._session_data['wheel']
        trials = self._session_data['trials']
        # Check the values and units of wheel position
        res = np.array([wh.ENC_RES, wh.ENC_RES / 2, wh.ENC_RES / 4])
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
        thresholds = wh.samples_to_cm(np.array([8, 1.5]), resolution=res[encoding])
        if units == 'rad':
            thresholds = wh.cm_to_rad(thresholds)
        kwargs = {'pos_thresh': thresholds[0], 'pos_thresh_onset': thresholds[1]}
        #  kwargs = {'make_plots': True, **kwargs}  # Uncomment for plot

        # Interpolate and get onsets
        pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
        # Get the positions and times between our trial start and the next trial start
        if self.quick_load or not self.trial_num:
            t_mask = np.logical_and(t >= trials['intervals'][self.trial_num, 0],
                                    t <= trials['intervals'][self.trial_num + 1, 0])
        else:
            t_mask = np.ones_like(t, dtype=bool)
        wheel_ts = t[t_mask]
        wheel_pos = pos[t_mask]
        on, off, _, _ = wh.movements(wheel_ts, wheel_pos, freq=1000, **kwargs)
        return on, off, wheel_ts, wheel_pos, units

    def init_plot(self):
        data = self._plot_data
        trials = self._session_data['trials']
        if 'im' in data.keys():
            data['im'].set_data(data['frame_images'][0])
        else:
            data['im'] = data['axes'][0].imshow(data['frame_images'][0])
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
        ax.plot(on, wheel_pos[onoff_samps[:, 0]], 'go')
        ax.plot(off, wheel_pos[onoff_samps[:, 1]], 'bo')
        t_split = np.split(np.vstack((wheel_ts, wheel_pos)).T, indicies, axis=0)
        ax.add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
        ax.add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving
        ax.set_ylabel('position / ' + data['wheel']['units'])
        ax.legend(['onsets', 'offsets', 'in movement'])

        # Plot some trial events
        t1 = trials['intervals'][self.trial_num, 0]
        t2 = trials['feedback_times'][self.trial_num]
        t3 = trials['goCue_times'][self.trial_num]
        pos_rng = [wheel_pos.min(), wheel_pos.max()]  # The range for vertical lines on plot
        ax.vlines([t1, t2, t3], pos_rng[0], pos_rng[1],
                  colors=['r', 'b', 'g'], linewidth=0.5,
                  label=['start_time', 'feedback_time', 'cue_time'])

        # # Plot each sample
        # #plt.plot(wheel['timestamps'], wheel['position'], 'kx')
        #
        #ax.set_xlim([wheel_ts[0], wheel_ts[0] + self.t_win])
        ax.set_ylim(pos_rng)

        # Plot time marker
        if 'ln' in data.keys():
            data['ln'].set_xdata([cam_ts[0], cam_ts[0]])
        else:
            data['ln'] = ax.axvline(x=cam_ts[0], color='k')
        ax.set_xlim([cam_ts[0]-(self.t_win/2), cam_ts[0]+(self.t_win/2)])

        self._plot_data = data

        return data['im'], data['ln']

    def animate(self, i):
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
        i = self._plot_data['frame_num']
        print('Frame {} / {}'.format(i, len(data['frame_images'])))
        frame = data['frame_images'][i]
        t_x = data['camera_ts'][i]
        data['ln'].set_xdata([t_x, t_x])
        data['axes'][1].set_xlim([t_x - (self.t_win / 2), t_x + (self.t_win / 2)])
        data['im'].set_data(frame)
        print('Render time:' + str(time.time() - t_start))

        return data['im'], data['ln']

    def process_key(self, event):
        #frame_num = anim.save_count % len(anim._save_seq)
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
            total_trials = self._session_data['trials']['intervals'].shape[0] -1
            self.trial_num = random.randint(0, total_trials)
            self.init_plot()
            self.anim.event_source.start()
            self.anim.running = True
        elif event.key == 't':
            # Select trial
            total_trials = self._session_data['trials']['intervals'].shape[0]
            trial = input("Input a trial within range (1, {}): \n".format(total_trials))
            if trial:
                self.trial_num = int(trial)
                self.init_plot()
                self.anim.event_source.start()
                self.anim.running = True




    # def animate():
#     tstart = time.time()  # for profiling
#     for i in cycle(frames):
#         #  frame = get_video_frame(video_path, i)
#         t_x = cam_ts[i]
#         ln.set_xdata([t_x, t_x])
#         axes[1].set_xlim([t_x - (t_win / 2), t_x + (t_win / 2)])
#         im.set_data(frame_images[i - frames[0]])
#
#         fig.canvas.draw()                         # redraw the canvas
#         print('FPS:' + str(200/(time.time()-tstart)))
#
#
# win = fig.canvas.manager.window
# fig.canvas.manager.window.after(100, animate)
# plt.show()