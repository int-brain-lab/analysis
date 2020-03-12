import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ibl_pipeline import acquisition, behavior
from oneibl.one import ONE
import numpy as np
import brainbox.behavior.wheel as wh
import cv2
import time
from oneibl.webclient import AlyxClient, http_download_file_list
from oneibl.params import get as get_pars
from itertools import cycle
import errno
import os

one = ONE()
eid = '89f0d6ff-69f4-45bc-b89e-72868abb042a'

#  cam_files = ['_ibl_leftCamera.times.npy', '_iblrig_leftCamera.raw.mp4']
#  pars = get_pars(silent=True)
#  alyx = AlyxClient(username=pars.ALYX_LOGIN, password=pars.ALYX_PWD, base_url=pars.ALYX_URL)
#  datasets = alyx.get('sessions/' + eid)['data_dataset_session_related']
#  urls = {ds['name']: ds['data_url'] for ds in datasets}
#  urls = [ds['data_url'] for ds in datasets if ds['name'] in cam_files]
# TODO Download to ALF folder
#  files = http_download_file_list(urls, username=pars.HTTP_DATA_SERVER_LOGIN, password=pars.HTTP_DATA_SERVER_PWD)


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


restriction = acquisition.Session & {'session_uuid': eid}
query = (behavior.TrialSet.Trial & restriction).proj(
    'trial_response_choice',
    'trial_response_time',
    'trial_stim_on_time',
    'trial_go_cue_time',
    'trial_feedback_time',
    'trial_start_time',
    'trial_end_time')
trial_data = query.fetch(order_by='trial_id')

#  all_move_onsets = wheel_move_data['movement_onset']
go_trial = trial_data['trial_response_choice'] != 'No Go'
feedback_times = trial_data['trial_feedback_time'][go_trial]
start_times = trial_data['trial_start_time'][go_trial]
cue_times = trial_data['trial_stim_on_time'][go_trial]

wheel = one.load_object(str(eid), 'wheel')  # Fails for some sessions, e.g. CSHL_007\2019-11-08\002
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
#  kwargs = {'make_plots': True, **kwargs}
# Interpolate and get onsets
pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)
on, off, amp, peak_vel = wh.movements(t, pos, freq=1000, **kwargs)

# Load video
_, cam_ts, _ = one.load(eid, ['camera.times'])  # leftCamera is in the middle of the list


def frames_for_period(start_time=None, end_time=None):
    if isinstance(start_time, int):
        end_times = trial_data['trial_end_time'][go_trial]
        strt_times = trial_data['trial_start_time'][go_trial]
        end_time = end_times[start_time] if not end_time else end_times[end_time]
        start_time = strt_times[start_time]
    mask = (cam_ts >= start_time) & (cam_ts <= end_time)
    return np.where(mask)[0]


def eid2ref(eid):
    one.path_from_eid(eid)
    pass


def ref2eid(eid):
    pass


fail = int(644)  # Failure trial
t1 = start_times[fail]
t2 = feedback_times[fail]
t3 = cue_times[fail]
pos_rng = [809, 811]

try:
    video_path = next(one.path_from_eid(eid).joinpath('alf', 'raw_video_data').glob('*leftCamera*.mp4'))
except StopIteration:
    filename = str(one.path_from_eid(eid).joinpath('alf', 'raw_video_data', '_iblrig_leftCamera.raw.mp4'))
    raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), filename)

fig, axes = plt.subplots(nrows=2)
frames = frames_for_period(fail)  # Get frame indicies for failed detection trial
frame_images = get_video_frames_preload(video_path, frames)
#  frame = get_video_frame(video_path, frames[0])
im = axes[0].imshow(frame_images[0])
axes[0].axis('off')

#  onoff_samps = np.array([[np.argmax(t >= a), np.argmax(t >= b)] for a, b in zip(on, off)])
onoff_samps = list()
on_samp = 0
off_samp = 0
for a, b in zip(on, off):
    on_samp += np.argmax(t[on_samp:] >= a)
    off_samp += np.argmax(t[off_samp:] >= b)
    onoff_samps.append((on_samp, off_samp))

onoff_samps = np.array(onoff_samps)
indicies = np.sort(np.hstack(onoff_samps))  # Points to split trace
# Plot the wheel position and velocity
axes[1].plot(on, pos[onoff_samps[:, 0]], 'go')
axes[1].plot(off, pos[onoff_samps[:, 1]], 'bo')
t_split = np.split(np.vstack((t, pos)).T, indicies, axis=0)
axes[1].add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
axes[1].add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving
axes[1].set_ylabel('position')
axes[1].legend(['onsets', 'offsets', 'in movement'])

axes[1].vlines([t1, t2, t3], pos_rng[0], pos_rng[1],
               colors=['r', 'b', 'g'], linewidth=0.5,
               label=['start_time', 'feedback_time', 'cue_time'])

# # Plot each sample
# #plt.plot(wheel['timestamps'], wheel['position'], 'kx')
#
t_win = 3
axes[1].set_xlim([2797, 2800])
axes[1].set_ylim(pos_rng)

# Plot time marker
ln = axes[1].axvline(x=cam_ts[frames[0]], color='k')
axes[1].set_xlim([cam_ts[frames[0]]-(t_win/2), cam_ts[frames[0]]+(t_win/2)])


def animate():
    tstart = time.time()  # for profiling
    for i in cycle(frames):
        #  frame = get_video_frame(video_path, i)
        t_x = cam_ts[i]
        ln.set_xdata([t_x, t_x])
        axes[1].set_xlim([t_x - (t_win / 2), t_x + (t_win / 2)])
        im.set_data(frame_images[i - frames[0]])

        fig.canvas.draw()                         # redraw the canvas
        print('FPS:' + str(200/(time.time()-tstart)))


win = fig.canvas.manager.window
fig.canvas.manager.window.after(100, animate)
plt.show()

#
#
# plt.sca(plt.gcf().axes[0])
# iblplt.vertical_lines(t1, ymin=pos_rng[0], ymax=pos_rng[1],
#                       color='r', linewidth=0.5, label='start_time')
# iblplt.vertical_lines(t2, ymin=pos_rng[0], ymax=pos_rng[1],
#                       color='b', linewidth=0.5, label='feedback_time')
# iblplt.vertical_lines(t3, ymin=pos_rng[0], ymax=pos_rng[1],
#                       color='g', linewidth=0.5, label='cue_time')
#
# # Plot each sample
# #plt.plot(wheel['timestamps'], wheel['position'], 'kx')
#
# plt.gca().set_xlim([2797, 2800])
# plt.gca().set_ylim(pos_rng)
# plt.legend()
#
# plt.show()
