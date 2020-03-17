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
import errno
import os
import random

one = ONE()
# eid = '89f0d6ff-69f4-45bc-b89e-72868abb042a'  # Truncated videos
# eid = 'e49d8ee7-24b9-416a-9d04-9be33b655f40'  # No raw video
# eid = '3d6f6788-0b99-410f-9703-c43ca3e42a21'  # no raw video
eid = '77224050-7848-4680-ad3c-109d3bcd562c'  # ibl_witten_13\2019-11-01\001

def find_sessions():
    """
    Compile list of eids with required files, i.e. raw camera and wheel data
    :return: list of session eids
    """
    datasets = ['_iblrig_Camera.raw', 'camera.times',
                'wheel.timestamps', 'wheel.position']
    return one.search(dataset_types=datasets)


#eids = find_sessions()
#eid = random.choice(eids)
print('using session {}'.format(eid))


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


def download_raw_video(eid, cameras=None):
    """
    Downloads the raw video from FlatIron or cache dir.  This allows you to download just one of the
    three videos
    :param eid: the experiment uuid
    :param cameras: the specific camera to load (i.e. 'left', 'right', or 'body') If None all
    three videos are downloaded.
    :return: the file path(s) of the raw videos
    """
    if cameras:
        cameras = [cameras] if isinstance(cameras, str) else cameras
        cam_files = ['_iblrig_{}Camera.raw.mp4'.format(cam) for cam in cameras]
        datasets = one._alyxClient.get('sessions/' + eid)['data_dataset_session_related']
        urls = [ds['data_url'] for ds in datasets if ds['name'] in cam_files]
        cache_dir = one.path_from_eid(eid).joinpath('raw_video_data')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        else:  # Check if file already downloaded
            cam_files = [file[:-4] for file in cam_files]  # Remove ext
            filenames = [f for f in os.listdir(cache_dir) if any([cam in f for cam in cam_files])]
            if filenames:
                return [cache_dir.joinpath(file) for file in filenames]
        return http_download_file_list(urls, username=one._par.HTTP_DATA_SERVER_LOGIN,
                password=one._par.HTTP_DATA_SERVER_PWD, cache_dir=str(cache_dir))
    else:
        return one.load(eid, ['_iblrig_Camera.raw'], download_only=True)


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


def get_trial_data(mode='ONE'):
    """
    Obtain dict of trial data
    :param mode: get data from ONE (default) or DataJoint
    :return: dict of ALF trials object
    @todo return with same keys
    """
    DJ2ONE = {}
    if mode is 'DataJoint':
        restriction = acquisition.Session & {'session_uuid': eid}
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
        return one.load_object(eid, 'trials')


# These are for the dict returned by DJ
# trial_data = get_trial_data('DataJoint')
# go_trial = trial_data['trial_response_choice'] != 'No Go'
# feedback_times = trial_data['trial_feedback_time'][go_trial]
# start_times = trial_data['trial_start_time'][go_trial]
# cue_times = trial_data['trial_stim_on_time'][go_trial]

# These are for the dict returned by ONE
trial_data = get_trial_data('ONE')
go_trial = trial_data['choice'] != 0
feedback_times = trial_data['feedback_times']
start_times = trial_data['intervals'][:, 0]
cue_times = trial_data['goCue_times']

trial_num = int(100)

wheel = one.load_object(eid, 'wheel')
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
t_mask = (t >= trial_data['intervals'][trial_num, 0]) & \
         (t <= trial_data['intervals'][trial_num+1, 0])
t = t[t_mask]
pos = pos[t_mask]
on, off, amp, peak_vel = wh.movements(t, pos, freq=1000, **kwargs)

# Load video
camera = 'left'
video_path, = download_raw_video(eid, camera)  # Download the raw video for left camera only
cam_ts = one.load(eid, ['camera.times'], dclass_output=True)
cam_ts, = [ts for ts, url in zip(cam_ts.data, cam_ts.url) if camera in url]
#  _, cam_ts, _ = one.load(eid, ['camera.times'])  # leftCamera is in the middle of the list


def frames_for_period(start_time=None, end_time=None):
    if isinstance(start_time, int):
        end_times = trial_data['intervals'][:, 1]
        strt_times = trial_data['intervals'][:, 0]
        end_time = end_times[start_time] if not end_time else end_times[end_time]
        start_time = strt_times[start_time]
    mask = (cam_ts >= start_time) & (cam_ts <= end_time)
    return np.where(mask)[0]


# Get the path to this session's left camera video @todo return this from load_raw_video
# try:
#     video_path = next(one.path_from_eid(eid).joinpath('raw_video_data').glob('*leftCamera*.mp4'))
# except StopIteration:
#     filename = str(one.path_from_eid(eid).joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4'))
#     raise FileNotFoundError(
#         errno.ENOENT, os.strerror(errno.ENOENT), filename)

# Plot the first fram in the upper subplot
fig, axes = plt.subplots(nrows=2)
frames = frames_for_period(trial_num)  # Get frame indicies for failed detection trial
cam_ts = cam_ts[frames]
frame_images = get_video_frames_preload(video_path, frames)
#  frame = get_video_frame(video_path, frames[0])

# Get the sample numbers for each onset and offset
#  onoff_samps = np.array([[np.argmax(t >= a), np.argmax(t >= b)] for a, b in zip(on, off)])
onoff_samps = list()
on_samp = 0
off_samp = 0
for a, b in zip(on, off):
    on_samp += np.argmax(t[on_samp:] >= a)
    off_samp += np.argmax(t[off_samp:] >= b)
    onoff_samps.append((on_samp, off_samp))

# Plot the wheel trace in the lower subplot
onoff_samps = np.array(onoff_samps)
indicies = np.sort(np.hstack(onoff_samps))  # Points to split trace
t_win = 3  # Time window of wheel plot


def init():
    im = axes[0].imshow(frame_images[0])
    axes[0].axis('off')

    # Plot the wheel position and velocity
    axes[1].plot(on, pos[onoff_samps[:, 0]], 'go')
    axes[1].plot(off, pos[onoff_samps[:, 1]], 'bo')
    t_split = np.split(np.vstack((t, pos)).T, indicies, axis=0)
    axes[1].add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
    axes[1].add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving
    axes[1].set_ylabel('position')
    axes[1].legend(['onsets', 'offsets', 'in movement'])

    # Plot some trial events
    t1 = start_times[trial_num]
    t2 = feedback_times[trial_num]
    t3 = cue_times[trial_num]
    pos_rng = [pos.min(), pos.max()]  # The range for vertical lines on plot
    axes[1].vlines([t1, t2, t3], pos_rng[0], pos_rng[1],
                   colors=['r', 'b', 'g'], linewidth=0.5,
                   label=['start_time', 'feedback_time', 'cue_time'])

    # # Plot each sample
    # #plt.plot(wheel['timestamps'], wheel['position'], 'kx')
    #
    axes[1].set_xlim([t[0], t[0] + t_win])
    axes[1].set_ylim(pos_rng)

    # Plot time marker
    ln = axes[1].axvline(x=cam_ts[0], color='k')
    axes[1].set_xlim([cam_ts[0]-(t_win/2), cam_ts[0]+(t_win/2)])

    return im, ln


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

def animate(i):
    tstart = time.time()  # for profiling
    frame = frame_images[i]
    t_x = cam_ts[i]
    ln.set_xdata([t_x, t_x])
    axes[1].set_xlim([t_x - (t_win / 2), t_x + (t_win / 2)])
    im.set_data(frame)
    print('FPS:' + str(200/(time.time()-tstart)))
    return im, ln

im, ln = init()
anim = animation.FuncAnimation(fig, animate,
                               frames=len(frame_images), interval=20, blit=False, repeat=True)
plt.show()