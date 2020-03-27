import numpy as np
import alf.io
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
from oneibl.one import ONE
from pathlib import Path
import cv2
import csv
from scipy.signal import resample
import os
from oneibl.webclient import http_download_file_list


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def download_raw_video(eid, cameras=None):
    """
    Downloads the raw video from FlatIron or cache dir.  This allows you to download just one of the
    three videos
    :param cameras: the specific camera to load (i.e. 'left', 'right', or 'body') If None all
    three videos are downloaded.
    :return: the file path(s) of the raw videos
    """
    one = ONE()
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



def Print_Wheel_Angle_On_Video(eid, video_type, trial_range, save_video=True):

    '''

    eid: session id, e.g. 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4'
    video_type: one of 'left', 'right', 'body'
    trial_range: first and last trial number of range to be shown, e.g. [5,7]
    save_video: video is displayed and saved in local folder
    '''
    download_raw_video(eid, cameras=[video_type])


    one = ONE()
    D = one.load(eid, dataset_types=['camera.times' ,
                                     'wheel.position', 
                                     'wheel.timestamps',
                                     'trials.intervals'], dclass_output=True)

    alf_path = Path(D.local_path[0]).parent.parent / 'alf'
    video_data = alf_path.parent / 'raw_video_data'
    video_path = video_data / str('_iblrig_%sCamera.raw.mp4' %video_type)

    # that gives cam time stamps and DLC output
    cam = alf.io.load_object(alf_path, '_ibl_%sCamera' %video_type)

    # set where to read and save video and get video info
    cap = cv2.VideoCapture(video_path.as_uri())
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)))
    if save_video:
        out = cv2.VideoWriter('vid_%s_trials_%s_%s.mp4' %(eid, trial_range[0], trial_range[-1]),cv2.VideoWriter_fourcc(*'mp4v'), fps, size)# put , 0 if grey scale
    assert length < len(cam['times']), '#frames > #stamps'

     # pick trial range for which to display stuff
    trials = alf.io.load_object(alf_path, '_ibl_trials')
    num_trials = len(trials['intervals'])
    if trial_range[-1] > num_trials - 1: print('There are only %s trials' %num_trials)

    frame_start = find_nearest(cam['times'], [trials['intervals'][trial_range[0]][0]]) 
    frame_stop = find_nearest(cam['times'], [trials['intervals'][trial_range[-1]][1]])


    '''
    wheel related stuff
    '''

    wheel = alf.io.load_object(alf_path, '_ibl_wheel')
    import brainbox.behavior.wheel as wh
    pos, t = wh.interpolate_position(wheel['timestamps'], wheel['position'], freq=1000)

    w_start = find_nearest(t,trials['intervals'][trial_range[0]][0])
    w_stop = find_nearest(t,trials['intervals'][trial_range[-1]][1])

    # confine to interval 
    pos_int = pos[w_start:w_stop]
    t_int = t[w_start:w_stop]

    # alignment of cam stamps and interpolated wheel stamps
    wheel_pos = []
    kk=0 
    for wt in cam['times'][frame_start:frame_stop]:      
        wheel_pos.append(pos_int[find_nearest(t_int, wt)])
        kk+=1
        if kk%3000 == 0: print('iteration',kk)


    # writing stuff on frames
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    #set start frame
    cap.set(1,frame_start)

    k = 0
    while(cap.isOpened()):
        ret, frame = cap.read() 
        gray = frame
        
        cv2.putText(gray,'Wheel angle: ' + str(round(wheel_pos[k],2)), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        if save_video:
            out.write(gray)
        cv2.imshow('frame',gray)
        cv2.waitKey(1)
        k += 1
        if k == (frame_stop - frame_start) - 1: break

    if save_video:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

