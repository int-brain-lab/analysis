from pylab import *
ion()
from scipy.io import wavfile
from ibllib.io import raw_data_loaders as raw

def plot_audio_events():
 fs, nm = wavfile.read('/home/mic/Videos/IBL/audio/noMouse/NoMouse.wav')
 nm=nm[:int(len(nm))]

 data=raw.load_data('/home/mic/Videos/IBL/audio/noMouse/')
 ntrials=len(data)
 print(ntrials)
 r=linspace(0,len(nm)/float(fs),len(nm)) #x axis in seconds

 fig,ax=subplots()
 plot(r,nm,label='wav')

 shift_valve=3.09586000 #time difference of that wav file and the jasonable times (found by visual inspection of wav and valve_opens)
 shift_error=0.29133
 shift_onset_tone=0.3485
 
 height=5000

 for i in range(ntrials):
  if i==0: 
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][0]+shift_valve,-height,height,color='g',label='valve_opens')
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][1]+shift_valve,-height,height,color='r',label='valve_closes')
   ax.vlines(data[i]['behavior_data']['States timestamps']['closed_loop'][0][0]+shift_onset_tone+shift_valve,-height,height,color='y',label='onset_tone')
   ax.vlines(data[i]['behavior_data']['States timestamps']['error'][0][0]+shift_error+shift_valve,-height,height,color='k',label='error_tone')
  else:
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][0]+shift_valve,-height,height,color='g')
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][1]+shift_valve,-height,height,color='r')
   ax.vlines(data[i]['behavior_data']['States timestamps']['closed_loop'][0][0]+shift_onset_tone+shift_valve,-height,height,color='y')
   ax.vlines(data[i]['behavior_data']['States timestamps']['error'][0][0]+shift_error+shift_valve,-height,height,color='k')


 ylabel('audio amplitude [a.u]')
 xlabel('time [sec]')
 ax.legend(loc='best', numpoints=1, fancybox=True, shadow=True, fontsize=15).draggable()
 title('noMouse')
