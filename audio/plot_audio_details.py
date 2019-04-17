from pylab import *
ion()
from scipy.io import wavfile
from ibllib.io import raw_data_loaders as raw
from scipy.signal import spectrogram,detrend
import os
#from scipy.fftpack import fft
from numpy.fft import rfft
from random import sample 
import pandas as pd

matplotlib.rcParams.update({'font.size': 25})

def AverageSpectra(nm,savePath,name,seg_length,fs):
 ''' 
 Input: nm being the wav time series
 Output: Power spectral density averaged across non-overlapping 6 sec segments
 '''

 ioff()

 for i in range(int(len(nm)/(fs*60*seg_length))):
  
  data=nm[i*int(fs*60*seg_length):(i+1)*int(fs*60*seg_length)]
  Pxx, freqs, _, _ = specgram(data,NFFT=2**12, Fs=fs, cmap=plt.get_cmap('plasma'), 
                                   scale='dB', vmin=-80, vmax=0)
  if i==0:
   ps=mean(20 * log10(Pxx),axis=1)
  else:
   ps+=mean(20 * log10(Pxx),axis=1)
  del data
  

 
 ps_mean=array(ps)/(float(len(nm)/(fs*60*seg_length)))
 d={'spectrum':ps_mean,'frequencies':freqs} #average 
 save('%s/spectrum_%s.npy' %(savePath,name),d)
 ion()

def ExampleImage(data,fs,i,name,nm,seg_length,savePath):
 ''' 
 Input: data being a 6 sec time series, i'th segment, fs its sampling rate in Hz
 Output: png image saved for 20 uniformly sampled 6 sec segments, showing spectrogram, wav and power spectral density 
 '''
 
 ioff()
 fig=figure(figsize=(40,25))
 ax= subplot(2,2,1)

 Pxx, freqs, bins, im = ax.specgram(data,NFFT=2**12, Fs=fs, cmap=plt.get_cmap('plasma'), 
                                   scale='dB', vmin=-80, vmax=0)
 t = np.linspace(0, len(data)/fs, num=len(data))
 
 ax.set(ylabel="frequency [Hz]", xlabel='time [sec]', title=name[:-4]+', segment '+str(i) + ' of %s' %int(len(nm)/(fs*60*seg_length)) + ', seg length = %s [min] ' %seg_length, xlim=[0, t.max()])

 ax3 = subplot(2,2,2,sharey=ax)
 ax3.plot(mean(20 * log10(Pxx),axis=1),freqs)
 ax3.set(xlabel="Intensity [dB], averaged over segment",ylabel="Frequency [Hz]")
 fig.colorbar(im).set_label('intensity [dB]')

 ax2 = subplot(2,2,3,sharex=ax)
 ax2.plot(t,data,'k',linewidth=0.4)
 ax2.set(ylabel="wav amplitude [a.u.]")
 ax2.set(xlabel="time [sec]")
   
 plt.tight_layout()
 fig.savefig('%s/%s_%s.png' %(savePath,name,i))
 
 del data,t,Pxx, freqs, bins, im,fig, ax
 close()
 ion()

def SaveExampleImagesAndSpectra():

 #wavs location source folder
 sourcePath='/home/mic/Audio/IdentifyEvents'
    
 #folder to save results
 savePath='/home/mic/Audio/IdentifyEvents/Res_mike'

 wavs=[f for f in os.listdir(sourcePath) if f.endswith(".wav")]
 seg_length= 1 # was 0.1 [min]

 for wav in wavs:

  fs, nm = wavfile.read('%s/%s' %(sourcePath,wav))

  if len(nm)==0:
   print("wav couldn't be read, %s" %wav)
   continue

  AverageSpectra(nm,savePath,wav,seg_length,fs)

  print("%s spectra saved, now images" %wav)
  if 1==1:# len(nm)/(fs*60*seg_length) < 20:
   t=range(int(len(nm)/(fs*60*seg_length)))
  else: 
   t=sample(range(int(len(nm)/(fs*60*seg_length))),20)  

  for i in t: 
   data=nm[i*int(fs*60*seg_length):(i+1)*int(fs*60*seg_length)]
   ExampleImage(data,fs,i,wav,nm,seg_length,savePath)
  del nm
 

def plot_average_spectrum():

 #path to folder with npy files that contain spectra
 sourceP='/mic/Audio/Audio-analysis-20190219T154018Z-001/Audio-analysis/large_results'

 wavs=os.listdir(sourcePath)


 fig,ax=subplots()
 d=load(path_to_npy_file)
 ax.plot(d.item().get('spectrum'),d.item().get('frequencies'))
 ax.set(xlabel="Intensity [dB]",ylabel="Frequency [Hz]")
 tight_layout()

 ax.fill_between(arange(20),M+SM/2,M-SM/2, alpha=0.1,color=cols[measures.index(measure)])



def save_spectra():

 #INPUT: wavs location
 sourcePath='/home/mic/Audio/karolina/wavs'

   #/home/mic/Videos/IBL/audio/Audio-analysis-20190219T154018Z-001/Audio-analysis/large' 
 #OUTPUT: plots of spectra + wavs (set location)
 savePath='/home/mic/Audio/karolina/images'

  #'/home/mic/Videos/IBL/audio/Audio-analysis-20190219T154018Z-001/spectra_large' 

 ioff()
 wavs=os.listdir(sourcePath)[:1]

 Peaks=[]
 seg_length= 0.1 # [min]


 ps_mean=array(ps)/(int(len(nm)/(fs*60*seg_length))) #average 

 PSdiffs=[]
 for wav in wavs:

  fs, nm = wavfile.read('%s/%s' %(sourcePath,wav)) 
  # segment length in minutes


  for i in range(int(len(nm)/(fs*60*seg_length))): 
   fig=figure(figsize=(15,7))
   ax= subplot(2,1,1)

   data=nm[i*int(fs*60*seg_length):(i+1)*int(fs*60*seg_length)]

   Pxx, freqs, bins, im = ax.specgram(data,NFFT=2**12, Fs=fs, cmap=plt.get_cmap('plasma'), 
                                     scale='dB', vmin=-80, vmax=0)
   t = np.linspace(0, len(data)/fs, num=len(data))
   fig.colorbar(im).set_label('Intensity (dB)')
   ax.set(ylabel="Frequency (Hz)", xlabel='Time (s)', title=wav[:-4]+', segment '+str(i) + ' of %s' %int(len(nm)/(fs*60*seg_length)) + ', seg length = %s [min] ' %seg_length, xlim=[0, t.max()])
 
   ax2 = subplot(2,1,2,sharex=ax)
   ax2.plot(t,data,'k',linewidth=0.4)
   fig.colorbar(im).set_label('Intensity (dB)')
   ax2.set(ylabel="wav amplitude [a.u.]")

   plt.tight_layout()

   fig.subplots_adjust(top=0.95,
bottom=0.055,
left=0.055,
right=0.99,
hspace=0.227,
wspace=0.2)

   fig.savefig('%s/%s_%s.png' %(savePath,wav,i))
   del data,t,Pxx, freqs, bins, im,fig, ax
   close()

 ion()

 return PSdiffs 

def plot_audio_events():
 fs, nm = wavfile.read('/home/mic/Audio/IdentifyEvents/BehaviourTrain__CCU_Mainen_Rig1_2019-02-20_56e6d293_ZM_1091_firstMinute.wav')
 nm=nm[:int(len(nm))]

 data=raw.load_data('/home/mic/Audio/IdentifyEvents/')
 ntrials=16 #len(data)
 print(ntrials)
 r=linspace(0,len(nm)/float(fs),len(nm)) #x axis in seconds

 fig,ax=subplots()
 plot(r,nm,label='wav')

 shift=4.667413 #time difference of that wav file and the jasonable times (found by visual inspection of wav and valve_opens)
 
 height=5000

 for i in range(ntrials):
  if i==0: 
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][0]+shift,-height,height,color='g',label='valve_opens')
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][1]+shift,-height,height,color='r',label='valve_closes')
   ax.vlines(data[i]['behavior_data']['States timestamps']['closed_loop'][0][0]+shift,-height,height,color='y',label='onset_tone')
   ax.vlines(data[i]['behavior_data']['States timestamps']['error'][0][0]+shift,-height,height,color='k',label='error_tone')
  else:
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][0]+shift,-height,height,color='g')
   ax.vlines(data[i]['behavior_data']['States timestamps']['reward'][0][1]+shift,-height,height,color='r')
   ax.vlines(data[i]['behavior_data']['States timestamps']['closed_loop'][0][0]+shift,-height,height,color='y')
   ax.vlines(data[i]['behavior_data']['States timestamps']['error'][0][0]+shift,-height,height,color='k')


 ylabel('audio amplitude [a.u]')
 xlabel('time [sec]')
 ax.legend(loc='best', numpoints=1, fancybox=True, shadow=True, fontsize=15).draggable()
