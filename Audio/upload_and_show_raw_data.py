import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
from botocore import UNSIGNED
from botocore.config import Config
from Ipython.display import Audio

frame_offset, num_frames = 16000, 16000
wave_path = './dataset/sample.wav'
waveform, sample_rate = torchaudio.load(wave_path, frame_offset = frame_offset, num_frames = num_grames)

def show_waveform(waveform, sample_rate):
  waveform = waveform.numpy()
  
  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate
  
  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels>1:
      axes[c].set_ylabel(f'Channel {c+1}')
  figure.suptitle('waveform')
  plt.show(block=False)
  
  def plot_spectram(waveform, sample_rate):
    waveform = waveform.numpy()
    
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplot(num_channels, 1)
    if num_channels == 1:
      axes = [axes]
    for c in range(num_channels):
      axes[c].specgram(waveform[c], Fs=sample_rate)
      if num_channels > 1:
        axes[c].set_ylabel(f'channel {c+1}')
    figure.subtitle('Spectrogram')
    plt.show(block=False)
    
    
    buffer_.seek(0)
    torchaudio.save(buffer_, waveform, sample_rate, format='wav')
