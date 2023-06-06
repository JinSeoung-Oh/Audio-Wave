import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import math
import timeit
import librosa
import resampy
from Ipython.display import Audio, display

DEFAULT_OFFSET = 201

### helper function
def _get_log_freq(sample_rate, max_sweep_rate, offset):
   """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

   offset is used to avoid negative infinity `log(offset + x)`.

   """    
   start, stop = math.log(offset), math.log(offset+max_sweep_rate//2)
   return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double))-offset


def _get_inverse_log_freq(freq, sample_rate, offset):
  """Find the time where the given frequency is given by _get_log_freq"""
  half = sample_rate//2
  return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))
  
  
def get_freq_ticks(sample_rate, offset, f_max):
   # Given the original sample rate used for generating the sweep,
   # find the x-axis value where the log-scale major frequency values fall in
  times, freq = [], []
  for exp in range(2,5):
    for v in range(1, 10):
      f = v*10**exp
      if f < sample_rate //2:
        t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
        times.append(t)
        freq.append(f)
          
  return times, freq
  
  
def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
  max_sweep_rate = sample_rate
  freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
  delta = 2*math.pi*freq/sample_rate
  cummulative = torch.cumsum(delta, dim=0)
  signal = torch.sin(cummulative).unsqueeze(dim=0)
  
  return signal


def plot_sweep(waveform, sample_rate, title, max_sweep_rate=48000, offset=DEFAULT_OFFSET):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)
    plt.show(block=True)

    
## original wave
sample_rate = 48000
waveform = get_sine_sweep(sample_rate)

plot_sweep(waveform, sample_rate, title="Original Waveform")
Audio(waveform.numpy()[0], rate=sample_rate)


## downsample
resample_rate = 32000
resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)

plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
Audio(resampled_waveform.numpy()[0], rate=resample_rate)


## resampling with lowpass_filter_width <--  be used to control for the width of the filter to use to window the interpolation
sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, lowpass_filter_width=6)
plot_sweep(resampled_waveform, resample_rate, title="lowpass_filter_width=6")


## Rolloff <-- be represented as a fraction of the Nyquist frequency, which is the maximal frequency representable by a given finite sample rate
sample_rate = 48000
resample_rate = 32000

resampled_waveform = F.resample(waveform, sample_rate, resample_rate, rolloff=0.99)
plot_sweep(resampled_waveform, resample_rate, title="rolloff=0.99")


## window function <-- torchaudio's resample uses the Hann window filter + Kaiser window
resampled_waveform = F.resample(waveform, sample_rate, resample_rate, resampling_method="sinc_interp_kaiser")
plot_sweep(resampled_waveform, resample_rate, title="Kaiser Window Default")

