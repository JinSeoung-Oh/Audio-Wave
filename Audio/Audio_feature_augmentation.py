import librosa
import torch
import torchaudio

sample_path = 'sample_path'

def get_sample(path, resample=None):
  effects = [['remix', '1']]
  if resample:
    effects.extend(
      [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)
  
def get_spectrogram(
    path,
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = get_sample(path)
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)
  
  
# SpecAugment
spec = get_spectrogram(sample_path, powr=None)
stretch = T.TimeStretch()
  
rate = 1.2
spec_ = stretch(spec, rate)
  
rate = 0.9
spec_=stretch(spec, rate)
  
  
# TimeMasking
torch.random.manual_seed(4)
spec = get_spectrogram(sample_path)
masking = T.TimeMasking(time_mask_param=80)
spec = masking(spec)
  
  
#FrequencyMasking
torch.random.manual_seed(4)
spec = get_spectrogram(sample_path)
masking = T.FrequencyMasking(freq_mask_param=80)
spec = masking(spec)
