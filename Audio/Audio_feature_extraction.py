import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# Spectrogram 
# A spectrogram is a picture of sound. 
# A spectrogram shows the frequencies that make up the sound, from low to high, and how they change over time

SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(sample_path)

n_fft = 1024
win_length = None
hop_length = 512

spectrogram = T.Spectrogram(
  n_fft = n_fft,
  win_length = win_length,
  hop_length = hop_length,
  center = True,
  pad_mode='reflect',
  power = 2.0)

spec = spectrogram(SPEECH_WAVEFORM)

# GriffinLim
# The Griffin-Lim Algorithm is a method for recovering a signal from its magnitude spectrogram.
torch.random.maual_seed(0)
n_fft=1024
win_length = None
hop_length = 512

spec = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
)(SPEECH_WAVEFORM)

griffin_lim = T.GriffinLim(
  n_fft = n_fft,
  win_length = win_length,
  hop_length = hop_length)

reconstructed_waveform = griffin_lim(spec)

# Mel Filter Bank
# Mel Filter Banks is a triangular filter bank that works 
# similar to the human ears perception of sound which is more discriminative 
# at lower frequencies and less discriminative at higher frequencies
n_fft = 256
n_mels = 64
sample_rate = 6000
mel_filters = F.melscale_fbanks(
  int(n_fft//2 + 1),
  n_mels = n_mels,
  f_min = 0.0,
  f_max = sample_rate/2.0,
  sample_rate = sample_rate,
  norm = 'slaney')


# MelSpectrogram
# A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
# mathematically speaking, is the result of some non-linear transformation of the frequency scale
n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk"
)

melspec = mel_spectrogram(SPEECH_WAVEFORM)

# MFCC
# MFCCs are a compact representation of the spectrum
n_fft = 2048
win_length = None
hop_length = 512
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    }
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)

# LFCC
# LFCCs are cepstral coefficients commonly used in speaker/speech recognition systems
n_fft = 2048
win_length = None
hop_length = 512
n_lfcc = 256

lfcc_transform = T.LFCC(
    sample_rate=sample_rate,
    n_lfcc=n_lfcc,
    speckwargs={
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
    },
)

lfcc = lfcc_transform(SPEECH_WAVEFORM)

# Pitch
pitch = F.detect_pitch_frequency(SPEECH_WAVEFORM, SAMPLE_RATE)

#Kaldi Pitch -  a pitch detection mechanism tuned for automatic speech recognition
pitch_feature = F.compute_kaldi_pitch(SPEECH_WAVEFORM, SAMPLE_RATE)
pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]
