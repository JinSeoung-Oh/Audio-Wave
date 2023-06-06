import torch
import torchaudio
import torchaudio.functional as F

import math
from Ipython.display import Audio

waveform1, sample_rate1 = torchaudio.load('./data/sample/wave/example.wav')

effects = [
  ['lowpass', '-1', '300'], #apply single-pole lowpass filter
  ['speed', '0.8'], #reduce the speed
  ['rate', f'{sample_rate1}'], #This only changes sample rate, so it is necessary to
                               # add `rate` effect with original sample rate after this.
  ['reverb', '-w']] ## Reverbration gives some dramatic feeling

waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform1, sample_rate1, effects)



## Simulating room reverberation <-- used to make clean audio sound as though it has been produced in a different environment
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.norm(rir, p=2)
speech, _ = torchaudio.load(SAMPLE_SPEECH)
augmented = F.fftconvolve(speech, rir)

## Adding background noise
speech, _ = torchaudio.load(SAMPLE_SPEECH)
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : speech.shape[1]]

snr_dbs = torch.tensor([20, 10, 3])
noisy_speeches = F.add_noise(speech, noise, snr_dbs)

snr_db, noisy_speech = snr_dbs[0], noisy_speeches[0:1] # SNR 20dB
snr_db, noisy_speech = snr_dbs[1], noisy_speeches[1:2] # SNR 10dB
snr_db, noisy_speech = snr_dbs[2], noisy_speeches[2:3] # SNR 3dB


## Applying codec to Tensor object
waveform, sample_rate = torchaudio.load(SAMPLE_SPEECH)

configs = [
    {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
    {"format": "gsm"},
    {"format": "vorbis", "compression": -1},
]
waveforms = []
for param in configs:
    augmented = F.apply_codec(waveform, sample_rate, **param)
    waveforms.append(augmented)
    

## Simulating a phone recoding 
sample_rate = 16000
original_speech, sample_rate = torchaudio.load(SAMPLE_SPEECH)

# Apply RIR <-- Room Impulse Response (RIR) RIR data. It can record just turn on your microphone and clap your hands.
rir_raw, sample_rate = torchaudio.load(SAMPLE_RIR)
rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.norm(rir, p=2)
rir_applied = F.fftconvolve(original_speech, rir)

# Add background noise
noise, _ = torchaudio.load(SAMPLE_NOISE)
noise = noise[:, : rir_applied.shape[1]]

snr_db = torch.tensor([8])
bg_added = F.add_noise(rir_applied, noise, snr_db)

# Apply filtering and change sample rate
filtered, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    bg_added,
    sample_rate,
    effects=[
        ["lowpass", "4000"],
        [
            "compand",
            "0.02,0.05",
            "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
            "-8",
            "-7",
            "0.05",
        ],
        ["rate", "8000"],
    ],
)


# Apply telephony codec
codec_applied = F.apply_codec(filtered, sample_rate2, format="gsm")
