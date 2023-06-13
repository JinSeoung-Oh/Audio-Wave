# check https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines
import torch
import torchaudio

torch.random.maual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_path = 'sample_path'

# https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines
bundle = torchaudio.piplelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

waveform, sample_rate = torchaudio.load(sample_path)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
  waveform = torchaudio.funtional.resample(waveform, sample_rate, bundle.sample_rate)
  
# if want extract feature from audio
with torch.inference_mode():
  features, _ = model.extract_features(waveform)
 
# Feature classfication
with torch.inference_mode():
  emission, _ = model(waveform)
labels=bundel.get_labels()


# Generating transcripts
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
      
decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
