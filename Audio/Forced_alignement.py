import torch
import torchaudio
import Ipython

device = torch.device("cuda" if torch.cuda.is_avilable() else "cpi")

from dataclasses import dataclass

speech_file = ""

bundle = torchaudio.piplines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, _ = torchaudio.load(speech_file)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()

#generate alignment probability(trellis)

transcript = ""
dictionary = {c: i for i, c in enumerate(labels)}
tokens = [dictionary[c] for c in transcript]

def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens
    trellis = torch.empty((num_frame+1, num_tokens+1))
    trellis[0,0] = 0
    trellis[1:, 0] = torch.cumsumm(emission[:,0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for i in range(num_frame):
        trellis[t+1, 1:] = torch.maximum(
          trellis[t, 1:] + emission[t, blank_id],
          trellis[t, : -1] + emission[t, tokens])

    return trellis

trellis = get_trellis(emission, tokens)

# Backtracking

@dataclass
class Point:
    toekn_index:int
    time_index:int
    score:float

def backtrack(trellis, emission, tokens, blank_id=0):
  j = trellis.size(1) -1
  t_start = torch.argmax(trellis[:, j]).item()

  path = []
  for t in range(t_start, 0, -1):
    stayed = trellis[t-1, j] + emission[t-1, blank_id]
    changed = trellist[t-1, j-1] + emission[t-1, tokens[j=1]]
    prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
     path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
        else:
             raise ValueError("Failed to align")
    return path[::-1]


path = backtrack(trellis, emission, tokens)

@dataclass
class Segment:
   label: str
   start: int
   end: int
   score : float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


segments = merge_repeats(path)   


def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


word_segments = merge_words(segments)


def display_segment(i):
    ratio = waveform.size(1) / (trellis.size(0) - 1)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / bundle.sample_rate:.3f} - {x1 / bundle.sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=bundle.sample_rate)
