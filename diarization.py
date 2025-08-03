import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

#SPEAKER EMBEDDING + CLASSIFICATION MODEL
class SpeakerCNN(nn.Module):
    def __init__(self, n_speakers):
        super().__init__()
        self.conv1  = nn.Conv2d(1,16, kernel_size=5, padding=2)
        self.bn1    = nn.BatchNorm2d(16)
        self.conv2  = nn.Conv2d(16,32, kernel_size=5, padding=2)
        self.bn2    = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(64)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc_emb = nn.Linear(64,128)
        self.fc_clf = nn.Linear(128,n_speakers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn2(self.conv2(x))); x = F.max_pool2d(x,2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)   
        emb    = F.relu(self.fc_emb(x))           
        logits = self.fc_clf(emb)
        return emb, logits

# VOICE ACTIVITY DETECTION + WINDOWING
def voice_activity_detection(wav, sr=16000, frame_dur=0.025, frame_shift=0.01, energy_thresh=0.02):
    fl = int(frame_dur*sr)
    fs = int(frame_shift*sr)
    pad = (fs - (len(wav)-fl)%fs) % fs
    wav_p = np.concatenate([wav, np.zeros(pad)])
    frames = librosa.util.frame(wav_p, frame_length=fl, hop_length=fs)
    energy = (frames**2).mean(axis=0)
    vad = energy > energy_thresh
    return vad, fl, fs

def segment_windows(wav, sr, vad, fl, fs, win_dur=1.0, win_shift=0.5):
    win_len = int(win_dur*sr)
    win_hop = int(win_shift*sr)
    segments, times = [], []
    for start in range(0, len(wav)-win_len+1, win_hop):
        end = start + win_len
        f0 = max(0, (start-fl)//fs)
        f1 = min(len(vad), (end-fl)//fs)
        if f1<=f0 or vad[f0:f1].mean() < 0.5:
            continue
        segments.append(wav[start:end])
        times.append((start/sr, end/sr))
    return segments, times

def extract_log_mel(seg, sr=16000, n_mels=64, win_length=400, hop_length=160):
    S = np.abs(librosa.stft(seg, n_fft=win_length,
                             hop_length=hop_length,
                             win_length=win_length))**2
    mel_basis = librosa.filters.mel(sr, win_length, n_mels)
    mel       = mel_basis.dot(S)
    log_mel   = np.log1p(mel).astype(np.float32)
    log_mel   = (log_mel - log_mel.mean())/(log_mel.std()+1e-6)
    return torch.from_numpy(log_mel)[None]  # (1,n_mels,T)

#DIARIZATION + SPEAKER ID
def diarize_and_identify_array(wav, sr, model, spk2idx,
                               threshold=0.6, win_dur=1.0, win_shift=0.5):
    device = next(model.parameters()).device
    inv_map = {v:k for k,v in spk2idx.items()}

    vad, fl, fs = voice_activity_detection(wav, sr)
    segs, times = segment_windows(wav, sr, vad, fl, fs, win_dur, win_shift)

    labels, prev_e = [], None
    with torch.no_grad():
        for seg in segs:
            x = extract_log_mel(seg, sr)[None].to(device)     # (1,1,n_mels,T)
            _, logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            pmax, idx = probs.max().item(), probs.argmax().item()
            labels.append(inv_map[idx] if pmax>=threshold else "unknown")

    # merge into contiguous speaker turns
    result = []
    if not labels:
        return result
    cur_lbl, cur_s = labels[0], times[0][0]
    for lbl, (s,e) in zip(labels, times):
        if lbl != cur_lbl:
            result.append({"start": cur_s, "end": prev_e, "speaker": cur_lbl})
            cur_lbl, cur_s = lbl, s
        prev_e = e
    result.append({"start": cur_s, "end": prev_e, "speaker": cur_lbl})
    return result