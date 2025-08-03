

import os
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import nn
from main import SpeakerCNN
from torch.utils.data import Dataset, DataLoader, Subset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ROOT = "/"
os.makedirs(ROOT, exist_ok=True)



def prepare_librispeech_dataset(root,
                                n_speakers=40,
                                min_utts=20,
                                seed=42):
    train_full = torchaudio.datasets.LIBRISPEECH(root, url='train-clean-360', download=True)
    ds_val_full = torchaudio.datasets.LIBRISPEECH(root, url='dev-clean', download=True)

    counts = {}
    for i in range(len(train_full)):
        _, _, _, spk, _, _ = train_full[i]
        counts[spk] = counts.get(spk, 0) + 1

    good_spks = sorted(
        [s for s,c in counts.items() if c>=min_utts],
        key=lambda s: counts[s],
        reverse=True
    )[:n_speakers]
    spk2idx = {spk:idx for idx,spk in enumerate(good_spks)}

    all_idxs = [i for i in range(len(train_full)) if train_full[i][3] in good_spks]
    random.seed(seed)
    random.shuffle(all_idxs)
    cut = int(0.8 * len(all_idxs))
    train_subset = Subset(train_full, all_idxs[:cut])
    val_subset   = Subset(train_full, all_idxs[cut:])

    return train_subset, val_subset, spk2idx


class LibriSpeechSpeakerDataset(Dataset):
    def __init__(self, subset, spk2idx,
                 sr=16000, chunk_sec=2.0,
                 n_mels=64, win_length=400, hop_length=160):
        self.subset     = subset
        self.sr         = sr
        self.chunk_len  = int(chunk_sec * sr)
        self.n_mels     = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.spk2idx    = spk2idx
        
        self.mel_basis = librosa.filters.mel(sr=self.sr,
                                             n_fft=self.win_length,
                                             n_mels=self.n_mels)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        wav_tensor, orig_sr, _, spk, _, _ = self.subset[idx]
        wav = wav_tensor.squeeze(0).numpy()
        if orig_sr != self.sr:
            wav = librosa.resample(wav, orig_sr, self.sr)

        
        if len(wav) < self.chunk_len:
            wav = np.pad(wav, (0, self.chunk_len - len(wav)))
        else:
            start = random.randint(0, len(wav) - self.chunk_len)
            wav = wav[start:start + self.chunk_len]

        
        S = np.abs(librosa.stft(wav,
                                n_fft=self.win_length,
                                hop_length=self.hop_length,
                                win_length=self.win_length))**2
        mel = self.mel_basis.dot(S)
        log_mel = np.log1p(mel).astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

        x = torch.from_numpy(log_mel)[None]      
        y = self.spk2idx[spk]
        return x, y

def collate_fn(batch):
    X, Y = zip(*batch)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

def train_speaker_model(train_ds, val_ds, n_speakers,
                        epochs=15, batch_size=64, lr=3e-4):
    model  = SpeakerCNN(n_speakers).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    ce     = nn.CrossEntropyLoss()

    tr_loader = DataLoader(train_ds,
                           batch_size=batch_size,
                           shuffle=True,
                           collate_fn=collate_fn,
                           pin_memory=(device.type=='cuda'))
    val_loader= DataLoader(val_ds,
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=collate_fn,
                           pin_memory=(device.type=='cuda'))

    for ep in range(1, epochs+1):
        
        model.train()
        tot_l, tot_acc, n = 0,0,0
        for X,y in tr_loader:
            
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, logits = model(X)
            loss = ce(logits,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_l   += loss.item() * X.size(0)
            tot_acc += (logits.argmax(1)==y).sum().item()
            n       += X.size(0)

        print(f"Epoch {ep}/{epochs}  "
              f"train_loss={tot_l/n:.3f} train_acc={tot_acc/n:.3f}", end='  ')

        
        model.eval()
        v_l, v_acc, m = 0,0,0
        with torch.no_grad():
            for X,y in val_loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                _, logits = model(X)
                loss = ce(logits,y)
                v_l    += loss.item() * X.size(0)
                v_acc  += (logits.argmax(1)==y).sum().item()
                m      += X.size(0)

        print(f"val_loss={v_l/m:.3f} val_acc={v_acc/m:.3f}")

    return model



ROOT = "/content/sample_data/LibriSpeech"
os.makedirs(ROOT, exist_ok=True)

train_ds, val_ds, spk2idx = prepare_librispeech_dataset(
    ROOT, n_speakers=40, min_utts=20
)
torch.save(spk2idx, 'spk2idx.pth')
train_ds = LibriSpeechSpeakerDataset(train_ds, spk2idx)
val_ds   = LibriSpeechSpeakerDataset(val_ds, spk2idx)

model = train_speaker_model(
    train_ds, val_ds, n_speakers=len(spk2idx),
    epochs=15, batch_size=512, lr=3e-4
)

torch.save(model.state_dict(), 'spd.pth')