
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from diarization import SpeakerCNN
import torch
import librosa
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ROOT = "/"
os.makedirs(ROOT, exist_ok=True)



class TestSpeakerDataset(Dataset):
    def __init__(self, ds_full, spk2idx,
                 sr=16000, chunk_sec=2.0,
                 n_mels=64, win_length=400, hop_length=160):
        self.ds_full   = ds_full
        self.sr        = sr
        self.chunk_len = int(chunk_sec * sr)
        self.n_mels    = n_mels
        self.win_length= win_length
        self.hop_length= hop_length
        self.spk2idx   = spk2idx
        self.mel_basis = librosa.filters.mel(sr=self.sr,
                                             n_fft=self.win_length,
                                             n_mels=self.n_mels)
        
        self.keep_idxs = [i for i in range(len(self.ds_full))
                          if self.ds_full[i][3] in spk2idx]

    def __len__(self):
        return len(self.keep_idxs)

    def __getitem__(self, idx):
        real_i = self.keep_idxs[idx]
        wav_tensor, orig_sr, _, spk, _, _ = self.ds_full[real_i]
        wav = wav_tensor.squeeze(0).numpy()
        if orig_sr != self.sr:
            wav = librosa.resample(wav, orig_sr, self.sr)
        
        if len(wav) < self.chunk_len:
            wav = np.pad(wav, (0, self.chunk_len - len(wav)))
        else:
            wav = wav[:self.chunk_len]
        
        S = np.abs(librosa.stft(wav,
                                n_fft=self.win_length,
                                hop_length=self.hop_length,
                                win_length=self.win_length))**2
        mel     = self.mel_basis.dot(S)
        log_mel = np.log1p(mel).astype(np.float32)
        log_mel = (log_mel - log_mel.mean())/(log_mel.std()+1e-6)
        x = torch.from_numpy(log_mel)[None]              
        y = self.spk2idx[spk]
        return x, y


spk2idx = torch.load('spk2idx.pth')
idx2spk = {v:k for k,v in spk2idx.items()}

model = SpeakerCNN(len(spk2idx)).to(device)
model.load_state_dict(torch.load('spd.pth', map_location=device))
model.eval()


ds_test_full = torchaudio.datasets.LIBRISPEECH(ROOT,
                                               url='test-clean',
                                               download=True)
test_ds = TestSpeakerDataset(ds_test_full, spk2idx)

def collate_det(batch):
    X = torch.stack([b[0] for b in batch]).to(device)
    Y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    return X, Y

test_loader = DataLoader(test_ds,
                         batch_size=64,
                         shuffle=False,
                         collate_fn=collate_det,
                         num_workers=4,
                         pin_memory=(device.type=='cuda'))


all_preds, all_labels, all_embs = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        emb, logits = model(X)
        preds = logits.argmax(dim=1)
        all_embs.append(emb.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_embs  = np.vstack(all_embs)
all_preds = np.hstack(all_preds)
all_labels= np.hstack(all_labels)


cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(spk2idx))))
disp = ConfusionMatrixDisplay(cm, display_labels=[idx2spk[i] for i in range(len(spk2idx))])
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, xticks_rotation=90, cmap='Blues')
plt.title("Test-clean Speaker Classification Confusion Matrix")
plt.tight_layout()


pca = PCA(n_components=2)
emb2d = pca.fit_transform(all_embs)
plt.figure(figsize=(8,6))
for spk_idx in range(len(spk2idx)):
    mask = (all_labels==spk_idx)
    plt.scatter(emb2d[mask,0], emb2d[mask,1], s=10, label=idx2spk[spk_idx])
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', ncol=2, fontsize='small')
plt.title("PCA of Test-clean Embeddings")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()