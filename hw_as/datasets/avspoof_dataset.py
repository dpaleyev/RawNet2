import numpy as np
import torchaudio
from torch.utils.data import Dataset
from random import shuffle

class AVSpoofDataset(Dataset):
    def __init__(self, data_dir: str, slice: str, limit: int = None,  max_seq_len: int = 64000, **kwargs):
        super().__init__()
        self.max_seq_len = max_seq_len

        proto_path = f"{data_dir}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{slice}.{'trn' if slice == 'train' else 'trl'}.txt"
        flac_dir = f"{data_dir}/ASVspoof2019_LA_{slice}/flac"

        self.data = []

        with open(proto_path, 'r') as f:
            for line in f.readlines():

                line = line.strip().split()
                name = line[1]
                label = 1 if line[-1] == "spoof" else 0
                flac_path = f"{flac_dir}/{name}.flac"
                self.data.append({'flac_path': flac_path, 'label': label})
        
        if limit is not None:
            shuffle(self.data)
            self.data = self.data[:limit]
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        flac_path = data['flac_path']
        label = data['label']

        wav, sample_rate = torchaudio.load(flac_path)
        audio = wav[0:1, :]
        if audio.shape[-1] < self.max_seq_len:
            audio = audio.repeat(1, self.max_seq_len // audio.shape[-1] + 1)
        audio = audio[:, :self.max_seq_len]
        return {
            'audio': audio,
            'label': label
        }
