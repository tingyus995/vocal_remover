import os
import random
import uuid
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile


class VocalRemovalSong(Dataset):

    def __init__(self,
                 media_path: str,
                 sr=44100,
                 vocal_r=True,
                 window_size=1024,
                 hop_len=768,
                 n_frames=128,
                 require_phase=False) -> None:
        super().__init__()

        self.window_size = window_size
        self.hop_len = hop_len
        self.n_frames = n_frames
        self.require_phase = require_phase

        self.signal, self.lr = librosa.load(media_path, sr=sr, mono=False)

        self.ins_sig = self.signal[0]
        self.vocal_sig = self.signal[1]

        if not vocal_r:
            self.ins_sig, self.vocal_sig = self.vocal_sig, self.ins_sig

        self.ins_features, self.ins_phase = self._convert_sig(self.ins_sig)
        self.vocal_features, self.vocal_phase = self._convert_sig(
            self.vocal_sig)

        total_frames = self.ins_features.shape[1]

        self.ranges = []
        split_points = list(range(0, total_frames, n_frames))

        for i in range(len(split_points) - 1):
            self.ranges.append(range(split_points[i], split_points[i + 1]))

    def _convert_sig(self, sig):
        d = librosa.stft(sig, n_fft=self.window_size, hop_length=self.hop_len)
        magnitude, phase = librosa.magphase(d)
        return magnitude, phase

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, idx):
        r = self.ranges[idx]
        if self.require_phase:
            return self.vocal_features[:, r], self.vocal_phase[:, r], self.ins_features[:, r], self.ins_phase[:, r]
        else:
            return self.vocal_features[:, r], self.ins_features[:, r]


class VocalRemovalSongDataset(Dataset):

    def __init__(self, root_dir: str, cache_dir: str, n_frames=30) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.n_frames = 30
        self.files = sorted(list(
            [f for f in os.listdir(root_dir) if f.endswith(".mp2")]))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        media_path = os.path.join(self.root_dir, self.files[index])

        # read from cache instead if there's any
        cache_path = os.path.join(self.cache_dir, f"{self.files[index]}.pickle") 
        if os.path.isfile(cache_path):
            print("using cache...")
            ds = torch.load(cache_path)
        else:
            print("not using cache...")
            ds = VocalRemovalSong(media_path, require_phase=False)

            # save to cache
            data = []

            for i in range(len(ds)):
                data.append(ds[i])

            temp_cache_path = os.path.join(self.cache_dir, str(uuid.uuid4()))
            torch.save(data, temp_cache_path)
            os.rename(temp_cache_path, cache_path)
            
        total_frames = len(ds)

        # randomly choose a start point
        begin_idx = random.randint(0, total_frames - 1 - self.n_frames)

        vocal_mags = []
        ins_mags = []

        for i in range(begin_idx, begin_idx + self.n_frames):
            vocal_mag, ins_mag = ds[i]
            vocal_mags.append(torch.from_numpy(vocal_mag).unsqueeze(0))
            ins_mags.append(torch.from_numpy(ins_mag).unsqueeze(0))

        return torch.cat(vocal_mags, dim=0), torch.cat(ins_mags, dim=0)


if __name__ == "__main__":

    ds = VocalRemovalSong("data/1_AVSEQ01.DAT.mp2")
    a, b = ds[0]

    print(a.shape)
    print(b.shape)
