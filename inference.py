import librosa
import soundfile
import torch
from torch.utils.data import DataLoader

from model import Net
from dataset import VocalRemovalSong

if __name__ == "__main__":
    media_path = "audio.wav"

    ds = VocalRemovalSong(media_path, require_phase=True)
    loader = DataLoader(ds, batch_size=10, shuffle=False, pin_memory=True, num_workers=2)
    device = "cuda"

    net = Net().to(device)
    net.load_state_dict(torch.load("ckpt/model_e50.ckpt"))

    spectrum = []

    net.eval()

    for vocal_mag, vocal_phase, ins_mag, ins_phase in loader:

        vocal_mag.unsqueeze_(1)
        vocal_mag_gpu = vocal_mag[:, :, 1:].to(device)

        with torch.no_grad():
            preds_mag = net(vocal_mag_gpu)

        preds_mag_cpu = preds_mag.cpu()
        vocal_mag[:, :, 1:] = preds_mag_cpu

        # vocal_mag: (b_size, 1, 513, 128)
        spectrum.append((vocal_mag[:, 0] * vocal_phase).permute(1, 0, 2).reshape(513, -1))
        # (b_size, 513, 128)
        # spectrum.append((vocal_mag * vocal_phase).permute(1, 0, 2).reshape(513, -1))
    
    # num_frames = len(ds)
    # for i in range(num_frames):
    #     vocal_mag, vocal_phase, _, _ = ds[i]
    #     spectrum.append(torch.tensor(vocal_mag * vocal_phase))
    
    print(spectrum[0].shape)
    d = torch.cat(spectrum, dim=1)
    print(d)
    d_np = d.numpy()

    signal = librosa.istft(d_np, hop_length=768, win_length=1024)
    soundfile.write("result.wav", signal, 44100)


