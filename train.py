import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VocalRemovalSong, VocalRemovalSongDataset
from model import Net

if __name__ == "__main__":
    device = "cuda"
    num_epochs = 1001
    data_path = "data"
    n_frames = 8
    ds = VocalRemovalSongDataset(data_path, "cache", n_frames=n_frames)
    loader = DataLoader(ds, batch_size=5, pin_memory=True, num_workers=2, shuffle=True)
    model = Net().to(device)
    # model.load_state_dict(torch.load("ckpt/model_e99.ckpt"))
    optimizer = optim.Adam(model.parameters())
    writer = SummaryWriter()

    criterion = nn.L1Loss()
    n_iters = 0
    begin_epoch = 1

    for epoch in range(begin_epoch, begin_epoch + num_epochs):
        print(f"epoch = {epoch}")
        for vocal, ins in loader:
            n_iters += 1

            # vocal = vocal[0]
            # ins = ins[0]
            # (b_size, n_frames, 513, 128)
            b_size = vocal.size(0)

            vocal = vocal.reshape(-1, 513, 128)
            ins = ins.reshape(-1, 513, 128)
            

            vocal.unsqueeze_(1)
            ins.unsqueeze_(1)

            vocal_gpu = vocal[:, :, 1:].to(device)
            ins_gpu = ins[:, :, 1:].to(device)

            optimizer.zero_grad()
            preds = model(vocal_gpu)
            loss = criterion(preds, ins_gpu)
            writer.add_scalar("loss", loss, n_iters)
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0:
            torch.save(model.state_dict(), f"ckpt/model_e{epoch}.ckpt")
    torch.save(model.state_dict(), f"ckpt/model_e{epoch}.ckpt")
