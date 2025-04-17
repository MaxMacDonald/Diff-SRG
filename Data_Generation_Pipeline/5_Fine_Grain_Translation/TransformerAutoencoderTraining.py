import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim=250, d_model=256, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, src):
        # src shape: (batch_size, 32, 250)
        src_emb = self.embedding(src)  # -> (batch_size, 32, d_model)
        src_emb = self.pos_encoder(src_emb)

        memory = self.encoder(src_emb)

        # Assume auto-regressive decoding with the same input shape
        tgt = src_emb  # Using same input for reconstruction
        output = self.decoder(tgt, memory)

        return self.output_layer(output)  # -> (batch_size, 32, 250)

class RadarDataset(Dataset):
    def __init__(self, coarse, fine):
        self.coarse_data = coarse  # shape: [N, 32, 250]
        self.fine_data = fine     # shape: [N, 32, 250]

    def __len__(self):
        return len(self.coarse_data)

    def __getitem__(self, idx):
        coarse = torch.tensor(self.coarse_data[idx], dtype=torch.float32)
        fine = torch.tensor(self.fine_data[idx], dtype=torch.float32)
        return coarse, fine



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)
criterion = nn.SmoothL1Loss()

### Load Dataset ###
synth_dir = "/home/max/Results/intensitymapping"
gt_dir = "/home/max/mastersProject/Midas/doppler_data/radar_data"

person = ['dkk']
angle = ['0', '45', '90']

gt_list = []
synth_list = []

for p in person:
    for a in angle:
        activity = sorted(os.listdir(os.path.join(synth_dir, p, a)))
        for active in activity:
            real_data = np.load(os.path.join(gt_dir, p, a, active, "doppler_gt.npy"))
            synth_data = np.load(os.path.join(synth_dir, p, a, active, "synthDopplerData.npy"))
            synth_data = np.concatenate([synth_data, np.zeros((32, 1))], axis=1)
            gt_list.append(real_data)
            synth_list.append(synth_data)

# Convert to final shape: (n, 32, 250)
gt_array = np.stack(gt_list)       # shape: (n, 32, 250)
synth_array = np.stack(synth_list) # shape: (n, 32, 250)


print("Loaded coarse shape:", synth_array.shape)
dataset = RadarDataset(synth_array, gt_array)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


for epoch in range(5000):
    model.train()
    for coarse_data, fine_data in dataloader:
        coarse_data, fine_data = coarse_data.to(device), fine_data.to(device)
        output = model(coarse_data)
        loss = criterion(output, fine_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_lr()[0]:.6f}")



torch.save(model.state_dict(), 'transformer_autoencoder_dkk_huber.pth')
print("Model saved as 'transformer_autoencoder_dkk_huber.pth'")

#torch.save(model.state_dict(), 'transformer_autoencoder_zzh_huber.pth')
#print("Model saved as 'transformer_autoencoder_zzh_huber.pth'")
