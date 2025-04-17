import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from alive_progress import alive_bar


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

input_folder = "/home/max/Results/TransformerAutoencoder"
print("loading model")
# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerAutoencoder().to(device)
model.load_state_dict(torch.load('transformer_autoencoder_zzh_huber.pth', map_location=device)) #Change model for other half of data
model.eval()
print("model loaded")
# --- Run prediction on a batch of files ---
with alive_bar(33, title="Processing videos") as bar:
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().startswith('synth'):
                if 'zzh' in root:  # You can modify this condition as needed
                    print(f"Skipping file {file} because it contains 'zzh' in the path.")
                    continue  # Skip this file

                #if 'dkk' in root:  # You can modify this condition as needed
                 #   print(f"Skipping file {file} because it contains 'dkk' in the path.")
                  #  continue  # Skip this file
                coarseDataPath = os.path.join(root, file)
                directory_path = os.path.dirname(coarseDataPath)

                coarseData = np.load(coarseDataPath)  # shape: [32, 250]
                coarseData = np.concatenate([coarseData, np.zeros((32, 1))], axis=1)
                coarseData_tensor = torch.tensor(coarseData, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 32, 250]
                print("prediction starting")
                with torch.no_grad():
                    reconstructed = model(coarseData_tensor)  # [1, 32, 250]
                    reconstructed = reconstructed.squeeze(0).cpu().numpy()  # [32, 250]
                    reconstructed = (reconstructed - np.min(reconstructed))/(np.max(reconstructed)-np.min(reconstructed))

                outputPath = os.path.join(directory_path, "reconstructed.npy")
                np.save(outputPath, reconstructed)
                bar()

