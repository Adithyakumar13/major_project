import sys
import os
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.physics_modules import PhysicsEncoder

ENCODER_PATH = os.path.join(BASE, 'checkpoints', 'physics_encoder.pt')
DATA_PATH = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')

# load data
data = np.load(DATA_PATH)
obs = torch.tensor(data['obs'], dtype=torch.float32)

# load encoder
encoder = PhysicsEncoder()
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location="cpu"))
encoder.eval()

params = obs[:1000, 111:117]

with torch.no_grad():
    embedding = encoder(params)

print("Encoder diagnostics")
print("-------------------")
print(f"Samples           : {len(params)}")
print(f"Embedding shape   : {embedding.shape}")
print(f"Mean              : {embedding.mean():.4f}")
print(f"Std               : {embedding.std():.4f}")
print(f"Min               : {embedding.min():.4f}")
print(f"Max               : {embedding.max():.4f}")

# variance of each embedding dimension
print("\nPer-dimension std:")
for i in range(embedding.shape[1]):
    print(f"dim {i}: {embedding[:, i].std():.4f}")

# compare two very different physics settings
with torch.no_grad():
    emb1 = encoder(obs[0:1, 111:117])
    emb2 = encoder(obs[500:501, 111:117])

distance = torch.norm(emb1 - emb2).item()

print(f"\nDistance between two sample embeddings: {distance:.4f}")