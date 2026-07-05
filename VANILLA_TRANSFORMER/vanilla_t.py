import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# --- CONFIGURATIONS ---
input_dir = Path('/kaggle/input/datasets/hardik07ai/thinkbook-embeddings')
output_dir = Path('/kaggle/working')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 30

# (Reuse master data pooling logic, path translations and PyTorchKFoldDataset block from Script 1)

class VanillaTransformerModel(nn.Module):
    def __init__(self, input_dim=1280, embed_dim=256, num_heads=8, num_layers=5):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, 
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dense_shared = nn.Linear(embed_dim, 64)
        self.gelu = nn.GELU()
        self.binary_head = nn.Linear(64, 1)
        self.multiclass_head = nn.Linear(64, 10)

    def forward(self, x, head='binary'):
        x = self.projection(x)
        out = self.transformer(x)
        pooled = torch.mean(out, dim=1) # Global Average Pooling
        shared = self.gelu(self.dense_shared(pooled))
        if head == 'binary':
            return torch.sigmoid(self.binary_head(shared)).squeeze(-1)
        return self.multiclass_head(shared)

# --- RUN LOOP ---
model = VanillaTransformerModel().to(device)
criterion_bin, criterion_mul = nn.BCELoss(), nn.CrossEntropyLoss()
opt_bin, opt_mul = optim.Adam(model.parameters(), lr=1e-4), optim.Adam(model.parameters(), lr=1e-4)

TOTAL_SAMPLES = len(full_master_df)
fold_size = int(TOTAL_SAMPLES / 10)

print("\nLaunching Shuffled 10-Fold Vanilla Transformer CV Engine...")
for epoch in range(EPOCHS):
    shuffled_master = full_master_df.sample(frac=1.0).reset_index(drop=True)
    val_fold_idx = np.random.choice(range(10))
    test_fold_idx = np.random.choice([i for i in range(10) if i != val_fold_idx])
    
    val_start, val_end = val_fold_idx * fold_size, (val_fold_idx + 1) * fold_size
    test_start, test_end = test_fold_idx * fold_size, (test_fold_idx + 1) * fold_size
    
    epoch_val_df = shuffled_master.iloc[val_start:val_end]
    epoch_test_df = shuffled_master.iloc[test_start:test_end]
    train_indices = [i for i in range(TOTAL_SAMPLES) if not (val_start <= i < val_end or test_start <= i < test_end)]
    epoch_train_df = shuffled_master.iloc[train_indices]
    
    loader_bin = DataLoader(PyTorchKFoldDataset(epoch_train_df, is_multiclass=False), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_mul = DataLoader(PyTorchKFoldDataset(epoch_train_df, is_multiclass=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model.train()
    for X_b, y_b in loader_bin:
        X_b, y_b = X_b.to(device), y_b.to(device)
        opt_bin.zero_grad(); loss = criterion_bin(model(X_b, head='binary'), y_b); loss.backward(); opt_bin.step()
    for X_m, y_m in loader_mul:
        X_m, y_m = X_m.to(device), y_m.to(device)
        opt_mul.zero_grad(); loss = criterion_mul(model(X_m, head='multiclass'), y_m); loss.backward(); opt_mul.step()

# --- EVALUATION AND EXPORTS ---
model.eval()
with torch.no_grad():
    val_ds_b = PyTorchKFoldDataset(epoch_val_df, is_multiclass=False)
    v_preds_b = [int(model(X.unsqueeze(0).to(device), head='binary').item() > 0.5) for X, _ in val_ds_b]
    v_trues_b = [int(y.item()) for _, y in val_ds_b]
    
    val_ds_m = PyTorchKFoldDataset(epoch_val_df, is_multiclass=True)
    v_preds_m = [int(torch.argmax(model(X.unsqueeze(0).to(device), head='multiclass'), dim=1).item()) for X, _ in val_ds_m]
    v_trues_m = [int(y.item()) for _, y in val_ds_m]

    test_ds_b = PyTorchKFoldDataset(epoch_test_df, is_multiclass=False)
    t_preds_b = [int(model(X.unsqueeze(0).to(device), head='binary').item() > 0.5) for X, _ in test_ds_b]
    t_trues_b = [int(y.item()) for _, y in test_ds_b]
    
    test_ds_m = PyTorchKFoldDataset(epoch_test_df, is_multiclass=True)
    t_preds_m = [int(torch.argmax(model(X.unsqueeze(0).to(device), head='multiclass'), dim=1).item()) for X, _ in test_ds_m]
    t_trues_m = [int(y.item()) for _, y in test_ds_m]

rep_v_b = classification_report(v_trues_b, v_preds_b, target_names=['Clean', 'Noise'], output_dict=True)
rep_v_m = classification_report(v_trues_m, v_preds_m, target_names=all_names, zero_division=0, output_dict=True)
rep_t_b = classification_report(t_trues_b, t_preds_b, target_names=['Clean', 'Noise'], output_dict=True)
rep_t_m = classification_report(t_trues_m, t_preds_m, target_names=all_names, zero_division=0, output_dict=True)

pd.DataFrame(rep_v_b).transpose().to_csv(output_dir / 'transformer_val_metrics_binary.csv')
pd.DataFrame(rep_v_m).transpose().to_csv(output_dir / 'transformer_val_metrics_multiclass.csv')
pd.DataFrame(rep_t_b).transpose().to_csv(output_dir / 'transformer_final_metrics_binary.csv')
pd.DataFrame(rep_t_m).transpose().to_csv(output_dir / 'transformer_final_metrics_multiclass.csv')

save_report_as_png(rep_v_b, "Validation Fold - Vanilla Transformer Binary Metrics", "transformer_val_report_binary.png")
save_report_as_png(rep_v_m, "Validation Fold - Vanilla Transformer Multiclass Metrics", "transformer_val_report_multiclass.png")
save_report_as_png(rep_t_b, "Test Fold - Vanilla Transformer Binary Metrics", "transformer_classification_report_binary.png")
save_report_as_png(rep_t_m, "Test Fold - Vanilla Transformer Multiclass Metrics", "transformer_classification_report_multiclass.png")

save_confusion_matrix_png(t_trues_b, t_preds_b, ['Clean', 'Noise'], "Transformer Test - Binary Matrix", "transformer_confusion_matrix_binary.png", "YlGnBu")
save_confusion_matrix_png(t_trues_m, t_preds_m, all_names, "Transformer Test - Multiclass Matrix", "transformer_confusion_matrix_multiclass.png", "Purples")
print("Vanilla Transformer processing complete.")