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

# --- MASTER POOL PREPARATION ---
print("Compiling raw master dataframe pool...")
train_df = pd.read_csv(input_dir / 'train_split.csv')
val_df = pd.read_csv(input_dir / 'val_split.csv')
test_df = pd.read_csv(input_dir / 'test_split.csv')
full_master_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Path cleaning
full_master_df['embedding_file'] = full_master_df['embedding_file'].apply(
    lambda x: str(input_dir / 'extracted_embeddings' / 'extracted_embeddings' / str(x).split('\\')[-1])
)

def get_class_name(path_str):
    parts = str(path_str).replace('\\', '/').split('/')
    return 'Clean' if any('clean' in p.lower() for p in parts) else (parts[-2] if len(parts) >= 2 else 'Unknown')

full_master_df['noise_type_name'] = full_master_df['original_video_path'].apply(get_class_name)
all_names = sorted(list(set(full_master_df['noise_type_name'].unique()) - {'Clean'}))
type_mapping = {name: i for i, name in enumerate(all_names)}
full_master_df['noise_type_label'] = full_master_df['noise_type_name'].apply(lambda x: type_mapping[x] if x != 'Clean' else -1)

# --- PYTORCH RUNTIME DATASET ---
class PyTorchKFoldDataset(Dataset):
    def __init__(self, dataframe, is_multiclass=False):
        self.df = dataframe.reset_index(drop=True)
        if is_multiclass:
            self.df = self.df[self.df['noise_type_label'] >= 0].reset_index(drop=True)
        self.is_multiclass = is_multiclass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        matrix = np.load(row['embedding_file'])
        label = row['noise_type_label'] if self.is_multiclass else row['label']
        return torch.tensor(matrix, dtype=torch.float32), torch.tensor(label, dtype=torch.long if self.is_multiclass else torch.float32)

# --- METRIC VISUALIZATION EXPORTERS ---
def save_report_as_png(report_dict, title, filename):
    df_rep = pd.DataFrame(report_dict).transpose().round(3)
    fig, ax = plt.subplots(figsize=(10, len(df_rep) * 0.4 + 2))
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=10)
    table = ax.table(cellText=df_rep.values, rowLabels=df_rep.index, colLabels=df_rep.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
    plt.tight_layout(); plt.savefig(output_dir / filename, dpi=300); plt.close()

def save_confusion_matrix_png(y_true, y_pred, target_names, title, filename, cmap="YlGnBu"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=target_names, yticklabels=target_names, annot_kws={"weight": "bold"})
    plt.title(title, fontsize=14, weight='bold', pad=15)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300); plt.close()

# --- MODEL CORE ---
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dense_shared = nn.Linear(hidden_dim, 64)
        self.gelu = nn.GELU()
        self.binary_head = nn.Linear(64, 1)
        self.multiclass_head = nn.Linear(64, 10)

    def forward(self, x, head='binary'):
        lstm_out, _ = self.lstm(x)
        last_step_feat = lstm_out[:, -1, :]
        shared_feat = self.gelu(self.dense_shared(last_step_feat))
        if head == 'binary':
            return torch.sigmoid(self.binary_head(shared_feat)).squeeze(-1)
        return self.multiclass_head(shared_feat)

# --- ENGINE EXECUTION ---
model = CNNLSTMModel().to(device)
criterion_bin, criterion_mul = nn.BCELoss(), nn.CrossEntropyLoss()
opt_bin, opt_mul = optim.Adam(model.parameters(), lr=1e-4), optim.Adam(model.parameters(), lr=1e-4)

TOTAL_SAMPLES = len(full_master_df)
fold_size = int(TOTAL_SAMPLES / 10)

print("\nLaunching Shuffled 10-Fold CNN-LSTM CV Engine...")
for epoch in range(EPOCHS):
    shuffled_master = full_master_df.sample(frac=1.0).reset_index(drop=True)
    
    available_folds = list(range(10))
    val_fold_idx = np.random.choice(available_folds)
    available_folds.remove(val_fold_idx)
    test_fold_idx = np.random.choice(available_folds)
    
    val_start, val_end = val_fold_idx * fold_size, (val_fold_idx + 1) * fold_size
    test_start, test_end = test_fold_idx * fold_size, (test_fold_idx + 1) * fold_size
    
    epoch_val_df = shuffled_master.iloc[val_start:val_end]
    epoch_test_df = shuffled_master.iloc[test_start:test_end]
    train_indices = [i for i in range(TOTAL_SAMPLES) if not (val_start <= i < val_end or test_start <= i < test_end)]
    epoch_train_df = shuffled_master.iloc[train_indices]
    
    loader_bin = DataLoader(PyTorchKFoldDataset(epoch_train_df, is_multiclass=False), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_mul = DataLoader(PyTorchKFoldDataset(epoch_train_df, is_multiclass=True), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Training Stages
    model.train()
    for X_b, y_b in loader_bin:
        X_b, y_b = X_b.to(device), y_b.to(device)
        opt_bin.zero_grad(); loss = criterion_bin(model(X_b, head='binary'), y_b); loss.backward(); opt_bin.step()
    for X_m, y_m in loader_mul:
        X_m, y_m = X_m.to(device), y_m.to(device)
        opt_mul.zero_grad(); loss = criterion_mul(model(X_m, head='multiclass'), y_m); loss.backward(); opt_mul.step()

# --- EVALUATION AND EXPORTS ---
print("Compiling final performance assets...")
model.eval()
with torch.no_grad():
    # Evaluate Validation Fold
    val_ds_b = PyTorchKFoldDataset(epoch_val_df, is_multiclass=False)
    v_preds_b = [int(model(X.unsqueeze(0).to(device), head='binary').item() > 0.5) for X, _ in val_ds_b]
    v_trues_b = [int(y.item()) for _, y in val_ds_b]
    
    val_ds_m = PyTorchKFoldDataset(epoch_val_df, is_multiclass=True)
    v_preds_m = [int(torch.argmax(model(X.unsqueeze(0).to(device), head='multiclass'), dim=1).item()) for X, _ in val_ds_m]
    v_trues_m = [int(y.item()) for _, y in val_ds_m]

    # Evaluate Test Fold
    test_ds_b = PyTorchKFoldDataset(epoch_test_df, is_multiclass=False)
    t_preds_b = [int(model(X.unsqueeze(0).to(device), head='binary').item() > 0.5) for X, _ in test_ds_b]
    t_trues_b = [int(y.item()) for _, y in test_ds_b]
    
    test_ds_m = PyTorchKFoldDataset(epoch_test_df, is_multiclass=True)
    t_preds_m = [int(torch.argmax(model(X.unsqueeze(0).to(device), head='multiclass'), dim=1).item()) for X, _ in test_ds_m]
    t_trues_m = [int(y.item()) for _, y in test_ds_m]

# Generate reports
rep_v_b = classification_report(v_trues_b, v_preds_b, target_names=['Clean', 'Noise'], output_dict=True)
rep_v_m = classification_report(v_trues_m, v_preds_m, target_names=all_names, zero_division=0, output_dict=True)
rep_t_b = classification_report(t_trues_b, t_preds_b, target_names=['Clean', 'Noise'], output_dict=True)
rep_t_m = classification_report(t_trues_m, t_preds_m, target_names=all_names, zero_division=0, output_dict=True)

# Save CSVs
pd.DataFrame(rep_v_b).transpose().to_csv(output_dir / 'lstm_val_metrics_binary.csv')
pd.DataFrame(rep_v_m).transpose().to_csv(output_dir / 'lstm_val_metrics_multiclass.csv')
pd.DataFrame(rep_t_b).transpose().to_csv(output_dir / 'lstm_final_metrics_binary.csv')
pd.DataFrame(rep_t_m).transpose().to_csv(output_dir / 'lstm_final_metrics_multiclass.csv')

# Save PNGs
save_report_as_png(rep_v_b, "Validation Fold - LSTM Binary Metrics", "lstm_val_report_binary.png")
save_report_as_png(rep_v_m, "Validation Fold - LSTM Multiclass Metrics", "lstm_val_report_multiclass.png")
save_report_as_png(rep_t_b, "Test Fold - LSTM Binary Metrics", "lstm_classification_report_binary.png")
save_report_as_png(rep_t_m, "Test Fold - LSTM Multiclass Metrics", "lstm_classification_report_multiclass.png")

save_confusion_matrix_png(t_trues_b, t_preds_b, ['Clean', 'Noise'], "LSTM Test - Binary Matrix", "lstm_confusion_matrix_binary.png", "YlGnBu")
save_confusion_matrix_png(t_trues_m, t_preds_m, all_names, "LSTM Test - Multiclass Matrix", "lstm_confusion_matrix_multiclass.png", "Purples")
print("LSTM processing complete.")