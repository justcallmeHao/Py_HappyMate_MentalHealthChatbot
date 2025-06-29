# MentalityAnalyserTraining.py
from trainer_interface import AnalyserTrainer
#For Training model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# For NLP and preprocessing
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
# For pipeline & download data from external source
import joblib
import os
import pandas as pd

# ----------------------------
# Dataset class
# ----------------------------
class MentalityDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ----------------------------
# Main Trainer
# ----------------------------
class MentalityAnalyserTrainer(AnalyserTrainer):
    def __init__(self):
        self.raw_data = None
        self.encoder = SentenceTransformer("all-MiniLM-L12-v2")
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")

    def load_data(self):
        # Load local CSV file (adjust the path if needed)
        df = pd.read_csv("Data/mental_health.csv")  # Downloaded file
        # Optionally convert numeric labels to string labels
        label_map = {0: "no_stress", 1: "stress"}
        # Format the data into a list of dictionaries
        self.raw_data = [
            {"text": row["text"], "label": label_map.get(row["label"], row["label"])}
            for _, row in df.iterrows()
        ]

    def preprocess(self):
        texts = [x["text"] for x in self.raw_data]
        labels = [x["label"] for x in self.raw_data]

        self.X = self.encoder.encode(texts, show_progress_bar=True)
        self.y = self.label_encoder.fit_transform(labels)

        # Save tools
        os.makedirs("Models", exist_ok=True)
        joblib.dump(self.encoder, "Models/mentality_encoder.pkl")
        joblib.dump(self.label_encoder, "Models/mentality_labels.pkl")

    def split_data(self):
        # First split: 80% train, 20% validate + test
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.20, random_state=42, stratify=self.y)

        # Second split: 10% validate + 10% test (50%x20%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        self.train_loader = DataLoader(MentalityDataset(X_train, y_train), batch_size=16, shuffle=True)
        self.val_loader = DataLoader(MentalityDataset(X_val, y_val), batch_size=16)
        self.test_loader = DataLoader(MentalityDataset(X_test, y_test), batch_size=16)

    def train_model(self):
        input_size = self.X.shape[1]
        output_size = len(set(self.y))

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    # Layer 1
                    nn.Linear(input_size, 256),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.2),

                    # Layer 2
                    nn.Linear(256, 192),
                    nn.BatchNorm1d(192),
                    nn.GELU(),

                    # Layer 3
                    nn.Linear(192, 128),
                    nn.BatchNorm1d(128),
                    nn.GELU(),

                    # Layer 4
                    nn.Linear(128, 96),
                    nn.BatchNorm1d(96),
                    nn.GELU(),

                    # Layer 5 (final hidden layer → output)
                    nn.Linear(96, output_size)
                )

            def forward(self, x):
                return self.net(x)

        self.model = Net().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        class_weights = compute_class_weight("balanced", classes=np.unique(self.y), y=self.y)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for epoch in range(30):
            total_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            val_loss, val_acc = self.validate_model()
            print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}")
            scheduler.step()

    def validate_model(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total if total > 0 else 0
        return 0, acc

    def evaluate_model(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self.model(x_batch)
                preds = torch.argmax(output, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc = correct / total if total > 0 else 0
        print(f"✅ Test Accuracy: {acc:.2f}")

    def export_model(self, save_path):
        self.model.eval()
        example = torch.rand(1, self.X.shape[1]).to(self.device)
        traced = torch.jit.trace(self.model, example)
        traced.save(save_path)
        print(f"📦 Saved TorchScript model to {save_path}")

    def run_pipeline(self, save_path):
        self.load_data()
        self.preprocess()
        self.split_data()
        self.train_model()
        self.evaluate_model()
        self.export_model(save_path)
