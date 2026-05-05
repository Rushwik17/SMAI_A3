import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

MODEL_PATH = "../models/l3_cube_bert/"
TRAIN_CSV = "../dataset_l3cube/Hindi_Train.csv"
VAL_CSV   = "../dataset_l3cube/Hindi_Valid.csv"
TEST_CSV  = "../dataset_l3cube/Hindi_Test.csv"

MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def build_label_map(csv_path):
    df = pd.read_csv(csv_path)
    labels = sorted(df["Category"].unique())
    return {l: i for i, l in enumerate(labels)}

label2id = build_label_map(TRAIN_CSV)
id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)

train_df = pd.read_csv(TRAIN_CSV)
y_train = train_df["Category"].map(label2id).values

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

class NewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, input_col):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.input_col = input_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.input_col])
        label = label2id[self.df.iloc[idx]["Category"]]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(label)
        return item

def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return acc, precision, recall, f1

def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    loop = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in loop:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            logits = outputs.logits

            batch_preds = torch.argmax(logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    return compute_metrics(preds, labels)

def run_experiment(input_col):
    print(f"\n===== Running for INPUT: {input_col} =====")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    ).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_ds = NewsDataset(TRAIN_CSV, tokenizer, input_col)
    val_ds   = NewsDataset(VAL_CSV, tokenizer, input_col)
    test_ds  = NewsDataset(TEST_CSV, tokenizer, input_col)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LR)

    save_dir = f"../saved_models/l3cube_{input_col.lower()}"
    os.makedirs(save_dir, exist_ok=True)

    best_f1 = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        print(f"Train Loss: {train_loss:.4f}")

        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader)
        print(f"Val -> Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "label_map.json"), "w") as f:
                json.dump(label2id, f)

    model = AutoModelForSequenceClassification.from_pretrained(save_dir).to(DEVICE)

    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader)

    print(f"\nTEST RESULTS ({input_col}):")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1 Score : {test_f1:.4f}")

if __name__ == "__main__":
    run_experiment("Title")