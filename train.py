import sys
import os
import json
import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from datetime import datetime

from src.preprocessing import load_and_processing
from src.models import BERTClassifier

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# Free up cache
gc.collect()
torch.cuda.empty_cache()

# Create a unique folder for each run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_folder_name = f"{config['bert_name'].replace('/', '_')}_{timestamp}"
run_folder = os.path.join("results", run_folder_name)
os.makedirs(run_folder, exist_ok=True)
checkpoints_folder = os.path.join(run_folder, "checkpoints")
os.makedirs(checkpoints_folder, exist_ok=True)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["bert_name"])
train_dataloader, val_dataloader = load_and_processing(config["dataset_path"], tokenizer)

# Initialize model
model = BERTClassifier(config["bert_name"], 2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
total_steps = len(train_dataloader) * config["epochs"]
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

# Training function
def train(model, data_loader, optimizer, scheduler, device, epoch):
    model.train()
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=True):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    predictions, actual_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

# Redirect output to log file
log_file = os.path.join(run_folder, "training_output.txt")
with open(log_file, "w") as f:
    sys.stdout = f

    print(f"### Training Started at {timestamp} ###")
    print(f"Using model: {config['bert_name']}")
    print(f"Dataset path: {config['dataset_path']}")
    print(f"Batch size: {config['batch_size']}, Max Length: {config['max_length']}, Epochs: {config['epochs']}")
    print("-" * 50)

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train(model, train_dataloader, optimizer, scheduler, device, epoch)
        accuracy, report = evaluate(model, val_dataloader, device)

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

        if config["save_model"]:
            checkpoint_path = os.path.join(checkpoints_folder, f"bert_pt_classifier_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at: {checkpoint_path}")

    print(f"\n### Training Completed at {datetime.now().strftime('%Y%m%d_%H%M%S')} ###")

    # Reset stdout
    sys.stdout = sys.__stdout__

print(f"Training output saved to: {log_file}")
