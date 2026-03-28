import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm
from transformers import ViTForImageClassification, ViTConfig
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves PNG without GUI)
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Settings
DATA_DIR = "./my_custom_dataset"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
BATCH_SIZE = 8          # Smaller batch = less RAM needed
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_SAMPLES = None      # None = use full dataset; set to e.g. 200 for quick testing

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx, 0]}.png")
        # Handle cases where extension might be missing
        if not os.path.exists(img_name):
            img_name = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx, 0]}.jpg")
            
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(model_type='resnet'):
    if model_type == 'vit':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def build_model(model_name='resnet50', num_classes=5):
    print(f"Building model: {model_name}")
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    elif model_name == 'cnn':
        # Simple baseline CNN
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'vit_b_16':
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', config=config)
    else:
        raise ValueError("Unsupported model name. Choose 'resnet50', 'cnn', or 'vit_b_16'")
    
    return model

def train(model_name='resnet50', max_samples=None, num_epochs=NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    if not os.path.exists(TRAIN_CSV):
        print(f"Dataset not found at {DATA_DIR}. Please download the APTOS 2019 Blindness Detection dataset from Kaggle.")
        return

    transform = get_transforms(model_type='vit' if model_name == 'vit_b_16' else 'resnet')
    dataset = APTOSDataset(csv_file=TRAIN_CSV, img_dir=TRAIN_IMG_DIR, transform=transform)

    # ---- QUICK-TRAIN: cap the dataset size ----
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"⚡ Quick-train mode: using {max_samples} samples out of the full dataset.")
    else:
        print(f"📊 Full dataset: {len(dataset)} samples.")
    # -------------------------------------------

    # Simple split (80-20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, criterion, optimizer
    model = build_model(model_name, num_classes=5)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() if model_name != 'vit_b_16' else model.classifier.parameters(), lr=LEARNING_RATE)

    # History tracking
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n🚀 Starting training: {model_name} | Epochs: {num_epochs} | Batch: {BATCH_SIZE}")
    print(f"   Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples\n")

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if model_name == 'vit_b_16':
                outputs = model(images).logits
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss_avg = running_loss / len(train_loader)
        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                if model_name == 'vit_b_16':
                    outputs = model(images).logits
                else:
                    outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_running_loss / len(val_loader)
        history["val_loss"].append(val_loss_avg)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val   Loss: {val_loss_avg:.4f} | Val   Acc: {val_acc:.2f}%")

    print(f"Finished Training {model_name}.")

    # ---- FINAL EVALUATION: Generating Dashboard Metrics ----
    print("\n🏁 FINAL EVALUATION: Generating Dashboard Metrics...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if model_name == 'vit_b_16':
                outputs = model(images).logits
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate Precision, Recall, F1 (Weighted Average)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate per-class accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    metrics_data = {
        "accuracy": f"{accuracy:.2f}%",
        "precision": f"{precision*100:.2f}%",
        "recall": f"{recall*100:.2f}%",
        "f1_score": f"{f1*100:.2f}%",
        "confusion_matrix": cm.tolist(),
        "model_name": model_name,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("performance_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"📊 Performance metrics saved to: performance_metrics.json")

    torch.save(model.state_dict(), f"tele_ophth_assistant_{model_name}_best.pth")

    # ---- Save Training Charts ----
    epochs_range = range(1, num_epochs + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Tele-Ophthalmology Screening Assistant Training Report — {model_name}', fontsize=14, fontweight='bold')

    # Accuracy Chart
    ax1.plot(epochs_range, history["train_acc"], 'b-o', label='Train Accuracy')
    ax1.plot(epochs_range, history["val_acc"], 'r-o', label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)

    # Loss Chart
    ax2.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
    ax2.plot(epochs_range, history["val_loss"], 'r-o', label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    chart_path = f"training_report_{model_name}.png"
    plt.savefig(chart_path, dpi=150)
    print(f"\n✅ Training charts saved to: {chart_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Tele-Ophthalmology Screening Assistant Models")
    parser.add_argument('--model', type=str, default='vit_b_16', choices=['cnn', 'resnet50', 'vit_b_16'],
                        help="Choose model architecture to train.")
    parser.add_argument('--quick', action='store_true',
                        help="Quick-train mode: uses only 200 samples and 3 epochs.")
    parser.add_argument('--samples', type=int, default=None,
                        help="Manually set number of samples to use (e.g. --samples 500).")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS}).")
    args = parser.parse_args()

    # --quick is a shortcut for small dataset + few epochs
    max_samples = args.samples
    num_epochs = args.epochs
    if args.quick:
        max_samples = max_samples or 200
        num_epochs = 3
        print("⚡ Quick-train mode enabled: 200 samples, 3 epochs.")

    train(model_name=args.model, max_samples=max_samples, num_epochs=num_epochs)
