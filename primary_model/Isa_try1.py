import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MiniCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.6)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def main():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    # 修改为你本地 FER+/RAF-DB 路径
    train_data = datasets.ImageFolder('fer2013plus/fer2013/train', transform=transform)
    val_data = datasets.ImageFolder('fer2013plus/fer2013/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

    model = MiniCNN(num_classes=len(train_data.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 7
    train_losses, val_losses = [], []
    train_errors, val_errors = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            pred = output.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_losses.append(total_loss / total)
        train_errors.append(1 - correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = F.nll_loss(output, y)
                val_loss += loss.item() * X.size(0)
                pred = output.argmax(1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        val_losses.append(val_loss / val_total)
        val_errors.append(1 - val_correct / val_total)

        print(f"Epoch {epoch}: Train Acc = {1-train_errors[-1]:.4f}, Val Acc = {1-val_errors[-1]:.4f}")

    os.makedirs("mini_model", exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_errors, label="Train Error")
    plt.plot(range(1, num_epochs+1), val_errors, label="Val Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Train vs Validation Error")
    plt.legend()
    plt.savefig("mini_model/train_val_error.png")
    plt.show()

if __name__ == '__main__':
    main()
