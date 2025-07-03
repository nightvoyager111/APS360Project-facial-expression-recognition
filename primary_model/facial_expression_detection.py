import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class EmotionAlexNet(nn.Module):
    def __init__(self, num_classes=7, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(256) if use_residual else nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.use_residual:
            x = self.residual(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Data augmentation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
])

# Data
train_data = datasets.ImageFolder('fer2013plus/fer2013/train', transform=transform)
test_data = datasets.ImageFolder('fer2013plus/fer2013/test', transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
]))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

# Model, optimizer, loss, scheduler
model = EmotionAlexNet(num_classes=len(train_data.classes), use_residual=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
class_weights = compute_class_weight('balanced', classes=np.arange(len(train_data.classes)), y=np.array(train_data.targets))
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = torch.nn.NLLLoss(weight=class_weights)

num_epochs = 10
train_losses, val_losses = [], []
train_errors, val_errors = [], []
train_accuracies = []

print(f"Training primary model with epochs={num_epochs}, lr={optimizer.param_groups[0]['lr']}, batch_size={train_loader.batch_size}")

def main():
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        train_err = 1 - train_acc
        train_losses.append(train_loss)
        train_errors.append(train_err)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = loss_fn(outputs, y)
                val_loss += loss.item() * X.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_err = 1 - val_acc
        val_losses.append(val_loss)
        val_errors.append(val_err)

        print(f"Epoch {epoch}: Train Accuracy: {train_acc*100:.4f}% | Train Loss: {train_loss:.4f}, Train Err: {train_err:.4f} | Val Loss: {val_loss:.4f}, Val Err: {val_err:.4f}")
        print(classification_report(all_targets, all_preds, target_names=train_data.classes, zero_division=0))

    # Plotting
    epochs = range(1, len(train_errors) + 1)
    plt.figure()
    plt.plot(epochs, train_errors, label='Train Error')
    plt.plot(epochs, val_errors, label='Validation Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Train vs Validation Error 10 epochs, 32 batch size, 0.0005 lr')
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig('primary_model/PM_train_vs_valError.png')
    plt.show()

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss 10 epochs, 32 batch size, 0.0005 lr')
    plt.legend()
    plt.ylim(bottom=0)
    plt.savefig('primary_model/PM_train_vs_valLoss.png')
    plt.show()

if __name__ == "__main__":
    main()
