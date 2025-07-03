import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
import os
from torchvision import datasets, transforms

# Device configuration
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
    def __init__(self, num_classes=7, use_residual=True):
        super().__init__()
        self.name = "EmotionAlexNet"
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

def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(model, train_dataset, valid_dataset, batch_size=32, learning_rate=0.001, num_epochs=20, save_dir='./models'):
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs(save_dir, exist_ok=True)

    iters, losses, val_losses, train_acc, val_acc = [], [], [], [], []
    best_val_acc = 0
    best_val_loss = float('inf')
    patience = 3
    counter = 0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1

        # Compute metrics
        avg_loss = running_loss / batch_count
        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, valid_loader, device)

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        val_batch_count = 0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                val_batch_count += 1
        avg_val_loss = running_val_loss / val_batch_count

        # Save metrics
        iters.append(epoch)
        losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)

        # Save model if validation accuracy improves
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(save_dir, f'model_{model.name}_bs{batch_size}_lr{learning_rate}_epoch{epoch+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
            torch.save(model.state_dict(), best_model_path)

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Step scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Training and Validation Loss")
    plt.plot(iters, losses, label="Train")
    plt.plot(iters, val_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Training and Validation Accuracy")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'train_val_metrics_{model.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.show()

    print(f"Final Training Accuracy: {train_acc[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")

    return best_model_path

def main():
    # Mount Google Drive (optional, for Colab)
    try:
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=True)
        save_dir = '/content/gdrive/My Drive/Colab Notebooks/APS360/models'
    except ImportError:
        save_dir = './models'  # Local directory if not in Colab
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        save_dir = './models'  # Fallback to local directory

    # Data transforms with augmentation
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    # Load FER2013+ dataset
    train_data = datasets.ImageFolder('fer2013plus/fer2013/train', transform=transform)
    valid_data = datasets.ImageFolder('fer2013plus/fer2013/test', transform=transform)

    # Uncomment for RAF-DB dataset
    # train_data = datasets.ImageFolder('RAF-DB/train Iskra, transform=transform)
    # valid_data = datasets.ImageFolder('RAF-DB/test', transform=transform)

    print(f"Train dataset size: {len(train_data)}, Valid dataset size: {len(valid_data)}")
    print(f"Number of classes: {len(train_data.classes)}")

    # Initialize model
    model = EmotionAlexNet(num_classes=len(train_data.classes), use_residual=True)

    # Train model
    best_model_path = train(model, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs= 7, save_dir=save_dir)

if __name__ == '__main__':
    main()