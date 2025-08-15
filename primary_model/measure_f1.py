import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ------------------ Device Setup ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------ Residual Block ------------------
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
        return self.relu(out)

# ------------------ AlexNet-like Emotion Classifier ------------------
class EmotionAlexNet(nn.Module):
    def __init__(self, num_classes=7, use_residual=True):
        super().__init__()
        self.name = "EmotionAlexNet"
        self.use_residual = use_residual

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
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
        x = self.residual(x)
        x = self.dropout(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# ------------------ Accuracy Evaluation ------------------
def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ------------------ Metrics Evaluation ------------------
def get_metrics(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)
    return accuracy, precision, recall, f1, y_true, y_pred

# ------------------ Training Loop ------------------
def train(model, train_dataset, valid_dataset, batch_size=32, learning_rate=0.001, num_epochs=20, save_dir='./models'):
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    os.makedirs(save_dir, exist_ok=True)

    iters, losses, val_losses = [], [], []
    train_acc_list, val_acc_list = [], []
    precision_list, recall_list, f1_list = [], [], []
    best_val_acc = 0
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_accuracy = get_accuracy(model, train_loader, device)

        # Validation metrics
        val_accuracy, precision, recall, f1, y_true, y_pred = get_metrics(model, valid_loader, device)

        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valid_loader)

        iters.append(epoch)
        losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Save best model by validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(save_dir, f'BEST_{model.name}_epoch{epoch+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt')
            torch.save(model.state_dict(), best_model_path)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
              f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # After training, print detailed classification report and confusion matrix
    print("\nDetailed Classification Report on Validation Set:")
    print(classification_report(y_true, y_pred, target_names=valid_dataset.classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_dataset.classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix - Validation Set")
    plt.show()

    # Plot loss, accuracy, precision, recall, and F1 curves
    plt.figure(figsize=(16, 6))
    

    plt.subplot(1, 3, 3)
    plt.plot(iters, precision_list, label='Precision')
    plt.plot(iters, recall_list, label='Recall')
    plt.plot(iters, f1_list, label='F1-score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision, Recall, F1 Score')

    plt.tight_layout()
    plot_name = f'{model.name}_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.show()

    print(f"Best model saved at: {best_model_path}")
    return best_model_path

# ------------------ Main Entry Point ------------------
def main():
    try:
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=True)
        save_dir = '/content/gdrive/My Drive/Colab Notebooks/APS360/models'
    except ImportError:
        save_dir = './models'
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        save_dir = './models'

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
    ])

    # Load dataset - adjust paths as needed
    train_data = datasets.ImageFolder('RAF-DB/train', transform=transform)
    valid_data = datasets.ImageFolder('RAF-DB /test', transform=transform)

    print(f"Train dataset size: {len(train_data)}, Valid dataset size: {len(valid_data)}")
    print(f"Number of classes: {len(train_data.classes)}")

    model = EmotionAlexNet(num_classes=len(train_data.classes), use_residual=True)
    train(model, train_data, valid_data, batch_size=64, learning_rate=0.0005, num_epochs=20, save_dir=save_dir)

if __name__ == '__main__':
    main()
