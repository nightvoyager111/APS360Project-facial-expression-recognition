model = EmotionAlexNet(num_classes=7, use_residual=True)
model.load_state_dict(torch.load('models/BEST_EmotionAlexNet_RAFDB_epoch18_20250807_152345.pt'))
model.to(device)
model.eval()
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the test dataset
test_data = datasets.ImageFolder('test_set', transform=transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Evaluate accuracy
test_acc = get_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")