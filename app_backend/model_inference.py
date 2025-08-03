import torch
from PIL import Image
from torchvision import transforms
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from primary_model.ame_try1 import EmotionAlextNet

model = EmotionAlextNet(num_classes=7, use_residual=True)
checkpoint = torch.load('./models/model_EmotionAlexNet_bs64_lr0.0005_epoch18_20250707_151009.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
model.eval()

EMOTION_MAPPING = {
    'anger': 'ANGRY',
    'disgust': 'DISGUSTED',
    'fear': 'FEARFUL',
    'happiness': 'HAPPY',
    'neutral': 'NEUTRAL',
    'sadness': 'SAD',
    'surprise': 'SURPRISED'
}

# Preprocessing (match training)
transform = transforms.Compose([
    #transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def predict_emotion(pil_img):
    print("Received image for prediction:", pil_img.size)
    img = pil_img.convert('RGB')
    tensor = transform(img).unsqueeze(0)  
    with torch.no_grad():
        output = model(tensor)
        probs = torch.exp(output)
        pred_idx = torch.argmax(probs, dim=1).item()
        emotion = list(EMOTION_MAPPING.keys())[pred_idx]
        return emotion
    