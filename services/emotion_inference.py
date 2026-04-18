import torch
import torchvision.transforms as transforms
from PIL import Image
from models.emotion_model import EmotionCNN

# Emotion labels
emotion_labels = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

# Load model once (important)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("models/best_emotion_model.pth", map_location=device))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def detect_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "label": emotion_labels[predicted.item()],
        "confidence": round(confidence.item(), 3)
    }