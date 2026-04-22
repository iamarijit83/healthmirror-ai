import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v2

# Load model once
model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 7)

model.load_state_dict(torch.load("models/best_emotion_model.pth", map_location="cpu"))
model.eval()

# Labels
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def detect_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return {
        "label": EMOTIONS[pred.item()],
        "confidence": round(confidence.item(), 3)
    }