from fastapi import FastAPI, UploadFile, File
import shutil
import os

from services.face_analysis import analyze_face
from services.mental_inference import analyze_mental_health
from services.physical_inference import analyze_physical_health
from services.emotion_inference import detect_emotion

app = FastAPI()


# -------------------------
# Health check route
# -------------------------
@app.get("/")
def home():
    return {"message": "HealthMirror AI API is running"}


# -------------------------
# Main API endpoint
# -------------------------
@app.post("/analyze-face")
async def analyze(file: UploadFile = File(...)):
    
    # Ensure temp folder exists
    os.makedirs("data", exist_ok=True)

    # Save uploaded file
    file_path = f"data/temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # -------------------------
        # Step 1: Face Analysis
        # -------------------------
        face_result = analyze_face(file_path)

        if "error" in face_result:
            return face_result

        signals = face_result["signals"]

        # -------------------------
        # Step 2: Emotion Detection
        # -------------------------
        emotion = detect_emotion(file_path)

        # -------------------------
        # Step 3: Mental Inference
        # -------------------------
        mental = analyze_mental_health(signals, emotion)

        # -------------------------
        # Step 4: Physical Inference
        # -------------------------
        physical = analyze_physical_health(signals)

        # -------------------------
        # Final Response
        # -------------------------
        return {
            "faceAnalysis": signals,
            "emotion": emotion,
            "mentalHealthInsights": mental,
            "physicalHealthInsights": physical,
            "confidenceScores": {
                "overall": 0.85
            },
            "recommendations": [
                "Maintain regular sleep schedule",
                "Stay hydrated",
                "Take short screen breaks"
            ]
        }

    finally:
        # Cleanup file safely
        if os.path.exists(file_path):
            os.remove(file_path)