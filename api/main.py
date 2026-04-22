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
        # Step 5: Dynamic Confidence Calculation
        # -------------------------
        emotion_conf = emotion.get("confidence", 0.5)
        mental_conf = mental.get("confidence", 0.7)
        physical_conf = physical.get("confidence", 0.7)

        # Weighted fusion
        overall_conf = (
            0.4 * emotion_conf +
            0.3 * mental_conf +
            0.3 * physical_conf
        )

        # Penalize if emotion is uncertain
        if emotion_conf < 0.5:
            overall_conf *= 0.9

        overall_conf = round(min(max(overall_conf, 0), 1), 2)

        # -------------------------
        # Final Response
        # -------------------------
        return {
            "faceAnalysis": signals,
            "emotion": emotion,
            "mentalHealthInsights": mental,
            "physicalHealthInsights": physical,
            "confidenceScores": {
                "overall": overall_conf
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