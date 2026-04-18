from services.face_analysis import analyze_face
from services.mental_inference import analyze_mental_health
from services.physical_inference import analyze_physical_health

image_path = r"D:\HealthMirror-AI\test.jpeg"

face_result = analyze_face(image_path)

signals = face_result["signals"]

mental = analyze_mental_health(signals)
physical = analyze_physical_health(signals)

final_output = {
    "faceAnalysis": face_result["signals"],
    "mentalHealthInsights": mental,
    "physicalHealthInsights": physical,
    "confidenceScores": {
        "overall": 0.8
    },
    "recommendations": [
        "Maintain regular sleep schedule",
        "Stay hydrated",
        "Take short screen breaks"
    ]
}



print(final_output)