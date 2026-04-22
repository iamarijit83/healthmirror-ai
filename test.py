from services.face_analysis import analyze_face
from services.mental_inference import analyze_mental_health
from services.physical_inference import analyze_physical_health
from services.emotion_inference import detect_emotion

image_path = r"D:\HealthMirror-AI\test.jpeg"

# Step 1: Face
face_result = analyze_face(image_path)
print("\nFACE RESULT:", face_result)

if "error" in face_result:
    print("Face detection failed")
    exit()

signals = face_result["signals"]

# Step 2: Emotion
emotion = detect_emotion(image_path)
print("\nEMOTION:", emotion)

# Step 3: Mental
mental = analyze_mental_health(signals, emotion)
print("\nMENTAL:", mental)

# Step 4: Physical
physical = analyze_physical_health(signals)
print("\nPHYSICAL:", physical)

# Step 5: Confidence Calculation (same as API)
emotion_conf = emotion.get("confidence", 0.5)
mental_conf = mental.get("confidence", 0.7)
physical_conf = physical.get("confidence", 0.7)

overall_conf = (
    0.4 * emotion_conf +
    0.3 * mental_conf +
    0.3 * physical_conf
)

if emotion_conf < 0.5:
    overall_conf *= 0.9

overall_conf = round(min(max(overall_conf, 0), 1), 2)

print("\nFINAL OUTPUT:")
print({
    "faceAnalysis": signals,
    "emotion": emotion,
    "mentalHealthInsights": mental,
    "physicalHealthInsights": physical,
    "confidenceScores": {
        "overall": overall_conf
    }
})