def analyze_mental_health(signals, emotion):
    fatigue = signals.get("fatigue")
    stress = signals.get("stress")
    eye_strain = signals.get("eye_strain")

    emotion_label = emotion.get("label", "neutral")
    emotion_conf = emotion.get("confidence", 0.5)

    # -------------------------
    # Emotion smoothing
    # -------------------------
    if emotion_conf < 0.65:
        emotion_label = "uncertain"

    # -------------------------
    # Anxiety Logic
    # -------------------------
    if emotion_label in ["fear", "sad"] and stress == "high":
        anxiety = "high"
    elif stress == "moderate":
        anxiety = "moderate"
    else:
        anxiety = "low"

    # -------------------------
    # Sleep Logic
    # -------------------------
    if fatigue == "high" or eye_strain == "high":
        sleep = "poor"
    else:
        sleep = "adequate"

    # -------------------------
    # Contradiction Fix
    # -------------------------
    if emotion_label == "fear" and stress == "low":
        anxiety = "low"

    # -------------------------
    # Recommendation Logic
    # -------------------------
    if anxiety == "high":
        action = "Consider stress management or consultation"
    elif anxiety == "moderate":
        action = "Practice relaxation techniques"
    else:
        action = "Maintain healthy routines"

    return {
        "anxietyRisk": anxiety,
        "sleepQuality": sleep,
        "recommendedAction": action,
        "emotionUsed": emotion_label,
        "confidence": round(0.7 + emotion_conf * 0.2, 2)
    }