def analyze_physical_health(signals):
    fatigue = signals["fatigue"]
    tension = signals["facial_tension"]

    if fatigue == "high":
        energy = "low"
        hydration = "needs improvement"
    else:
        energy = "moderate"
        hydration = "adequate"

    posture = "medium" if tension == "high" else "low"

    return {
        "energyEstimate": energy,
        "hydrationHint": hydration,
        "postureRisk": posture,
        "confidence": 0.8
    }