import cv2
import mediapipe as mp
import math
import os

mp_face_mesh = mp.solutions.face_mesh


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def analyze_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return {"error": "Image not found"}

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return {"error": "No face detected"}

        landmarks = results.multi_face_landmarks[0].landmark

        # Convert normalized → pixel coords
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # -------------------------
        # 1. Eye Openness (Fatigue)
        # -------------------------
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        left_eye_width = distance(points[33], points[133])

        eye_openness = distance(left_eye_top, left_eye_bottom) / (left_eye_width + 1e-6)

        # -------------------------
        # 2. Face Symmetry
        # -------------------------
        left_cheek = points[234]
        right_cheek = points[454]
        nose_center = points[1]

        left_dist = distance(left_cheek, nose_center)
        right_dist = distance(right_cheek, nose_center)

        symmetry_score = 1 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)

        # -------------------------
        # 3. Facial Tension
        # -------------------------
        brow_left = points[70]
        eye_left = points[159]

        tension_score = distance(brow_left, eye_left)

        # -------------------------
        # 4. Signals
        # -------------------------
        eye_strain = "high" if eye_openness < 0.2 else "normal"
        fatigue = "high" if eye_openness < 0.2 else "low"
        stress = "high" if symmetry_score < 0.85 else "moderate"
        tension_level = "high" if tension_score < 15 else "low"

        # -------------------------
        # 5. Confidence Score
        # -------------------------
        confidence = 0.5

        if 0.15 < eye_openness < 0.35:
            confidence += 0.2

        if symmetry_score > 0.85:
            confidence += 0.2

        if tension_score > 20:
            confidence += 0.1

        confidence = round(min(confidence, 1.0), 2)

        # -------------------------
        # 6. Visualization
        # -------------------------
        annotated = image.copy()

        # Eyes
        cv2.circle(annotated, points[159], 3, (255, 0, 0), -1)
        cv2.circle(annotated, points[145], 3, (255, 0, 0), -1)

        # Cheeks
        cv2.circle(annotated, points[234], 3, (0, 0, 255), -1)
        cv2.circle(annotated, points[454], 3, (0, 0, 255), -1)

        # Nose
        cv2.circle(annotated, points[1], 4, (0, 255, 255), -1)

        # Text
        cv2.putText(annotated, f"Fatigue: {fatigue}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated, f"Stress: {stress}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(annotated, f"Eye: {eye_strain}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # -------------------------
        # 7. Save Image
        # -------------------------
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/result.jpg"
        cv2.imwrite(output_path, annotated)

        # -------------------------
        # Final Return
        # -------------------------
        return {
            "face_detected": True,
            "metrics": {
                "eye_openness": round(eye_openness, 3),
                "symmetry_score": round(symmetry_score, 3),
                "tension_score": round(tension_score, 2)
            },
            "signals": {
                "fatigue": fatigue,
                "stress": stress,
                "eye_strain": eye_strain,
                "facial_tension": tension_level
            },
            "confidence": confidence,
            "output_image": output_path
        }