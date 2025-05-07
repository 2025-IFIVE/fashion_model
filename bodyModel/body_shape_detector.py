import cv2
import mediapipe as mp
import numpy as np
import tempfile

mp_pose = mp.solutions.pose

def detect_body_shape_from_bytes(image_bytes) -> str:
    # numpy 배열로 디코딩
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return "이미지를 불러올 수 없습니다."

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return "체형 분석 실패"

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        shoulder_l = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_r = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        waist_l = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        waist_r = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        sx1, sx2 = int(shoulder_l.x * w), int(shoulder_r.x * w)
        wx1, wx2 = int(waist_l.x * w), int(waist_r.x * w)
        shoulder_size = abs(sx1 - sx2)
        waist_size = abs(wx1 - wx2)

        pelvis_y = int(((waist_l.y + waist_r.y) / 2) * h)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        row_pelvis = thresh[pelvis_y, :]
        left_edge = np.argmax(row_pelvis > 0)
        right_edge = len(row_pelvis) - np.argmax(np.flip(row_pelvis) > 0) - 1
        hip_size = abs(right_edge - left_edge)

        # 비율 계산
        if shoulder_size == 0 or waist_size == 0 or hip_size == 0:
            return "체형 분석 실패"

        s_w_ratio = shoulder_size / waist_size
        w_h_ratio = waist_size / hip_size

        # 체형 판별 로직
        if waist_size / shoulder_size >= 1.05:
            return "사과형"
        elif waist_size / shoulder_size >= 0.75:
            return "직사각형"
        elif waist_size / shoulder_size <= 0.75 and waist_size / hip_size <= 0.75:
            return "모래시계형"
        elif hip_size > shoulder_size:
            return "삼각형"
        elif shoulder_size > hip_size:
            return "역삼각형"
        else:
            return "기타"