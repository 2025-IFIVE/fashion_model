# 최종코드
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from PIL import Image

mp_drawing = mp.solutions.drawing_utils

def get_pose_lines(image_bgr):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, None

        h, w, _ = image_bgr.shape
        lm = results.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        y_shoulder = int(((lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2) * h)
        y_hip = int(((lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2) * h)

        y_shoulder_f = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
        y_hip_f = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        y_waist_f = (2 * y_shoulder_f + 3 * y_hip_f) / 5
        y_waist = int(y_waist_f * h)

        return (y_shoulder, y_waist, y_hip), lm

def remove_arm_from_mask(mask, landmarks, image_shape):
    h, w = image_shape[:2]

    def to_px(landmark):
        return int(landmark.x * w), int(landmark.y * h)

    arms = [
        [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
         mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
         mp.solutions.pose.PoseLandmark.LEFT_WRIST],
        [mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
         mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
         mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    ]

    for arm in arms:
        try:
            points = np.array([to_px(landmarks[pt]) for pt in arm], dtype=np.int32)
            cv2.fillPoly(mask, [points], 0)
        except:
            continue

    return mask

def extract_mask_with_rembg(image_bgr, landmarks=None):
    image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    pil_image = Image.fromarray(image_rgba)

    output = remove(pil_image)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    binary_mask = (alpha > 100).astype(np.uint8) * 255

    no_arm_mask = binary_mask.copy()
    if landmarks:
        no_arm_mask = remove_arm_from_mask(no_arm_mask, landmarks, image_bgr.shape)

    return binary_mask, no_arm_mask

def detect_body_shape_with_pose_and_segmentation(image_bgr):
    y_coords, landmarks = get_pose_lines(image_bgr)
    if y_coords is None:
        return "체형 분석 실패", image_bgr

    y_shoulder, y_waist, y_hip = y_coords
    base_mask, no_arm_mask = extract_mask_with_rembg(image_bgr, landmarks)

    def get_width_and_edges(y, mask):
        row_mask = mask[y:y+1, :]
        contours, _ = cv2.findContours(row_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0

        center_x = mask.shape[1] // 2
        min_dist = float('inf')
        selected_contour = None

        for cnt in contours:
            x, _, w_box, _ = cv2.boundingRect(cnt)
            cnt_center = x + w_box // 2
            dist = abs(cnt_center - center_x)
            if dist < min_dist:
                min_dist = dist
                selected_contour = (x, x + w_box)

        left, right = selected_contour
        width = abs(right - left)
        return width, left, right

    sw, sl, sr = get_width_and_edges(y_shoulder, base_mask)
    ww, wl, wr = get_width_and_edges(y_waist, no_arm_mask)
    hw, hl, hr = get_width_and_edges(y_hip, no_arm_mask)

    if sw == 0 or ww == 0 or hw == 0:
        return "체형 분석 실패", base_mask

    if ww / sw >= 1.05:
        body_shape = "사과형"
    elif ww / sw >= 0.75:
        body_shape = "직사각형"
    elif ww / sw <= 0.75 and ww / hw <= 0.75:
        body_shape = "모래시계형"
    elif hw > sw:
        body_shape = "삼각형"
    elif sw > hw:
        body_shape = "역삼각형"
    else:
        body_shape = "기타"

    return body_shape, base_mask
