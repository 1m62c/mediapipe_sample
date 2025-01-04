import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Poseの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# スクワット判定に使用する角度を計算する関数
def calculate_angle(a, b, c):
    a = np.array(a)  # 始点
    b = np.array(b)  # 中間点
    c = np.array(c)  # 終点
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

# カメラキャプチャ開始
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mediapipe用に画像をRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # ポーズ推定
    results = pose.process(image)

    # 画像を描画用に再変換
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # ランドマークの取得
        landmarks = results.pose_landmarks.landmark

        # 必要なポイント（例: 右腰, 右膝, 右足首）
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # 角度計算
        angle = calculate_angle(hip, knee, ankle)

        # 角度を描画
        cv2.putText(image, f'Angle: {int(angle)}', 
                    tuple(np.multiply(knee, [640, 480]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # フォームチェック
        if angle > 160:
            cv2.putText(image, 'Bend your knees!', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif angle < 80:
            cv2.putText(image, 'Excellent', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif angle < 100:
            cv2.putText(image, 'Good Form!', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Go deeper!', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # ランドマークを描画
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 表示
    cv2.imshow('Squat Correction App', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
