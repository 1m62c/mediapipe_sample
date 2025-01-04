# 無量空処

import cv2
import mediapipe as mp

# MediaPipeのセットアップ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 画像の読み込み
overlay_image = cv2.imread("image/muryokusho.jpg")  # 合成する画像

def resize_image_to_frame(image, width, height):
    """画像をウィンドウ全体のサイズにリサイズする関数"""
    return cv2.resize(image, (width, height))

def blend_images(background, overlay, alpha):
    """
    背景画像に透過エフェクトで画像を重ねる
    - background: 背景画像
    - overlay: 合成する画像
    - alpha: 透明度（0.0～1.0）
    """
    blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)
    return blended

def is_thumb_near_ring_joint(landmarks):
    """
    親指の先が薬指の第三関節に近いかを判定
    """
    thumb_tip = landmarks[4]
    ring_joint = landmarks[14]

    # 距離を計算
    distance = ((thumb_tip.x - ring_joint.x)**2 + (thumb_tip.y - ring_joint.y)**2)**0.5
    return distance < 0.05  # 閾値を設定

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームをRGB形式に変換してMediaPipeで処理
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # フレームサイズを取得
    frame_height, frame_width = frame.shape[:2]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # ランドマークを映像上に描画
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            if is_thumb_near_ring_joint(landmarks):
                # 条件を満たした場合、画像をリサイズしてフレームに合成
                resized_image = resize_image_to_frame(overlay_image, frame_width, frame_height)
                frame = blend_images(frame, resized_image, 0.7)  # 透明度70%で合成

    # ウィンドウにフレームを表示
    cv2.imshow('Domain Expansion Application', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()