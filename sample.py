import cv2
import mediapipe as mp
import random
import math
import time

# MediaPipe Poseのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# スコアの初期化
score = 0

# ランダムに赤い丸を表示するための初期位置
circle_x = random.randint(100, 500)
circle_y = random.randint(100, 400)
circle_radius = 30

# カメラからの映像をキャプチャ
cap = cv2.VideoCapture(0)

# ゲームの制限時間（秒）
game_duration = 30
start_time = time.time()

# タッチ判定
def is_touching_circle(point_x, point_y, circle_x, circle_y, circle_radius):
    distance = math.sqrt((point_x - circle_x) ** 2 + (point_y - circle_y) ** 2)
    return distance < circle_radius

def generate_new_circle_position(frame_width, frame_height, circle_radius):
    """ 新しい円の位置をランダムに生成 """
    return random.randint(circle_radius, frame_width - circle_radius), random.randint(circle_radius, frame_height - circle_radius)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像を水平方向に反転
    frame = cv2.flip(frame, 1)

    # 画像をRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ポーズのランドマークを検出
    results = pose.process(image)

    # 画像をBGRに戻す
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # スコアとタイマー表示領域の背景を描画
    cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)

    if results.pose_landmarks:
        # ランドマークを描画
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 検出された各ランドマークの座標を取得
        landmarks = results.pose_landmarks.landmark

        # 両手（左右の手首）と両足（左右の足首）の座標を取得
        keypoints = {
            'left_hand': (int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0])),
            'right_hand': (int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1]),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0])),
            'left_foot': (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0])),
            'right_foot': (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame.shape[1]),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]))
        }

        # スコア可算
        for key, (x, y) in keypoints.items():
            if is_touching_circle(x, y, circle_x, circle_y, circle_radius):
                score += 1
                circle_x, circle_y = generate_new_circle_position(frame.shape[1], frame.shape[0], circle_radius)

    # 赤い丸を描画
    cv2.circle(frame, (circle_x, circle_y), circle_radius, (0, 0, 255), -1)

    # スコアを表示
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 残り時間を計算
    elapsed_time = time.time() - start_time
    remaining_time = max(0, game_duration - int(elapsed_time))
    cv2.putText(frame, f'Time: {remaining_time}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # ゲーム終了条件
    if remaining_time <= 0:
        break

    # 画像を表示
    cv2.imshow('Full Body Tracking Game', frame)

    # キー入力をチェック
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ゲーム終了後にスコアを5秒間表示
end_time = time.time()  # ゲーム終了時刻を記録
while time.time() - end_time < 5:  # 5秒間ループ
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # ゲーム終了メッセージとスコアを表示
    cv2.putText(frame, f'Time Over!', (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Your Score: {score}', (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Full Body Tracking Game', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # 'q' を押したら即終了
        break

# リソース解放
cap.release()
cv2.destroyAllWindows()
